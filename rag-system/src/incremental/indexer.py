#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量索引器 - IncrementalIndexer

实现高效的增量索引更新功能
只处理变更文档，避免全量重建
支持批量处理和并发更新

作者: RAG系统开发团队
日期: 2024-01-15
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入监控模块
try:
    from .monitoring import get_monitoring_manager
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@dataclass
class IndexEntry:
    """索引条目"""
    document_id: str
    file_path: str
    hash: str
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    indexed_at: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexEntry':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class IndexStats:
    """索引统计信息"""
    total_documents: int
    total_chunks: int
    total_embeddings: int
    index_size_mb: float
    last_update: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class IncrementalIndexer:
    """增量索引器
    
    功能:
    1. 增量更新文档索引
    2. 批量处理文档变更
    3. 并发索引处理
    4. 索引优化和压缩
    5. 索引统计和监控
    """
    
    def __init__(self,
                 index_dir: str = "incremental_index",
                 batch_size: int = 10,
                 max_workers: int = 4,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 config: Optional[Dict] = None):
        """
        初始化增量索引器
        
        Args:
            index_dir: 索引存储目录
            batch_size: 批处理大小
            max_workers: 最大工作线程数
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            config: 配置字典
        """
        self.index_dir = Path(index_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = config or {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 获取监控管理器
        self.monitoring = None
        if MONITORING_AVAILABLE:
            try:
                self.monitoring = get_monitoring_manager()
            except ValueError:
                # 监控管理器未初始化
                pass
        
        # 创建索引目录
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 索引存储
        self.index_file = self.index_dir / "index.json"
        self.stats_file = self.index_dir / "stats.json"
        
        # 加载现有索引
        self.index: Dict[str, IndexEntry] = self._load_index()
        self.stats = self._load_stats()
        
        # 运行时统计
        self.runtime_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0.0,
            'errors_encountered': 0
        }
    
    def process_changes(self, 
                       changes: Dict[str, List[str]], 
                       embedder=None,
                       force_reindex: bool = False) -> Dict[str, Any]:
        """
        处理文档变更
        
        Args:
            changes: 变更字典，包含added, modified, deleted列表
            embedder: 嵌入生成器
            force_reindex: 是否强制重新索引
            
        Returns:
            处理结果统计
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"开始处理文档变更: {sum(len(files) for files in changes.values())}个文件")
            
            # 使用监控计时器
            if self.monitoring:
                with self.monitoring.timer("incremental_indexing"):
                    result = self._perform_change_processing(changes, embedder, force_reindex)
            else:
                result = self._perform_change_processing(changes, embedder, force_reindex)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self.runtime_stats['processing_time'] += processing_time
            
            # 记录监控日志
            if self.monitoring:
                self.monitoring.log_operation(
                    "incremental_indexing",
                    {
                        "files_processed": sum(len(files) for files in changes.values()),
                        "documents_indexed": result.get('indexed', 0),
                        "documents_removed": result.get('removed', 0),
                        "processing_duration": processing_time,
                        "force_reindex": force_reindex
                    }
                )
                
                # 记录指标
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "documents_indexed", result.get('indexed', 0)
                )
                self.monitoring.performance_monitor.metrics_collector.record_metric(
                    "chunks_processed", result.get('chunks_created', 0)
                )
            
            self.logger.info(
                f"变更处理完成: 索引{result.get('indexed', 0)}个文档, "
                f"删除{result.get('removed', 0)}个文档, "
                f"耗时{processing_time:.2f}秒"
            )
            
            return result
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.handle_error(e, {
                    "operation": "process_changes",
                    "change_count": sum(len(files) for files in changes.values()),
                    "force_reindex": force_reindex
                })
            self.logger.error(f"变更处理失败: {e}")
            self.runtime_stats['errors_encountered'] += 1
            raise
    
    def _perform_change_processing(self, 
                                  changes: Dict[str, List[str]], 
                                  embedder,
                                  force_reindex: bool) -> Dict[str, Any]:
        """执行实际的变更处理"""
        result = {
            'indexed': 0,
            'removed': 0,
            'chunks_created': 0,
            'errors': []
        }
        
        # 处理删除的文档
        for file_path in changes.get('deleted', []):
            try:
                if self._remove_document(file_path):
                    result['removed'] += 1
            except Exception as e:
                error_msg = f"删除文档索引失败 {file_path}: {e}"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
        
        # 处理新增和修改的文档
        files_to_process = changes.get('added', []) + changes.get('modified', [])
        
        if files_to_process:
            # 批量处理
            for i in range(0, len(files_to_process), self.batch_size):
                batch = files_to_process[i:i + self.batch_size]
                batch_result = self._process_batch(batch, embedder, force_reindex)
                
                result['indexed'] += batch_result['indexed']
                result['chunks_created'] += batch_result['chunks_created']
                result['errors'].extend(batch_result['errors'])
        
        # 保存索引和统计信息
        self._save_index()
        self._update_stats()
        
        return result
    
    def _process_batch(self, 
                      file_paths: List[str], 
                      embedder,
                      force_reindex: bool) -> Dict[str, Any]:
        """
        批量处理文档
        
        Args:
            file_paths: 文件路径列表
            embedder: 嵌入生成器
            force_reindex: 是否强制重新索引
            
        Returns:
            批处理结果
        """
        result = {
            'indexed': 0,
            'chunks_created': 0,
            'errors': []
        }
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self._process_single_document, file_path, embedder, force_reindex): file_path
                for file_path in file_paths
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    doc_result = future.result()
                    if doc_result:
                        result['indexed'] += 1
                        result['chunks_created'] += doc_result.get('chunks_created', 0)
                        
                        # 记录监控指标
                        if self.monitoring:
                            self.monitoring.performance_monitor.metrics_collector.record_metric(
                                "document_processed", 1
                            )
                    
                except Exception as e:
                    error_msg = f"处理文档失败 {file_path}: {e}"
                    self.logger.error(error_msg)
                    result['errors'].append(error_msg)
                    
                    if self.monitoring:
                        self.monitoring.handle_error(e, {
                            "operation": "process_single_document",
                            "file_path": file_path
                        })
        
        return result
    
    def _process_single_document(self, 
                               file_path: str, 
                               embedder,
                               force_reindex: bool) -> Optional[Dict[str, Any]]:
        """
        处理单个文档
        
        Args:
            file_path: 文件路径
            embedder: 嵌入生成器
            force_reindex: 是否强制重新索引
            
        Returns:
            处理结果，失败返回None
        """
        try:
            # 读取文档内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文档哈希
            import hashlib
            doc_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要重新索引
            if not force_reindex and file_path in self.index:
                existing_entry = self.index[file_path]
                if existing_entry.hash == doc_hash:
                    # 文档未变更，跳过
                    return None
            
            # 文档分块
            chunks = self._chunk_document(content, file_path)
            
            # 生成嵌入
            embeddings = []
            if embedder:
                for chunk in chunks:
                    try:
                        embedding = embedder.embed_text(chunk['text'])
                        embeddings.append(embedding)
                    except Exception as e:
                        self.logger.warning(f"生成嵌入失败 {file_path}: {e}")
                        # 使用零向量作为占位符
                        embeddings.append([0.0] * 768)  # 假设768维
            
            # 创建索引条目
            document_id = f"doc_{doc_hash[:8]}"
            index_entry = IndexEntry(
                document_id=document_id,
                file_path=file_path,
                hash=doc_hash,
                chunks=chunks,
                embeddings=embeddings,
                metadata={
                    'file_size': len(content),
                    'chunk_count': len(chunks),
                    'embedding_count': len(embeddings)
                },
                indexed_at=datetime.now().isoformat()
            )
            
            # 更新索引
            self.index[file_path] = index_entry
            
            # 更新运行时统计
            self.runtime_stats['documents_processed'] += 1
            self.runtime_stats['chunks_created'] += len(chunks)
            self.runtime_stats['embeddings_generated'] += len(embeddings)
            
            return {
                'chunks_created': len(chunks),
                'embeddings_generated': len(embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"处理文档失败 {file_path}: {e}")
            return None

    def _load_index(self) -> Dict[str, IndexEntry]:
        """加载现有索引"""
        if not self.index_file.exists():
            return {}
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为IndexEntry对象
            index = {}
            for file_path, entry_data in data.items():
                index[file_path] = IndexEntry.from_dict(entry_data)
            
            return index
        except Exception as e:
            self.logger.warning(f"加载索引失败: {e}")
            return {}
    
    def _load_stats(self) -> IndexStats:
        """加载索引统计信息"""
        if not self.stats_file.exists():
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                index_size_mb=0.0,
                last_update=datetime.now().isoformat()
            )
        
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return IndexStats(**data)
        except Exception as e:
            self.logger.warning(f"加载统计信息失败: {e}")
            return IndexStats(
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                index_size_mb=0.0,
                last_update=datetime.now().isoformat()
            )
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            # 转换为可序列化的字典
            data = {}
            for file_path, entry in self.index.items():
                data[file_path] = entry.to_dict()
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
    
    def _update_stats(self):
        """更新统计信息"""
        try:
            total_documents = len(self.index)
            total_chunks = sum(len(entry.chunks) for entry in self.index.values())
            total_embeddings = sum(len(entry.embeddings) for entry in self.index.values())
            
            # 计算索引文件大小
            index_size_mb = 0.0
            if self.index_file.exists():
                index_size_mb = self.index_file.stat().st_size / (1024 * 1024)
            
            self.stats = IndexStats(
                total_documents=total_documents,
                total_chunks=total_chunks,
                total_embeddings=total_embeddings,
                index_size_mb=index_size_mb,
                last_update=datetime.now().isoformat()
            )
            
            # 保存统计信息
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
    
    def _remove_document(self, file_path: str) -> bool:
        """从索引中移除文档"""
        if file_path in self.index:
            del self.index[file_path]
            return True
        return False
    
    def _chunk_document(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """文档分块"""
        chunks = []
        
        # 简单的分块策略：按段落分割
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        'chunk_id': f"{file_path}_{chunk_id}",
                        'text': current_chunk.strip(),
                        'start_pos': 0,  # 简化处理
                        'end_pos': len(current_chunk),
                        'metadata': {
                            'file_path': file_path,
                            'chunk_index': chunk_id
                        }
                    })
                    chunk_id += 1
                
                current_chunk = paragraph + "\n\n"
        
        # 处理最后一个分块
        if current_chunk.strip():
            chunks.append({
                'chunk_id': f"{file_path}_{chunk_id}",
                'text': current_chunk.strip(),
                'start_pos': 0,
                'end_pos': len(current_chunk),
                'metadata': {
                    'file_path': file_path,
                    'chunk_index': chunk_id
                }
            })
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        stats_dict = self.stats.to_dict()
        stats_dict.update(self.runtime_stats)
        return stats_dict
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        # 简化的相似度搜索实现
        results = []
        
        for file_path, entry in self.index.items():
            for i, embedding in enumerate(entry.embeddings):
                # 计算余弦相似度（简化版）
                similarity = sum(a * b for a, b in zip(query_embedding, embedding))
                
                results.append({
                    'file_path': file_path,
                    'chunk_id': entry.chunks[i]['chunk_id'],
                    'text': entry.chunks[i]['text'],
                    'similarity': similarity,
                    'metadata': entry.chunks[i]['metadata']
                })
        
        # 按相似度排序并返回top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]