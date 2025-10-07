"""文档向量化管理器"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import time

from ..embedding.embedder import TextEmbedder
from .qdrant_client import QdrantVectorStore
from ..document.document_manager import document_manager
from ..document.chunker import TextChunker

logger = logging.getLogger(__name__)

class DocumentVectorizer:
    """文档向量化管理器"""
    
    def __init__(self, 
                 embedder: TextEmbedder,
                 vector_store: QdrantVectorStore,
                 collection_name: str = "documents",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化文档向量化管理器
        
        Args:
            embedder: 文本向量化器
            vector_store: 向量存储
            collection_name: 集合名称
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        
        # 初始化文档解析器和分块器
        self.parser = document_manager
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 确保集合存在
        self._ensure_collection_exists()
        
        # 处理日志
        self.processing_log = []
    
    def _ensure_collection_exists(self):
        """确保向量集合存在"""
        try:
            # 获取向量维度
            vector_size = self.embedder.get_vector_dimension()
            
            # 创建集合（如果不存在）
            success = self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=vector_size,
                distance="Cosine",
                recreate=False
            )
            
            if success:
                logger.info(f"集合 {self.collection_name} 准备就绪")
            else:
                raise Exception(f"无法创建或访问集合 {self.collection_name}")
                
        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            raise
    
    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成chunk的唯一ID"""
        # 使用文件路径和chunk索引生成唯一ID
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            处理结果统计
        """
        start_time = time.time()
        result = {
            'file_path': file_path,
            'success': False,
            'chunks_count': 0,
            'vectors_count': 0,
            'processing_time': 0,
            'error': None
        }
        
        try:
            logger.info(f"开始处理文档: {file_path}")
            
            # 1. 解析文档
            parsed_doc = self.parser.parse_document(file_path)
            text_content = parsed_doc.content
            if not text_content:
                raise ValueError("文档解析失败或内容为空")
            
            # 2. 文本分块
            chunks = self.chunker.chunk_text(text_content)
            result['chunks_count'] = len(chunks)
            
            if not chunks:
                raise ValueError("文本分块失败")
            
            logger.info(f"文档分块完成: {len(chunks)} 个块")
            
            # 3. 向量化
            vectors = self.embedder.encode_batch(chunks)
            result['vectors_count'] = len(vectors)
            
            logger.info(f"向量化完成: {len(vectors)} 个向量")
            
            # 4. 准备元数据
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            file_size = os.path.getsize(file_path)
            
            payloads = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(file_path, i)
                chunk_ids.append(chunk_id)
                
                payload = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_ext': file_ext,
                    'file_size': file_size,
                    'chunk_index': i,
                    'chunk_id': chunk_id,
                    'chunk_text': chunk,
                    'chunk_length': len(chunk),
                    'processed_at': datetime.now().isoformat()
                }
                payloads.append(payload)
            
            # 5. 存储向量
            success = self.vector_store.insert_vectors(
                collection_name=self.collection_name,
                vectors=vectors,
                payloads=payloads,
                ids=chunk_ids
            )
            
            if not success:
                raise Exception("向量存储失败")
            
            # 6. 更新结果
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            
            logger.info(f"文档处理完成: {file_path} ({result['processing_time']:.2f}s)")
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logger.error(f"文档处理失败 {file_path}: {e}")
        
        # 记录处理日志
        self.processing_log.append(result)
        return result
    
    def batch_process_directory(self, 
                               directory_path: str,
                               file_extensions: Optional[List[str]] = None,
                               recursive: bool = True) -> Dict[str, Any]:
        """
        批量处理目录中的文档
        
        Args:
            directory_path: 目录路径
            file_extensions: 支持的文件扩展名列表
            recursive: 是否递归处理子目录
            
        Returns:
            批量处理结果统计
        """
        start_time = time.time()
        
        # 默认支持的文件类型
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.pdf', '.docx', '.doc']
        
        # 收集文件
        files_to_process = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"目录不存在: {directory_path}")
        
        # 查找文件
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                files_to_process.append(str(file_path))
        
        logger.info(f"找到 {len(files_to_process)} 个文件待处理")
        
        # 批量处理结果
        batch_result = {
            'directory_path': directory_path,
            'total_files': len(files_to_process),
            'successful_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'processing_time': 0,
            'file_results': []
        }
        
        # 逐个处理文件
        for file_path in files_to_process:
            result = self.process_document(file_path)
            batch_result['file_results'].append(result)
            
            if result['success']:
                batch_result['successful_files'] += 1
                batch_result['total_chunks'] += result['chunks_count']
                batch_result['total_vectors'] += result['vectors_count']
            else:
                batch_result['failed_files'] += 1
        
        batch_result['processing_time'] = time.time() - start_time
        
        logger.info(f"批量处理完成: {batch_result['successful_files']}/{batch_result['total_files']} 文件成功")
        logger.info(f"总计: {batch_result['total_chunks']} 个文本块, {batch_result['total_vectors']} 个向量")
        logger.info(f"耗时: {batch_result['processing_time']:.2f}s")
        
        return batch_result
    
    def batch_process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        批量处理指定的文档列表
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            批量处理结果统计
        """
        start_time = time.time()
        
        batch_result = {
            'total_files': len(file_paths),
            'successful_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_vectors': 0,
            'processing_time': 0,
            'file_results': []
        }
        
        logger.info(f"开始批量处理 {len(file_paths)} 个文档")
        
        for file_path in file_paths:
            result = self.process_document(file_path)
            batch_result['file_results'].append(result)
            
            if result['success']:
                batch_result['successful_files'] += 1
                batch_result['total_chunks'] += result['chunks_count']
                batch_result['total_vectors'] += result['vectors_count']
            else:
                batch_result['failed_files'] += 1
        
        batch_result['processing_time'] = time.time() - start_time
        
        logger.info(f"批量处理完成: {batch_result['successful_files']}/{batch_result['total_files']} 文件成功")
        
        return batch_result
    
    def search_documents(self, 
                        query: str,
                        limit: int = 10,
                        score_threshold: Optional[float] = None,
                        file_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            score_threshold: 分数阈值
            file_filter: 文件过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 向量化查询
            query_vector = self.embedder.encode(query)
            
            # 执行搜索
            search_results = self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=file_filter
            )
            
            # 格式化结果
            results = []
            for result in search_results:
                formatted_result = {
                    'chunk_id': result.id,
                    'score': result.score,
                    'file_path': result.payload.get('file_path', ''),
                    'file_name': result.payload.get('file_name', ''),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'chunk_text': result.payload.get('chunk_text', ''),
                    'chunk_length': result.payload.get('chunk_length', 0)
                }
                results.append(formatted_result)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息
        """
        try:
            info = self.vector_store.get_collection_info(self.collection_name)
            if info:
                return {
                    'collection_name': self.collection_name,
                    'total_vectors': info['vectors_count'],
                    'indexed_vectors': info['indexed_vectors_count'],
                    'total_points': info['points_count'],
                    'vector_dimension': info['config']['vector_size'],
                    'distance_metric': info['config']['distance'],
                    'status': info['status']
                }
            else:
                return {'error': '无法获取集合信息'}
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {'error': str(e)}
    
    def save_processing_log(self, output_path: str):
        """
        保存处理日志
        
        Args:
            output_path: 输出文件路径
        """
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name,
                'total_processed': len(self.processing_log),
                'processing_log': self.processing_log
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理日志已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"保存处理日志失败: {e}")