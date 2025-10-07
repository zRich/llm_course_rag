#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson13 向量数据库优化器实现模板
解决索引优化、查询性能和存储管理缺失问题

功能特性：
1. 多种索引策略（HNSW、IVF、LSH等）
2. 查询性能优化和缓存机制
3. 存储空间管理和压缩
4. 向量质量评估和清理
5. 实时监控和性能分析
"""

import logging
import time
import json
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import psutil
import gc

# 向量数据库相关
try:
    import faiss
except ImportError:
    faiss = None
    
try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import chromadb
except ImportError:
    chromadb = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VectorMetadata:
    """向量元数据"""
    vector_id: str
    document_id: str
    chunk_id: str
    content: str
    embedding_model: str
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0
    quality_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict = field(default_factory=dict)

@dataclass
class IndexConfig:
    """索引配置"""
    index_type: str  # 'hnsw', 'ivf', 'flat', 'lsh'
    dimension: int
    metric: str = 'cosine'  # 'cosine', 'l2', 'ip'
    parameters: Dict = field(default_factory=dict)
    
@dataclass
class QueryResult:
    """查询结果"""
    vector_id: str
    score: float
    metadata: VectorMetadata
    distance: float
    rank: int

@dataclass
class PerformanceMetrics:
    """性能指标"""
    query_time: float
    index_time: float
    memory_usage: float
    disk_usage: float
    cache_hit_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)

class VectorCache:
    """向量缓存管理器"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # 检查TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                self._remove(key)
                return None
            
            # 更新访问信息
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """添加缓存项"""
        with self._lock:
            # 检查容量
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
    
    def _remove(self, key: str) -> None:
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def _evict_lru(self) -> None:
        """LRU淘汰策略"""
        if not self.cache:
            return
        
        # 找到最少使用的项
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: (self.access_counts[k], self.access_times[k]))
        self._remove(lru_key)
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        with self._lock:
            total_accesses = sum(self.access_counts.values())
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': len(self.cache) / max(total_accesses, 1),
                'total_accesses': total_accesses
            }

class VectorIndex:
    """向量索引基类"""
    
    def __init__(self, config: IndexConfig):
        self.config = config
        self.index = None
        self.metadata_store = {}
        self.vector_count = 0
        self._lock = threading.RLock()
    
    def build_index(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """构建索引（需要子类实现）"""
        raise NotImplementedError
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """添加向量（需要子类实现）"""
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[QueryResult]:
        """搜索向量（需要子类实现）"""
        raise NotImplementedError
    
    def remove_vectors(self, vector_ids: List[str]) -> None:
        """移除向量（需要子类实现）"""
        raise NotImplementedError
    
    def save_index(self, file_path: str) -> None:
        """保存索引（需要子类实现）"""
        raise NotImplementedError
    
    def load_index(self, file_path: str) -> None:
        """加载索引（需要子类实现）"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'index_type': self.config.index_type,
            'dimension': self.config.dimension,
            'vector_count': self.vector_count,
            'memory_usage': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        base_size = self.vector_count * self.config.dimension * 4  # float32
        metadata_size = len(pickle.dumps(self.metadata_store))
        return (base_size + metadata_size) / (1024 * 1024)

class HNSWIndex(VectorIndex):
    """HNSW索引实现"""
    
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        if hnswlib is None:
            raise ImportError("需要安装hnswlib: pip install hnswlib")
        
        # HNSW参数
        self.M = config.parameters.get('M', 16)
        self.ef_construction = config.parameters.get('ef_construction', 200)
        self.ef_search = config.parameters.get('ef_search', 50)
        self.max_elements = config.parameters.get('max_elements', 100000)
        
        # 创建索引
        space = 'cosine' if config.metric == 'cosine' else 'l2'
        self.index = hnswlib.Index(space=space, dim=config.dimension)
        self.index.init_index(
            max_elements=self.max_elements,
            M=self.M,
            ef_construction=self.ef_construction
        )
        self.index.set_ef(self.ef_search)
        
        self.id_to_vector_id = {}
        self.vector_id_to_id = {}
        self.next_id = 0
    
    def build_index(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """构建HNSW索引"""
        with self._lock:
            start_time = time.time()
            
            # 重新初始化索引
            space = 'cosine' if self.config.metric == 'cosine' else 'l2'
            self.index = hnswlib.Index(space=space, dim=self.config.dimension)
            self.index.init_index(
                max_elements=max(len(vectors), self.max_elements),
                M=self.M,
                ef_construction=self.ef_construction
            )
            
            # 添加向量
            ids = list(range(len(vectors)))
            self.index.add_items(vectors, ids)
            
            # 更新映射和元数据
            for i, metadata in enumerate(metadata_list):
                self.id_to_vector_id[i] = metadata.vector_id
                self.vector_id_to_id[metadata.vector_id] = i
                self.metadata_store[metadata.vector_id] = metadata
            
            self.vector_count = len(vectors)
            self.next_id = len(vectors)
            
            build_time = time.time() - start_time
            logger.info(f"HNSW索引构建完成: {len(vectors)}个向量, 耗时{build_time:.2f}秒")
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """添加向量到HNSW索引"""
        with self._lock:
            if len(vectors) == 0:
                return
            
            # 检查容量
            if self.next_id + len(vectors) > self.max_elements:
                logger.warning("索引容量不足，需要重建索引")
                # 可以选择重建索引或扩容
                return
            
            # 添加向量
            ids = list(range(self.next_id, self.next_id + len(vectors)))
            self.index.add_items(vectors, ids)
            
            # 更新映射和元数据
            for i, metadata in enumerate(metadata_list):
                internal_id = self.next_id + i
                self.id_to_vector_id[internal_id] = metadata.vector_id
                self.vector_id_to_id[metadata.vector_id] = internal_id
                self.metadata_store[metadata.vector_id] = metadata
            
            self.vector_count += len(vectors)
            self.next_id += len(vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[QueryResult]:
        """搜索最相似的向量"""
        with self._lock:
            if self.vector_count == 0:
                return []
            
            # 执行搜索
            labels, distances = self.index.knn_query(query_vector, k=min(k, self.vector_count))
            
            # 转换结果
            results = []
            for rank, (label, distance) in enumerate(zip(labels[0], distances[0])):
                vector_id = self.id_to_vector_id.get(label)
                if vector_id and vector_id in self.metadata_store:
                    metadata = self.metadata_store[vector_id]
                    # 更新访问信息
                    metadata.last_accessed = datetime.now()
                    metadata.access_count += 1
                    
                    # 计算相似度分数
                    if self.config.metric == 'cosine':
                        score = 1.0 - distance  # cosine distance to similarity
                    else:
                        score = 1.0 / (1.0 + distance)  # L2 distance to similarity
                    
                    results.append(QueryResult(
                        vector_id=vector_id,
                        score=score,
                        metadata=metadata,
                        distance=distance,
                        rank=rank
                    ))
            
            return results
    
    def remove_vectors(self, vector_ids: List[str]) -> None:
        """移除向量（HNSW不支持删除，需要重建）"""
        logger.warning("HNSW索引不支持删除操作，需要重建索引")
        # 可以标记为删除，在重建时过滤
        for vector_id in vector_ids:
            if vector_id in self.metadata_store:
                del self.metadata_store[vector_id]
                if vector_id in self.vector_id_to_id:
                    internal_id = self.vector_id_to_id[vector_id]
                    del self.vector_id_to_id[vector_id]
                    del self.id_to_vector_id[internal_id]
    
    def save_index(self, file_path: str) -> None:
        """保存HNSW索引"""
        with self._lock:
            # 保存索引文件
            self.index.save_index(file_path)
            
            # 保存元数据
            metadata_path = file_path + ".metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'id_to_vector_id': self.id_to_vector_id,
                    'vector_id_to_id': self.vector_id_to_id,
                    'vector_count': self.vector_count,
                    'next_id': self.next_id,
                    'config': self.config
                }, f)
    
    def load_index(self, file_path: str) -> None:
        """加载HNSW索引"""
        with self._lock:
            # 加载索引文件
            space = 'cosine' if self.config.metric == 'cosine' else 'l2'
            self.index = hnswlib.Index(space=space, dim=self.config.dimension)
            self.index.load_index(file_path)
            
            # 加载元数据
            metadata_path = file_path + ".metadata"
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.id_to_vector_id = data['id_to_vector_id']
                self.vector_id_to_id = data['vector_id_to_id']
                self.vector_count = data['vector_count']
                self.next_id = data['next_id']

class FaissIndex(VectorIndex):
    """Faiss索引实现"""
    
    def __init__(self, config: IndexConfig):
        super().__init__(config)
        if faiss is None:
            raise ImportError("需要安装faiss: pip install faiss-cpu 或 faiss-gpu")
        
        # 创建索引
        self.index = self._create_faiss_index()
        self.vector_id_to_idx = {}
        self.idx_to_vector_id = {}
    
    def _create_faiss_index(self):
        """创建Faiss索引"""
        d = self.config.dimension
        
        if self.config.index_type == 'flat':
            if self.config.metric == 'cosine':
                index = faiss.IndexFlatIP(d)  # Inner Product for cosine
            else:
                index = faiss.IndexFlatL2(d)
        elif self.config.index_type == 'ivf':
            nlist = self.config.parameters.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(d)
            if self.config.metric == 'cosine':
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        else:
            # 默认使用Flat索引
            index = faiss.IndexFlatL2(d)
        
        return index
    
    def build_index(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """构建Faiss索引"""
        with self._lock:
            start_time = time.time()
            
            # 预处理向量
            if self.config.metric == 'cosine':
                # 归一化向量用于余弦相似度
                faiss.normalize_L2(vectors)
            
            # 训练索引（如果需要）
            if hasattr(self.index, 'train'):
                self.index.train(vectors)
            
            # 添加向量
            self.index.add(vectors)
            
            # 更新映射和元数据
            for i, metadata in enumerate(metadata_list):
                self.vector_id_to_idx[metadata.vector_id] = i
                self.idx_to_vector_id[i] = metadata.vector_id
                self.metadata_store[metadata.vector_id] = metadata
            
            self.vector_count = len(vectors)
            
            build_time = time.time() - start_time
            logger.info(f"Faiss索引构建完成: {len(vectors)}个向量, 耗时{build_time:.2f}秒")
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """添加向量到Faiss索引"""
        with self._lock:
            if len(vectors) == 0:
                return
            
            # 预处理向量
            if self.config.metric == 'cosine':
                faiss.normalize_L2(vectors)
            
            # 添加向量
            start_idx = self.index.ntotal
            self.index.add(vectors)
            
            # 更新映射和元数据
            for i, metadata in enumerate(metadata_list):
                idx = start_idx + i
                self.vector_id_to_idx[metadata.vector_id] = idx
                self.idx_to_vector_id[idx] = metadata.vector_id
                self.metadata_store[metadata.vector_id] = metadata
            
            self.vector_count = self.index.ntotal
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[QueryResult]:
        """搜索最相似的向量"""
        with self._lock:
            if self.vector_count == 0:
                return []
            
            # 预处理查询向量
            query = query_vector.reshape(1, -1).astype(np.float32)
            if self.config.metric == 'cosine':
                faiss.normalize_L2(query)
            
            # 执行搜索
            distances, indices = self.index.search(query, min(k, self.vector_count))
            
            # 转换结果
            results = []
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx == -1:  # 无效结果
                    continue
                
                vector_id = self.idx_to_vector_id.get(idx)
                if vector_id and vector_id in self.metadata_store:
                    metadata = self.metadata_store[vector_id]
                    # 更新访问信息
                    metadata.last_accessed = datetime.now()
                    metadata.access_count += 1
                    
                    # 计算相似度分数
                    if self.config.metric == 'cosine':
                        score = distance  # Inner product (已归一化)
                    else:
                        score = 1.0 / (1.0 + distance)  # L2 distance to similarity
                    
                    results.append(QueryResult(
                        vector_id=vector_id,
                        score=score,
                        metadata=metadata,
                        distance=distance,
                        rank=rank
                    ))
            
            return results
    
    def remove_vectors(self, vector_ids: List[str]) -> None:
        """移除向量（Faiss不支持删除，需要重建）"""
        logger.warning("Faiss索引不支持删除操作，需要重建索引")
        # 标记为删除
        for vector_id in vector_ids:
            if vector_id in self.metadata_store:
                del self.metadata_store[vector_id]
                if vector_id in self.vector_id_to_idx:
                    idx = self.vector_id_to_idx[vector_id]
                    del self.vector_id_to_idx[vector_id]
                    del self.idx_to_vector_id[idx]
    
    def save_index(self, file_path: str) -> None:
        """保存Faiss索引"""
        with self._lock:
            # 保存索引文件
            faiss.write_index(self.index, file_path)
            
            # 保存元数据
            metadata_path = file_path + ".metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'vector_id_to_idx': self.vector_id_to_idx,
                    'idx_to_vector_id': self.idx_to_vector_id,
                    'vector_count': self.vector_count,
                    'config': self.config
                }, f)
    
    def load_index(self, file_path: str) -> None:
        """加载Faiss索引"""
        with self._lock:
            # 加载索引文件
            self.index = faiss.read_index(file_path)
            
            # 加载元数据
            metadata_path = file_path + ".metadata"
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.vector_id_to_idx = data['vector_id_to_idx']
                self.idx_to_vector_id = data['idx_to_vector_id']
                self.vector_count = data['vector_count']

class VectorDatabaseOptimizer:
    """向量数据库优化器"""
    
    def __init__(self, index_config: IndexConfig, cache_size: int = 10000):
        self.index_config = index_config
        self.index = self._create_index()
        self.cache = VectorCache(max_size=cache_size)
        self.performance_history = deque(maxlen=1000)
        self.optimization_rules = []
        self._lock = threading.RLock()
        
        # 性能监控
        self.query_count = 0
        self.total_query_time = 0.0
        self.last_optimization = datetime.now()
    
    def _create_index(self) -> VectorIndex:
        """创建向量索引"""
        if self.index_config.index_type == 'hnsw':
            return HNSWIndex(self.index_config)
        elif self.index_config.index_type in ['flat', 'ivf']:
            return FaissIndex(self.index_config)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_config.index_type}")
    
    def build_index(self, vectors: np.ndarray, metadata_list: List[VectorMetadata]) -> None:
        """构建优化的索引"""
        with self._lock:
            start_time = time.time()
            
            # 向量质量评估
            quality_scores = self._evaluate_vector_quality(vectors)
            for i, metadata in enumerate(metadata_list):
                metadata.quality_score = quality_scores[i]
            
            # 构建索引
            self.index.build_index(vectors, metadata_list)
            
            # 记录性能指标
            build_time = time.time() - start_time
            metrics = PerformanceMetrics(
                query_time=0.0,
                index_time=build_time,
                memory_usage=self._get_memory_usage(),
                disk_usage=0.0,
                cache_hit_rate=0.0,
                throughput=len(vectors) / build_time
            )
            self.performance_history.append(metrics)
            
            logger.info(f"索引构建完成: {len(vectors)}个向量, 耗时{build_time:.2f}秒")
    
    def search_optimized(self, query_vector: np.ndarray, k: int = 10, 
                        use_cache: bool = True) -> List[QueryResult]:
        """优化的搜索"""
        start_time = time.time()
        
        # 生成查询缓存键
        query_key = None
        if use_cache:
            query_key = hashlib.md5(query_vector.tobytes()).hexdigest()
            cached_result = self.cache.get(query_key)
            if cached_result:
                return cached_result
        
        # 执行搜索
        results = self.index.search(query_vector, k)
        
        # 后处理优化
        results = self._post_process_results(results)
        
        # 缓存结果
        if use_cache and query_key:
            self.cache.put(query_key, results)
        
        # 更新性能指标
        query_time = time.time() - start_time
        self.query_count += 1
        self.total_query_time += query_time
        
        # 记录性能指标
        cache_stats = self.cache.get_stats()
        metrics = PerformanceMetrics(
            query_time=query_time,
            index_time=0.0,
            memory_usage=self._get_memory_usage(),
            disk_usage=0.0,
            cache_hit_rate=cache_stats['hit_rate'],
            throughput=1.0 / query_time
        )
        self.performance_history.append(metrics)
        
        return results
    
    def _evaluate_vector_quality(self, vectors: np.ndarray) -> List[float]:
        """评估向量质量"""
        quality_scores = []
        
        for vector in vectors:
            # 计算向量的各种质量指标
            norm = np.linalg.norm(vector)
            sparsity = np.count_nonzero(vector) / len(vector)
            variance = np.var(vector)
            
            # 综合质量分数
            quality = 0.4 * min(norm, 1.0) + 0.3 * sparsity + 0.3 * min(variance, 1.0)
            quality_scores.append(quality)
        
        return quality_scores
    
    def _post_process_results(self, results: List[QueryResult]) -> List[QueryResult]:
        """后处理搜索结果"""
        # 质量过滤
        filtered_results = []
        for result in results:
            if result.metadata.quality_score > 0.3:  # 质量阈值
                filtered_results.append(result)
        
        # 多样性优化
        diverse_results = self._diversify_results(filtered_results)
        
        return diverse_results
    
    def _diversify_results(self, results: List[QueryResult]) -> List[QueryResult]:
        """结果多样性优化"""
        if len(results) <= 1:
            return results
        
        # 简单的多样性策略：基于内容相似度去重
        diverse_results = [results[0]]  # 保留最相似的结果
        
        for result in results[1:]:
            is_diverse = True
            for existing in diverse_results:
                # 简单的内容相似度检查
                if self._content_similarity(result.metadata.content, 
                                          existing.metadata.content) > 0.8:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        # 简单的Jaccard相似度
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    def optimize_index(self) -> None:
        """索引优化"""
        with self._lock:
            logger.info("开始索引优化...")
            
            # 清理低质量向量
            self._cleanup_low_quality_vectors()
            
            # 内存优化
            self._optimize_memory()
            
            # 缓存优化
            self._optimize_cache()
            
            self.last_optimization = datetime.now()
            logger.info("索引优化完成")
    
    def _cleanup_low_quality_vectors(self) -> None:
        """清理低质量向量"""
        low_quality_ids = []
        
        for vector_id, metadata in self.index.metadata_store.items():
            # 基于质量分数和访问频率决定是否清理
            if (metadata.quality_score < 0.2 and 
                metadata.access_count < 5 and
                (datetime.now() - metadata.last_accessed).days > 30):
                low_quality_ids.append(vector_id)
        
        if low_quality_ids:
            logger.info(f"清理 {len(low_quality_ids)} 个低质量向量")
            self.index.remove_vectors(low_quality_ids)
    
    def _optimize_memory(self) -> None:
        """内存优化"""
        # 强制垃圾回收
        gc.collect()
        
        # 检查内存使用情况
        memory_usage = self._get_memory_usage()
        if memory_usage > 1000:  # 超过1GB
            logger.warning(f"内存使用过高: {memory_usage:.2f}MB")
            # 可以考虑压缩索引或清理缓存
    
    def _optimize_cache(self) -> None:
        """缓存优化"""
        cache_stats = self.cache.get_stats()
        
        # 如果命中率过低，清理缓存
        if cache_stats['hit_rate'] < 0.1:
            logger.info("缓存命中率过低，清理缓存")
            self.cache.clear()
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        if not self.performance_history:
            return {}
        
        recent_metrics = list(self.performance_history)[-100:]  # 最近100次
        
        avg_query_time = sum(m.query_time for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        
        return {
            'total_queries': self.query_count,
            'avg_query_time': avg_query_time,
            'avg_memory_usage': avg_memory_usage,
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'avg_throughput': avg_throughput,
            'index_stats': self.index.get_stats(),
            'cache_stats': self.cache.get_stats(),
            'last_optimization': self.last_optimization.isoformat()
        }
    
    def save_optimizer(self, base_path: str) -> None:
        """保存优化器状态"""
        # 保存索引
        index_path = f"{base_path}.index"
        self.index.save_index(index_path)
        
        # 保存优化器状态
        optimizer_path = f"{base_path}.optimizer"
        with open(optimizer_path, 'wb') as f:
            pickle.dump({
                'index_config': self.index_config,
                'performance_history': list(self.performance_history),
                'query_count': self.query_count,
                'total_query_time': self.total_query_time,
                'last_optimization': self.last_optimization
            }, f)
    
    def load_optimizer(self, base_path: str) -> None:
        """加载优化器状态"""
        # 加载索引
        index_path = f"{base_path}.index"
        self.index.load_index(index_path)
        
        # 加载优化器状态
        optimizer_path = f"{base_path}.optimizer"
        with open(optimizer_path, 'rb') as f:
            data = pickle.load(f)
            self.performance_history = deque(data['performance_history'], maxlen=1000)
            self.query_count = data['query_count']
            self.total_query_time = data['total_query_time']
            self.last_optimization = data['last_optimization']

def main():
    """示例用法"""
    # 创建索引配置
    config = IndexConfig(
        index_type='hnsw',
        dimension=768,
        metric='cosine',
        parameters={
            'M': 16,
            'ef_construction': 200,
            'ef_search': 50,
            'max_elements': 100000
        }
    )
    
    # 创建优化器
    optimizer = VectorDatabaseOptimizer(config, cache_size=5000)
    
    # 生成示例数据
    n_vectors = 1000
    vectors = np.random.random((n_vectors, config.dimension)).astype(np.float32)
    
    metadata_list = []
    for i in range(n_vectors):
        metadata = VectorMetadata(
            vector_id=f"vec_{i}",
            document_id=f"doc_{i//10}",
            chunk_id=f"chunk_{i}",
            content=f"这是第{i}个文档块的内容",
            embedding_model="text-embedding-ada-002",
            created_time=datetime.now(),
            last_accessed=datetime.now()
        )
        metadata_list.append(metadata)
    
    # 构建索引
    print("构建索引...")
    optimizer.build_index(vectors, metadata_list)
    
    # 执行搜索测试
    print("\n执行搜索测试...")
    query_vector = np.random.random((config.dimension,)).astype(np.float32)
    
    for i in range(10):
        results = optimizer.search_optimized(query_vector, k=5)
        print(f"搜索 {i+1}: 找到 {len(results)} 个结果")
        
        if results:
            best_result = results[0]
            print(f"  最佳匹配: {best_result.vector_id}, 分数: {best_result.score:.4f}")
    
    # 性能报告
    print("\n性能报告:")
    report = optimizer.get_performance_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # 索引优化
    print("\n执行索引优化...")
    optimizer.optimize_index()
    
    print("示例完成！")

if __name__ == "__main__":
    main()