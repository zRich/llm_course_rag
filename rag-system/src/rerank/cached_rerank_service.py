#!/usr/bin/env python3
"""
缓存重排序服务模块

提供带缓存功能的重排序服务，包括：
- 查询结果缓存
- 重排序结果缓存
- 缓存失效策略
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from .rerank_service import RerankService

logger = logging.getLogger(__name__)

class CachedRerankService(RerankService):
    """带缓存的重排序服务类"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_ttl: int = 3600):
        """
        初始化缓存重排序服务
        
        Args:
            model_name: 用于重排序的模型名称
            cache_ttl: 缓存生存时间（秒）
        """
        super().__init__(model_name)
        self.cache_ttl = cache_ttl
        self.cache = {}  # 简单的内存缓存
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _generate_cache_key(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """生成缓存键"""
        # 创建查询和文档内容的哈希
        content = {
            'query': query,
            'doc_contents': [doc.get('content', '')[:100] for doc in documents]  # 只取前100字符
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否有效"""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < timedelta(seconds=self.cache_ttl)
    
    def _clean_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if 'timestamp' in entry:
                cache_time = datetime.fromisoformat(entry['timestamp'])
                if current_time - cache_time >= timedelta(seconds=self.cache_ttl):
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        scores: List[float],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        带缓存的重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            scores: 原始分数列表
            top_k: 返回的文档数量
        
        Returns:
            重排序后的文档列表和分数列表
        """
        self.cache_stats['total_requests'] += 1
        
        if not documents:
            return documents, scores
        
        # 生成缓存键
        cache_key = self._generate_cache_key(query, documents)
        
        # 检查缓存
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.cache_stats['hits'] += 1
            cached_result = self.cache[cache_key]
            logger.info(f"缓存命中: {cache_key[:8]}...")
            
            # 从缓存返回结果
            cached_docs = cached_result['documents']
            cached_scores = cached_result['scores']
            
            if top_k:
                return cached_docs[:top_k], cached_scores[:top_k]
            return cached_docs, cached_scores
        
        # 缓存未命中，执行重排序
        self.cache_stats['misses'] += 1
        logger.info(f"缓存未命中: {cache_key[:8]}...")
        
        reranked_docs, reranked_scores = super().rerank_documents(
            query, documents, scores, top_k=None  # 不在这里限制top_k，缓存完整结果
        )
        
        # 存储到缓存
        self.cache[cache_key] = {
            'documents': reranked_docs,
            'scores': reranked_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        # 定期清理过期缓存
        if self.cache_stats['total_requests'] % 100 == 0:
            self._clean_expired_cache()
        
        # 返回结果（应用top_k限制）
        if top_k:
            return reranked_docs[:top_k], reranked_scores[:top_k]
        return reranked_docs, reranked_scores
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_stats['total_requests']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self.cache),
            'cache_ttl_seconds': self.cache_ttl
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存详细信息"""
        cache_info = []
        current_time = datetime.now()
        
        for key, entry in self.cache.items():
            if 'timestamp' in entry:
                cache_time = datetime.fromisoformat(entry['timestamp'])
                age_seconds = (current_time - cache_time).total_seconds()
                is_valid = age_seconds < self.cache_ttl
                
                cache_info.append({
                    'key': key[:8] + '...',
                    'age_seconds': round(age_seconds, 2),
                    'is_valid': is_valid,
                    'document_count': len(entry.get('documents', []))
                })
        
        return {
            'cache_entries': cache_info,
            'stats': self.get_cache_stats()
        }
    
    def set_cache_ttl(self, ttl: int):
        """设置缓存生存时间"""
        self.cache_ttl = ttl
        logger.info(f"缓存TTL设置为 {ttl} 秒")
    
    def preload_cache(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        scores_list: List[List[float]]
    ):
        """预加载缓存"""
        logger.info(f"开始预加载 {len(queries)} 个查询的缓存")
        
        for query, documents, scores in zip(queries, documents_list, scores_list):
            # 执行重排序并缓存结果
            self.rerank_documents(query, documents, scores)
        
        logger.info(f"缓存预加载完成，当前缓存大小: {len(self.cache)}")