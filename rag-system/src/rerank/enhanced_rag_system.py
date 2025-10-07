#!/usr/bin/env python3
"""
增强RAG系统模块

提供完整的增强RAG查询系统，包括：
- 标准RAG与重排序RAG对比
- 批量查询处理
- 元数据过滤
- 性能监控
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..rag.rag_service import RAGService
from .rerank_service import RerankService
from .cached_rerank_service import CachedRerankService

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    """增强RAG系统类"""
    
    def __init__(
        self,
        rag_service: RAGService,
        use_cache: bool = True,
        rerank_model: str = "all-MiniLM-L6-v2",
        cache_ttl: int = 3600
    ):
        """
        初始化增强RAG系统
        
        Args:
            rag_service: RAG服务实例
            use_cache: 是否使用缓存
            rerank_model: 重排序模型名称
            cache_ttl: 缓存生存时间（秒）
        """
        self.rag_service = rag_service
        
        # 初始化重排序服务
        if use_cache:
            self.rerank_service = CachedRerankService(rerank_model, cache_ttl)
        else:
            self.rerank_service = RerankService(rerank_model)
        
        self.performance_stats = {
            'total_queries': 0,
            'standard_queries': 0,
            'reranked_queries': 0,
            'avg_standard_time': 0.0,
            'avg_rerank_time': 0.0,
            'total_standard_time': 0.0,
            'total_rerank_time': 0.0
        }
    
    def query_standard(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        执行标准RAG查询
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
        
        Returns:
            查询结果字典
        """
        start_time = time.time()
        
        try:
            # 执行标准RAG查询
            results = self.rag_service.query(query, top_k=top_k)
            
            query_time = time.time() - start_time
            
            # 更新统计信息
            self.performance_stats['total_queries'] += 1
            self.performance_stats['standard_queries'] += 1
            self.performance_stats['total_standard_time'] += query_time
            self.performance_stats['avg_standard_time'] = (
                self.performance_stats['total_standard_time'] / 
                self.performance_stats['standard_queries']
            )
            
            return {
                'query': query,
                'method': 'standard',
                'results': results,
                'query_time': query_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"标准查询失败: {str(e)}")
            return {
                'query': query,
                'method': 'standard',
                'error': str(e),
                'query_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def query_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 20
    ) -> Dict[str, Any]:
        """
        执行带重排序的RAG查询
        
        Args:
            query: 查询文本
            top_k: 最终返回的文档数量
            retrieve_k: 初始检索的文档数量
        
        Returns:
            查询结果字典
        """
        start_time = time.time()
        
        try:
            # 1. 执行初始检索（获取更多文档）
            initial_results = self.rag_service.query(query, top_k=retrieve_k)
            
            if not initial_results or 'documents' not in initial_results:
                return {
                    'query': query,
                    'method': 'reranked',
                    'error': '初始检索未返回文档',
                    'query_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            
            documents = initial_results['documents']
            scores = initial_results.get('scores', [1.0] * len(documents))
            
            # 2. 执行重排序
            reranked_docs, reranked_scores = self.rerank_service.rerank_documents(
                query, documents, scores, top_k=top_k
            )
            
            query_time = time.time() - start_time
            
            # 更新统计信息
            self.performance_stats['total_queries'] += 1
            self.performance_stats['reranked_queries'] += 1
            self.performance_stats['total_rerank_time'] += query_time
            self.performance_stats['avg_rerank_time'] = (
                self.performance_stats['total_rerank_time'] / 
                self.performance_stats['reranked_queries']
            )
            
            # 构建结果
            reranked_results = {
                'documents': reranked_docs,
                'scores': reranked_scores,
                'total_documents': len(reranked_docs)
            }
            
            return {
                'query': query,
                'method': 'reranked',
                'results': reranked_results,
                'initial_retrieved': len(documents),
                'final_returned': len(reranked_docs),
                'query_time': query_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"重排序查询失败: {str(e)}")
            return {
                'query': query,
                'method': 'reranked',
                'error': str(e),
                'query_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def compare_methods(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 20
    ) -> Dict[str, Any]:
        """
        对比标准RAG和重排序RAG的结果
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            retrieve_k: 重排序时初始检索的文档数量
        
        Returns:
            对比结果字典
        """
        logger.info(f"开始对比查询: {query}")
        
        # 执行标准查询
        standard_result = self.query_standard(query, top_k)
        
        # 执行重排序查询
        reranked_result = self.query_with_rerank(query, top_k, retrieve_k)
        
        # 计算改进指标
        improvement_metrics = self._calculate_improvement_metrics(
            standard_result, reranked_result
        )
        
        return {
            'query': query,
            'standard_result': standard_result,
            'reranked_result': reranked_result,
            'improvement_metrics': improvement_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_query(
        self,
        queries: List[str],
        use_rerank: bool = True,
        top_k: int = 5,
        retrieve_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        批量查询处理
        
        Args:
            queries: 查询列表
            use_rerank: 是否使用重排序
            top_k: 返回的文档数量
            retrieve_k: 重排序时初始检索的文档数量
        
        Returns:
            批量查询结果列表
        """
        logger.info(f"开始批量查询，共 {len(queries)} 个查询")
        
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries):
            logger.info(f"处理查询 {i+1}/{len(queries)}: {query[:50]}...")
            
            if use_rerank:
                result = self.query_with_rerank(query, top_k, retrieve_k)
            else:
                result = self.query_standard(query, top_k)
            
            results.append(result)
        
        total_time = time.time() - start_time
        
        logger.info(f"批量查询完成，总耗时: {total_time:.2f}秒")
        
        return results
    
    def query_with_metadata_filter(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        use_rerank: bool = True,
        top_k: int = 5,
        retrieve_k: int = 20
    ) -> Dict[str, Any]:
        """
        带元数据过滤的查询
        
        Args:
            query: 查询文本
            metadata_filters: 元数据过滤条件
            use_rerank: 是否使用重排序
            top_k: 返回的文档数量
            retrieve_k: 重排序时初始检索的文档数量
        
        Returns:
            查询结果字典
        """
        logger.info(f"执行元数据过滤查询: {query}, 过滤条件: {metadata_filters}")
        
        try:
            # 执行带过滤的查询
            if hasattr(self.rag_service, 'query_with_filter'):
                initial_results = self.rag_service.query_with_filter(
                    query, metadata_filters, top_k=retrieve_k if use_rerank else top_k
                )
            else:
                # 如果RAG服务不支持过滤，先查询后过滤
                initial_results = self.rag_service.query(
                    query, top_k=retrieve_k if use_rerank else top_k
                )
                initial_results = self._apply_metadata_filter(
                    initial_results, metadata_filters
                )
            
            if not use_rerank:
                return {
                    'query': query,
                    'method': 'standard_filtered',
                    'results': initial_results,
                    'metadata_filters': metadata_filters,
                    'timestamp': datetime.now().isoformat()
                }
            
            # 应用重排序
            if initial_results and 'documents' in initial_results:
                documents = initial_results['documents']
                scores = initial_results.get('scores', [1.0] * len(documents))
                
                reranked_docs, reranked_scores = self.rerank_service.rerank_documents(
                    query, documents, scores, top_k=top_k
                )
                
                reranked_results = {
                    'documents': reranked_docs,
                    'scores': reranked_scores,
                    'total_documents': len(reranked_docs)
                }
                
                return {
                    'query': query,
                    'method': 'reranked_filtered',
                    'results': reranked_results,
                    'metadata_filters': metadata_filters,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'query': query,
                'method': 'reranked_filtered',
                'results': {'documents': [], 'scores': [], 'total_documents': 0},
                'metadata_filters': metadata_filters,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"元数据过滤查询失败: {str(e)}")
            return {
                'query': query,
                'method': 'filtered',
                'error': str(e),
                'metadata_filters': metadata_filters,
                'timestamp': datetime.now().isoformat()
            }
    
    def _apply_metadata_filter(
        self,
        results: Dict[str, Any],
        metadata_filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用元数据过滤"""
        if not results or 'documents' not in results:
            return results
        
        filtered_docs = []
        filtered_scores = []
        
        documents = results['documents']
        scores = results.get('scores', [1.0] * len(documents))
        
        for doc, score in zip(documents, scores):
            metadata = doc.get('metadata', {})
            
            # 检查所有过滤条件
            match = True
            for key, value in metadata_filters.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return {
            'documents': filtered_docs,
            'scores': filtered_scores,
            'total_documents': len(filtered_docs)
        }
    
    def _calculate_improvement_metrics(
        self,
        standard_result: Dict[str, Any],
        reranked_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算改进指标"""
        metrics = {
            'time_difference': 0.0,
            'score_improvement': 0.0,
            'has_error': False
        }
        
        # 检查是否有错误
        if 'error' in standard_result or 'error' in reranked_result:
            metrics['has_error'] = True
            return metrics
        
        # 计算时间差异
        standard_time = standard_result.get('query_time', 0)
        reranked_time = reranked_result.get('query_time', 0)
        metrics['time_difference'] = reranked_time - standard_time
        
        # 计算分数改进（如果有分数信息）
        try:
            standard_scores = standard_result.get('results', {}).get('scores', [])
            reranked_scores = reranked_result.get('results', {}).get('scores', [])
            
            if standard_scores and reranked_scores:
                avg_standard_score = sum(standard_scores) / len(standard_scores)
                avg_reranked_score = sum(reranked_scores) / len(reranked_scores)
                metrics['score_improvement'] = avg_reranked_score - avg_standard_score
        except Exception as e:
            logger.warning(f"计算分数改进时出错: {str(e)}")
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        
        # 添加缓存统计（如果使用缓存）
        if hasattr(self.rerank_service, 'get_cache_stats'):
            stats['cache_stats'] = self.rerank_service.get_cache_stats()
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.performance_stats = {
            'total_queries': 0,
            'standard_queries': 0,
            'reranked_queries': 0,
            'avg_standard_time': 0.0,
            'avg_rerank_time': 0.0,
            'total_standard_time': 0.0,
            'total_rerank_time': 0.0
        }
        
        # 清理缓存统计（如果使用缓存）
        if hasattr(self.rerank_service, 'clear_cache'):
            self.rerank_service.clear_cache()
        
        logger.info("性能统计已重置")
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 检查RAG服务
            if hasattr(self.rag_service, 'health_check'):
                health_status['components']['rag_service'] = self.rag_service.health_check()
            else:
                health_status['components']['rag_service'] = {'status': 'unknown'}
            
            # 检查重排序服务
            rerank_health = {'status': 'healthy'}
            if hasattr(self.rerank_service, 'get_model_info'):
                rerank_health['model_info'] = self.rerank_service.get_model_info()
            health_status['components']['rerank_service'] = rerank_health
            
            # 检查缓存服务（如果使用）
            if hasattr(self.rerank_service, 'get_cache_stats'):
                cache_stats = self.rerank_service.get_cache_stats()
                health_status['components']['cache_service'] = {
                    'status': 'healthy',
                    'stats': cache_stats
                }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"健康检查失败: {str(e)}")
        
        return health_status