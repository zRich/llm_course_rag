#!/usr/bin/env python3
"""
重排序服务模块

提供文档重排序功能，包括：
- 基础重排序算法
- 语义相似度重排序
- 多种重排序策略
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RerankService:
    """重排序服务类"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化重排序服务
        
        Args:
            model_name: 用于重排序的模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"重排序模型 {self.model_name} 加载成功")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            # 使用简单的基于关键词的重排序作为后备方案
            self.model = None
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        scores: List[float],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            scores: 原始分数列表
            top_k: 返回的文档数量
        
        Returns:
            重排序后的文档列表和分数列表
        """
        if not documents:
            return documents, scores
        
        try:
            if self.model is not None:
                # 使用语义模型重排序
                reranked_docs, reranked_scores = self._semantic_rerank(
                    query, documents, scores, top_k
                )
            else:
                # 使用关键词重排序作为后备
                reranked_docs, reranked_scores = self._keyword_rerank(
                    query, documents, scores, top_k
                )
            
            logger.info(f"重排序完成，返回 {len(reranked_docs)} 个文档")
            return reranked_docs, reranked_scores
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 返回原始结果
            if top_k:
                return documents[:top_k], scores[:top_k]
            return documents, scores
    
    def _semantic_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: List[float],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """基于语义相似度的重排序"""
        # 提取文档内容
        doc_texts = [doc.get('content', '') for doc in documents]
        
        # 计算查询和文档的嵌入
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(doc_texts)
        
        # 计算语义相似度
        semantic_scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # 结合原始分数和语义分数
        combined_scores = []
        for i, (orig_score, sem_score) in enumerate(zip(original_scores, semantic_scores)):
            # 加权组合：70%语义分数 + 30%原始分数
            combined_score = 0.7 * sem_score + 0.3 * orig_score
            combined_scores.append(combined_score)
        
        # 按组合分数排序
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # 重排序文档和分数
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_scores = [combined_scores[i] for i in sorted_indices]
        
        if top_k:
            reranked_docs = reranked_docs[:top_k]
            reranked_scores = reranked_scores[:top_k]
        
        return reranked_docs, reranked_scores
    
    def _keyword_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: List[float],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """基于关键词匹配的重排序"""
        query_words = set(query.lower().split())
        
        # 计算关键词匹配分数
        keyword_scores = []
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_words = set(content.split())
            
            # 计算交集比例
            intersection = query_words.intersection(doc_words)
            if query_words:
                keyword_score = len(intersection) / len(query_words)
            else:
                keyword_score = 0.0
            
            keyword_scores.append(keyword_score)
        
        # 结合原始分数和关键词分数
        combined_scores = []
        for orig_score, kw_score in zip(original_scores, keyword_scores):
            # 加权组合：60%关键词分数 + 40%原始分数
            combined_score = 0.6 * kw_score + 0.4 * orig_score
            combined_scores.append(combined_score)
        
        # 按组合分数排序
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # 重排序文档和分数
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_scores = [combined_scores[i] for i in sorted_indices]
        
        if top_k:
            reranked_docs = reranked_docs[:top_k]
            reranked_scores = reranked_scores[:top_k]
        
        return reranked_docs, reranked_scores
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        scores_list: List[List[float]],
        top_k: Optional[int] = None
    ) -> List[Tuple[List[Dict[str, Any]], List[float]]]:
        """批量重排序"""
        results = []
        
        for query, documents, scores in zip(queries, documents_list, scores_list):
            reranked_docs, reranked_scores = self.rerank_documents(
                query, documents, scores, top_k
            )
            results.append((reranked_docs, reranked_scores))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'fallback_mode': self.model is None
        }