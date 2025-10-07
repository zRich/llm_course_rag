"""文档检索器模块"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from dataclasses import dataclass

from src.embedding.embedder import TextEmbedder
from src.vector_store.qdrant_client import QdrantVectorStore, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_index: int = 0

class DocumentRetriever:
    """文档检索器
    
    负责从向量数据库中检索与查询相关的文档片段
    """
    
    def __init__(self, 
                 embedder: TextEmbedder,
                 vector_store: QdrantVectorStore,
                 collection_name: str = "documents"):
        """
        初始化文档检索器
        
        Args:
            embedder: 文本向量化器
            vector_store: 向量存储
            collection_name: 集合名称
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"文档检索器初始化完成: {collection_name}")
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                score_threshold: float = 0.3,
                filter_conditions: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            filter_conditions: 过滤条件
            
        Returns:
            检索结果列表
        """
        try:
            self.logger.info(f"开始检索: {query[:50]}...")
            
            # 1. 将查询文本向量化
            query_vector = self.embedder.encode(query)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 2. 在向量数据库中搜索
            search_results = self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions
            )
            
            # 3. 转换搜索结果
            retrieval_results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.payload.get('content', ''),
                    score=result.score,
                    metadata=result.payload,
                    source=result.payload.get('file_path', 'unknown'),
                    chunk_index=result.payload.get('chunk_index', 0)
                )
                retrieval_results.append(retrieval_result)
            
            self.logger.info(f"检索完成: 找到 {len(retrieval_results)} 个相关文档")
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            raise
    
    def retrieve_with_rerank(self, 
                           query: str,
                           top_k: int = 5,
                           rerank_top_k: int = 20,
                           score_threshold: float = 0.3) -> List[RetrievalResult]:
        """
        带重排序的检索
        
        Args:
            query: 查询文本
            top_k: 最终返回结果数量
            rerank_top_k: 重排序前的候选数量
            score_threshold: 分数阈值
            
        Returns:
            重排序后的检索结果列表
        """
        try:
            # 1. 先检索更多候选结果
            candidates = self.retrieve(
                query=query,
                top_k=rerank_top_k,
                score_threshold=score_threshold
            )
            
            if len(candidates) <= top_k:
                return candidates
            
            # 2. 简单的重排序：基于内容长度和分数的综合评分
            for candidate in candidates:
                content_length_score = min(len(candidate.content) / 1000, 1.0)  # 内容长度评分
                candidate.score = candidate.score * 0.8 + content_length_score * 0.2
            
            # 3. 按新分数排序并返回top_k
            candidates.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"重排序完成: {len(candidates)} -> {top_k}")
            return candidates[:top_k]
            
        except Exception as e:
            self.logger.error(f"重排序检索失败: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息
        """
        try:
            # 这里需要根据实际的vector_store实现来获取统计信息
            # 简化版本返回基本信息
            return {
                "collection_name": self.collection_name,
                "status": "active"
            }
        except Exception as e:
            self.logger.error(f"获取集合统计失败: {e}")
            return {"error": str(e)}
    
    def format_context(self, results: List[RetrievalResult], max_length: int = 2000) -> str:
        """
        格式化检索结果为上下文文本
        
        Args:
            results: 检索结果列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文文本
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # 格式化单个结果
            formatted_part = f"[文档{i+1}] {result.content}"
            
            # 检查长度限制
            if current_length + len(formatted_part) > max_length:
                if current_length == 0:  # 如果第一个文档就超长，截断它
                    available_length = max_length - len(f"[文档{i+1}] ")
                    truncated_content = result.content[:available_length] + "..."
                    formatted_part = f"[文档{i+1}] {truncated_content}"
                    context_parts.append(formatted_part)
                break
            
            context_parts.append(formatted_part)
            current_length += len(formatted_part) + 2  # +2 for \n\n
        
        return "\n\n".join(context_parts)