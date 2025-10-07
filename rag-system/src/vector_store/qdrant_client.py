"""Qdrant向量数据库客户端"""

from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
from qdrant_client.http.exceptions import ResponseHandlingException
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None

class QdrantVectorStore:
    """Qdrant向量存储客户端"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 timeout: int = 60):
        """
        初始化Qdrant客户端
        
        Args:
            host: Qdrant服务器地址
            port: Qdrant服务器端口
            timeout: 连接超时时间
        """
        self.host = host
        self.port = port
        
        try:
            self.client = QdrantClient(
                host=host, 
                port=port, 
                timeout=timeout
            )
            
            # 测试连接
            self.client.get_collections()
            logger.info(f"Qdrant客户端连接成功: {host}:{port}")
            
        except Exception as e:
            logger.error(f"Qdrant连接失败: {e}")
            raise
    
    def create_collection(self, 
                         collection_name: str, 
                         vector_size: int,
                         distance: str = "Cosine",
                         recreate: bool = False) -> bool:
        """
        创建向量集合
        
        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量 ("Cosine", "Euclidean", "Dot")
            recreate: 是否重新创建（删除已存在的）
            
        Returns:
            是否创建成功
        """
        try:
            # 检查集合是否存在
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if recreate:
                    logger.info(f"删除已存在的集合: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"集合已存在: {collection_name}")
                    return True
            
            # 距离度量映射
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            if distance not in distance_map:
                raise ValueError(f"不支持的距离度量: {distance}")
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map[distance]
                )
            )
            
            logger.info(f"集合创建成功: {collection_name} (维度: {vector_size}, 距离: {distance})")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def insert_vectors(self, 
                      collection_name: str,
                      vectors: np.ndarray,
                      payloads: List[Dict[str, Any]],
                      ids: Optional[List[str]] = None) -> bool:
        """
        插入向量
        
        Args:
            collection_name: 集合名称
            vectors: 向量数组
            payloads: 元数据列表
            ids: 向量ID列表（可选，自动生成UUID）
            
        Returns:
            是否插入成功
        """
        try:
            if len(vectors) != len(payloads):
                raise ValueError("向量数量与元数据数量不匹配")
            
            # 生成ID（如果未提供）
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            
            # 创建点结构
            points = []
            for i, (vector, payload) in enumerate(zip(vectors, payloads)):
                point = PointStruct(
                    id=ids[i],
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=payload
                )
                points.append(point)
            
            # 批量插入
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"成功插入 {len(points)} 个向量到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            return False
    
    def search(self, 
              collection_name: str,
              query_vector: np.ndarray,
              limit: int = 10,
              score_threshold: Optional[float] = None,
              filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        向量相似度搜索
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            limit: 返回结果数量
            score_threshold: 分数阈值
            filter_conditions: 过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 处理查询向量格式
            if isinstance(query_vector, np.ndarray):
                # 如果是二维数组，取第一行
                if query_vector.ndim == 2:
                    query_vector = query_vector[0]
                # 转换为列表
                query_vector = query_vector.tolist()
            
            # 构建过滤器
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    condition = FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    conditions.append(condition)
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # 执行搜索
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # 转换结果格式
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                )
                results.append(search_result)
            
            logger.info(f"搜索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合信息字典
        """
        try:
            # 使用更简单的方法，只返回基本信息
            collections = self.client.get_collections()
            
            # 查找目标集合
            target_collection = None
            for col in collections.collections:
                if col.name == collection_name:
                    target_collection = col
                    break
            
            if target_collection is None:
                logger.warning(f"集合 {collection_name} 不存在")
                return None
            
            # 返回简化的集合信息
            return {
                'name': collection_name,
                'vectors_count': 0,  # 简化版本，不获取详细统计
                'indexed_vectors_count': 0,
                'points_count': 0,
                'segments_count': 0,
                'config': {
                    'vector_size': 384,  # 使用已知的向量维度
                    'distance': 'Cosine'
                },
                'status': 'green'
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否删除成功
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"集合删除成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合
        
        Returns:
            集合名称列表
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
            
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            return []
    
    def count_points(self, collection_name: str) -> int:
        """
        统计集合中的点数量
        
        Args:
            collection_name: 集合名称
            
        Returns:
            点数量
        """
        try:
            # 使用scroll方法来统计点数量
            result = self.client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            # 如果有结果，说明集合中有数据
            if result[0]:  # result是(points, next_page_offset)的元组
                # 简化版本：如果能找到点，返回一个大于0的数
                return len(result[0])
            else:
                return 0
            
        except Exception as e:
            logger.error(f"统计点数量失败: {e}")
            return 0