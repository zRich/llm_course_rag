"""
向量存储服务
实现与Qdrant向量数据库的交互功能
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)

from src.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储服务"""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.embedding_dimensions
        
    def create_collection(self, collection_name: str = None, 
                         vector_size: int = None, 
                         distance: Distance = Distance.COSINE) -> bool:
        """
        创建向量集合
        
        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量方式
            
        Returns:
            是否创建成功
        """
        collection_name = collection_name or self.collection_name
        vector_size = vector_size or self.vector_size
        
        try:
            # 检查集合是否已存在
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name in existing_names:
                logger.info(f"集合已存在: {collection_name}")
                return True
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            logger.info(f"创建集合成功: {collection_name}, 维度: {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def delete_collection(self, collection_name: str = None) -> bool:
        """
        删除向量集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否删除成功
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"删除集合成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def collection_exists(self, collection_name: str = None) -> bool:
        """
        检查集合是否存在
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否存在
        """
        collection_name = collection_name or self.collection_name
        
        try:
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            return collection_name in existing_names
            
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False
    
    def add_vectors(self, vectors: List[List[float]], 
                   payloads: List[Dict], 
                   ids: List[str] = None,
                   collection_name: str = None) -> bool:
        """
        添加向量到集合
        
        Args:
            vectors: 向量列表
            payloads: 载荷数据列表
            ids: 向量ID列表（可选）
            collection_name: 集合名称
            
        Returns:
            是否添加成功
        """
        collection_name = collection_name or self.collection_name
        
        if not vectors or not payloads:
            raise ValueError("向量和载荷数据不能为空")
        
        if len(vectors) != len(payloads):
            raise ValueError("向量数量和载荷数据数量不匹配")
        
        # 生成ID（如果未提供）
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError("ID数量和向量数量不匹配")
        
        try:
            # 构建点数据
            points = []
            for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
                # 添加时间戳到载荷
                payload_with_timestamp = {
                    **payload,
                    "created_at": datetime.utcnow().isoformat(),
                    "vector_id": point_id
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload_with_timestamp
                )
                points.append(point)
            
            # 批量插入
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"添加向量成功: {len(vectors)} 个向量到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], 
                      limit: int = 10,
                      score_threshold: float = None,
                      filter_conditions: Dict = None,
                      collection_name: str = None) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            filter_conditions: 过滤条件
            collection_name: 集合名称
            
        Returns:
            搜索结果列表
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # 构建过滤器
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            # 执行搜索
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # 格式化结果
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload,
                    "vector": scored_point.vector
                }
                results.append(result)
            
            logger.debug(f"向量搜索完成: 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _build_filter(self, conditions: Dict) -> Filter:
        """
        构建查询过滤器
        
        Args:
            conditions: 过滤条件字典
            
        Returns:
            Qdrant过滤器对象
        """
        must_conditions = []
        
        for field, value in conditions.items():
            if isinstance(value, (str, int, float, bool)):
                condition = FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
                must_conditions.append(condition)
        
        return Filter(must=must_conditions) if must_conditions else None
    
    def get_vector(self, vector_id: str, collection_name: str = None) -> Optional[Dict]:
        """
        获取指定ID的向量
        
        Args:
            vector_id: 向量ID
            collection_name: 集合名称
            
        Returns:
            向量数据或None
        """
        collection_name = collection_name or self.collection_name
        
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[vector_id]
            )
            
            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
            
            return None
            
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None
    
    def update_vector(self, vector_id: str, vector: List[float] = None, 
                     payload: Dict = None, collection_name: str = None) -> bool:
        """
        更新向量数据
        
        Args:
            vector_id: 向量ID
            vector: 新向量（可选）
            payload: 新载荷数据（可选）
            collection_name: 集合名称
            
        Returns:
            是否更新成功
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # 获取现有数据
            existing = self.get_vector(vector_id, collection_name)
            if not existing:
                logger.warning(f"向量不存在: {vector_id}")
                return False
            
            # 准备更新数据
            update_vector = vector if vector is not None else existing["vector"]
            update_payload = payload if payload is not None else existing["payload"]
            
            # 添加更新时间戳
            if isinstance(update_payload, dict):
                update_payload["updated_at"] = datetime.utcnow().isoformat()
            
            # 执行更新
            point = PointStruct(
                id=vector_id,
                vector=update_vector,
                payload=update_payload
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.info(f"更新向量成功: {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新向量失败: {e}")
            return False
    
    def delete_vectors(self, vector_ids: List[str], collection_name: str = None) -> bool:
        """
        删除向量
        
        Args:
            vector_ids: 向量ID列表
            collection_name: 集合名称
            
        Returns:
            是否删除成功
        """
        collection_name = collection_name or self.collection_name
        
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids
                )
            )
            
            logger.info(f"删除向量成功: {len(vector_ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def count_vectors(self, collection_name: str = None) -> int:
        """
        统计向量数量
        
        Args:
            collection_name: 集合名称
            
        Returns:
            向量数量
        """
        collection_name = collection_name or self.collection_name
        
        try:
            info = self.client.get_collection(collection_name)
            return info.points_count
            
        except Exception as e:
            logger.error(f"统计向量数量失败: {e}")
            return 0
    
    def get_collection_info(self, collection_name: str = None) -> Optional[Dict]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合信息字典
        """
        collection_name = collection_name or self.collection_name
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
                "status": info.status.value
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return None
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否健康
        """
        try:
            collections = self.client.get_collections()
            logger.info(f"Qdrant健康检查通过，共有 {len(collections.collections)} 个集合")
            return True
            
        except Exception as e:
            logger.error(f"Qdrant健康检查失败: {e}")
            return False
    
    def clear_collection(self, collection_name: str = None) -> bool:
        """
        清空集合中的所有向量
        
        Args:
            collection_name: 集合名称
            
        Returns:
            是否清空成功
        """
        collection_name = collection_name or self.collection_name
        
        try:
            # 删除集合中的所有点
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            
            logger.info(f"清空集合成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False