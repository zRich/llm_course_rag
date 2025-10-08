"""
向量服务
整合文档向量化和向量存储功能
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time

from sqlalchemy.orm import Session

from src.models.database import get_db
from src.models.chunk import Chunk
from src.models.document import Document
from src.services.embedding_service import EmbeddingService
from src.services.vector_store import VectorStore
from src.config.settings import settings

logger = logging.getLogger(__name__)


class VectorService:
    """向量服务"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
    def initialize(self) -> bool:
        """
        初始化向量服务
        
        Returns:
            是否初始化成功
        """
        try:
            # 检查向量存储健康状态
            if not self.vector_store.health_check():
                logger.error("向量存储服务不可用")
                return False
            
            # 创建向量集合（如果不存在）
            if not self.vector_store.collection_exists():
                success = self.vector_store.create_collection()
                if not success:
                    logger.error("创建向量集合失败")
                    return False
            
            logger.info("向量服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"向量服务初始化失败: {e}")
            return False

    async def vectorize_documents(self, document_ids: Optional[List[str]] = None, 
                                force_revectorize: bool = False) -> Dict:
        """
        批量向量化文档
        
        Args:
            document_ids: 文档ID列表，为空则处理所有未向量化文档
            force_revectorize: 是否强制重新向量化
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        processed_documents = 0
        processed_chunks = 0
        failed_documents = []
        
        db = next(get_db())
        
        try:
            # 获取需要处理的文档
            if document_ids:
                # 处理指定文档
                documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
            else:
                # 处理所有未向量化的文档
                if force_revectorize:
                    documents = db.query(Document).all()
                else:
                    documents = db.query(Document).filter(Document.is_vectorized == False).all()
            
            for document in documents:
                try:
                    if force_revectorize and document.is_vectorized:
                        # 删除旧向量
                        self.delete_document_vectors(str(document.id), db)
                        # 重置分块的向量化状态
                        chunks_to_reset = db.query(Chunk).filter(Chunk.document_id == document.id).all()
                        for chunk in chunks_to_reset:
                            chunk.vector_id = None
                            chunk.embedding_model = None
                            chunk.embedding_dimension = None
                            chunk.is_embedded = 0
                            chunk.embedded_at = None
                        # 重置文档的向量化状态
                        document.is_vectorized = False
                        document.vectorized_at = None
                        db.commit()
                    
                    # 向量化文档
                    success = self.vectorize_document(document.id, db)
                    if success:
                        processed_documents += 1
                        # 计算处理的分块数
                        chunk_count = db.query(Chunk).filter(
                            Chunk.document_id == document.id,
                            Chunk.is_embedded == 1
                        ).count()
                        processed_chunks += chunk_count
                    else:
                        failed_documents.append(str(document.id))
                        
                except Exception as e:
                    logger.error(f"处理文档 {document.id} 失败: {e}")
                    failed_documents.append(str(document.id))
            
            processing_time = time.time() - start_time
            
            return {
                "processed_documents": processed_documents,
                "processed_chunks": processed_chunks,
                "processing_time": processing_time,
                "failed_documents": failed_documents
            }
            
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            raise
        finally:
            db.close()

    def vectorize_document(self, document_id: str, db: Session = None) -> bool:
        """
        对文档进行向量化处理
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            是否处理成功
        """
        if db is None:
            db = next(get_db())
        
        try:
            # 获取文档
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                logger.error(f"文档不存在: {document_id}")
                return False
            
            # 获取文档的所有分块
            chunks = db.query(Chunk).filter(
                Chunk.document_id == document_id,
                Chunk.is_embedded == 0  # 只处理未向量化的分块
            ).all()
            
            if not chunks:
                logger.info(f"文档 {document_id} 没有需要向量化的分块")
                return True
            
            # 批量向量化
            texts = [chunk.content for chunk in chunks]
            vectors, metadata_list = self.embedding_service.get_embeddings_batch(texts)
            
            if not vectors or len(vectors) != len(chunks):
                logger.error(f"向量化失败: 期望 {len(chunks)} 个向量，实际获得 {len(vectors) if vectors else 0} 个")
                return False
            
            # 准备向量存储数据
            vector_ids = []
            payloads = []
            
            for chunk, vector in zip(chunks, vectors):
                vector_id = str(chunk.id)  # 使用UUID字符串作为向量ID
                vector_ids.append(vector_id)
                
                payload = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "token_count": chunk.token_count,
                    "char_count": chunk.char_count,
                    "document_filename": document.filename,
                    "document_title": document.title,
                    "metadata": chunk.metadata_ or {}
                }
                payloads.append(payload)
            
            # 存储向量
            success = self.vector_store.add_vectors(
                vectors=vectors,
                payloads=payloads,
                ids=vector_ids
            )
            
            if not success:
                logger.error(f"向量存储失败: 文档 {document_id}")
                return False
            
            # 更新数据库中的向量信息
            for chunk, vector_id, vector in zip(chunks, vector_ids, vectors):
                chunk.vector_id = vector_id
                chunk.embedding_model = self.embedding_service.model_name
                chunk.embedding_dimension = len(vector)
                chunk.is_embedded = 1
                chunk.embedded_at = datetime.utcnow()
            
            # 更新文档状态
            document.is_vectorized = True
            document.vectorized_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"文档向量化成功: 文档 {document_id}, 处理 {len(chunks)} 个分块")
            return True
            
        except Exception as e:
            logger.error(f"文档向量化失败: {e}")
            if db:
                db.rollback()
            return False
    
    def search_similar_chunks(self, query: str, 
                            limit: int = 10,
                            score_threshold: float = None,
                            document_ids: List[int] = None,
                            db: Session = None) -> List[Dict]:
        """
        搜索相似的文档分块
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            score_threshold: 相似度阈值
            document_ids: 限制搜索的文档ID列表
            db: 数据库会话
            
        Returns:
            搜索结果列表
        """
        if db is None:
            db = next(get_db())
        
        try:
            # 将查询文本向量化
            query_vector, metadata = self.embedding_service.get_embedding(query)
            if not query_vector:
                logger.error("查询文本向量化失败")
                return []
            
            # 构建过滤条件
            filter_conditions = {}
            if document_ids:
                # 注意：Qdrant的过滤器可能需要特殊处理列表条件
                # 这里简化处理，实际使用时可能需要调整
                if len(document_ids) == 1:
                    filter_conditions["document_id"] = document_ids[0]
            
            # 执行向量搜索
            search_results = self.vector_store.search_vectors(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions if filter_conditions else None
            )
            
            # 丰富搜索结果
            enriched_results = []
            for result in search_results:
                payload = result["payload"]
                chunk_id = payload.get("chunk_id")
                
                # 从数据库获取完整的分块信息
                chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
                if chunk:
                    enriched_result = {
                        "chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "score": result["score"],
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        "token_count": chunk.token_count,
                        "char_count": chunk.char_count,
                        "document_filename": payload.get("document_filename"),
                        "document_title": payload.get("document_title"),
                        "metadata": chunk.metadata_ or {} if isinstance(chunk.metadata_, dict) else {},
                        "vector_id": result["id"]
                    }
                    enriched_results.append(enriched_result)
            
            logger.info(f"相似度搜索完成: 查询='{query[:50]}...', 找到 {len(enriched_results)} 个结果")
            return enriched_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def get_chunk_context(self, chunk_id: int, 
                         context_size: int = 2,
                         db: Session = None) -> Dict:
        """
        获取分块的上下文信息
        
        Args:
            chunk_id: 分块ID
            context_size: 上下文大小（前后各几个分块）
            db: 数据库会话
            
        Returns:
            包含上下文的分块信息
        """
        if db is None:
            db = next(get_db())
        
        try:
            # 获取目标分块
            target_chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if not target_chunk:
                logger.error(f"分块不存在: {chunk_id}")
                return {}
            
            # 获取同一文档的相邻分块
            document_id = target_chunk.document_id
            chunk_index = target_chunk.chunk_index
            
            # 计算上下文范围
            start_index = max(0, chunk_index - context_size)
            end_index = chunk_index + context_size + 1
            
            # 查询上下文分块
            context_chunks = db.query(Chunk).filter(
                Chunk.document_id == document_id,
                Chunk.chunk_index >= start_index,
                Chunk.chunk_index < end_index
            ).order_by(Chunk.chunk_index).all()
            
            # 构建结果
            result = {
                "target_chunk": {
                    "id": target_chunk.id,
                    "chunk_index": target_chunk.chunk_index,
                    "content": target_chunk.content,
                    "start_position": target_chunk.start_pos,
                    "end_position": target_chunk.end_pos
                },
                "context_chunks": [],
                "full_context": ""
            }
            
            # 添加上下文分块
            context_contents = []
            for chunk in context_chunks:
                chunk_info = {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "is_target": chunk.id == chunk_id
                }
                result["context_chunks"].append(chunk_info)
                context_contents.append(chunk.content)
            
            # 合并上下文内容
            result["full_context"] = "\n\n".join(context_contents)
            
            return result
            
        except Exception as e:
            logger.error(f"获取分块上下文失败: {e}")
            return {}
    
    def delete_document_vectors(self, document_id: int, db: Session = None) -> bool:
        """
        删除文档的所有向量
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            是否删除成功
        """
        if db is None:
            db = next(get_db())
        
        try:
            # 获取文档的所有分块
            chunks = db.query(Chunk).filter(
                Chunk.document_id == document_id,
                Chunk.vector_id.isnot(None)
            ).all()
            
            if not chunks:
                logger.info(f"文档 {document_id} 没有向量需要删除")
                return True
            
            # 收集向量ID
            vector_ids = [chunk.vector_id for chunk in chunks]
            
            # 从向量存储中删除
            success = self.vector_store.delete_vectors(vector_ids)
            if not success:
                logger.error(f"从向量存储删除失败: 文档 {document_id}")
                return False
            
            # 更新数据库
            for chunk in chunks:
                chunk.vector_id = None
                chunk.embedding_model = None
                chunk.embedding_dimension = None
                chunk.is_embedded = 0
                chunk.embedded_at = None
            
            # 更新文档状态
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.is_vectorized = False
                document.vectorized_at = None
            
            db.commit()
            
            logger.info(f"删除文档向量成功: 文档 {document_id}, 删除 {len(vector_ids)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"删除文档向量失败: {e}")
            if db:
                db.rollback()
            return False
    
    def get_vectorization_stats(self, db: Session = None) -> Dict:
        """
        获取向量化统计信息
        
        Args:
            db: 数据库会话
            
        Returns:
            统计信息字典
        """
        if db is None:
            db = next(get_db())
        
        try:
            # 数据库统计
            total_documents = db.query(Document).count()
            vectorized_documents = db.query(Document).filter(Document.is_vectorized == True).count()
            total_chunks = db.query(Chunk).count()
            vectorized_chunks = db.query(Chunk).filter(Chunk.is_embedded == 1).count()
            
            # 向量存储统计
            vector_count = self.vector_store.count_vectors()
            collection_info = self.vector_store.get_collection_info()
            
            stats = {
                "database": {
                    "total_documents": total_documents,
                    "vectorized_documents": vectorized_documents,
                    "vectorization_rate": vectorized_documents / total_documents if total_documents > 0 else 0,
                    "total_chunks": total_chunks,
                    "vectorized_chunks": vectorized_chunks,
                    "chunk_vectorization_rate": vectorized_chunks / total_chunks if total_chunks > 0 else 0
                },
                "vector_store": {
                    "total_vectors": vector_count,
                    "collection_info": collection_info
                },
                "consistency": {
                    "vectors_match_chunks": vector_count == vectorized_chunks
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取向量化统计失败: {e}")
            return {}
    
    def reindex_document(self, document_id: int, db: Session = None) -> bool:
        """
        重新索引文档（删除旧向量并重新向量化）
        
        Args:
            document_id: 文档ID
            db: 数据库会话
            
        Returns:
            是否重新索引成功
        """
        try:
            # 删除旧向量
            delete_success = self.delete_document_vectors(document_id, db)
            if not delete_success:
                logger.error(f"删除旧向量失败: 文档 {document_id}")
                return False
            
            # 重新向量化
            vectorize_success = self.vectorize_document(document_id, db)
            if not vectorize_success:
                logger.error(f"重新向量化失败: 文档 {document_id}")
                return False
            
            logger.info(f"文档重新索引成功: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"文档重新索引失败: {e}")
            return False