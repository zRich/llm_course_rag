"""文档仓库"""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from ..models.document import (
    Document,
    DocumentChunk,
    DocumentChunkCreate,
    DocumentChunkUpdate,
    DocumentCreate,
    DocumentStatus,
    DocumentType,
    DocumentUpdate,
    ProcessingStatus
)
from .base import BaseRepository


class DocumentRepository(BaseRepository[Document, DocumentCreate, DocumentUpdate]):
    """文档仓库类"""
    
    def __init__(self):
        super().__init__(Document)
    
    # 文档查询方法
    def get_by_title(self, session: Session, *, title: str) -> Optional[Document]:
        """根据标题获取文档"""
        query = select(Document).where(Document.title == title)
        return session.exec(query).first()
    
    def get_by_hash(self, session: Session, *, file_hash: str) -> Optional[Document]:
        """根据文件哈希获取文档"""
        query = select(Document).where(Document.file_hash == file_hash)
        return session.exec(query).first()
    
    def get_by_owner(
        self, 
        session: Session, 
        *, 
        owner_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """获取用户的文档列表"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"owner_id": owner_id},
            order_by="-created_at"
        )
    
    def get_by_status(
        self, 
        session: Session, 
        *, 
        status: DocumentStatus, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """根据状态获取文档列表"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"status": status},
            order_by="-created_at"
        )
    
    def get_by_type(
        self, 
        session: Session, 
        *, 
        document_type: DocumentType, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """根据类型获取文档列表"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"document_type": document_type},
            order_by="-created_at"
        )
    
    def search_documents(
        self, 
        session: Session, 
        *, 
        query: str, 
        owner_id: Optional[UUID] = None,
        document_type: Optional[DocumentType] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """搜索文档"""
        conditions = [
            or_(
                Document.title.like(f"%{query}%"),
                Document.description.like(f"%{query}%")
            )
        ]
        
        if owner_id:
            conditions.append(Document.owner_id == owner_id)
        
        if document_type:
            conditions.append(Document.document_type == document_type)
        
        stmt = select(Document).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(desc(Document.created_at))
        
        return session.exec(stmt).all()
    
    def get_processing_documents(self, session: Session) -> List[Document]:
        """获取正在处理的文档"""
        return self.get_multi(
            session,
            filters={"processing_status": ProcessingStatus.PROCESSING}
        )
    
    def get_failed_documents(self, session: Session) -> List[Document]:
        """获取处理失败的文档"""
        return self.get_multi(
            session,
            filters={"processing_status": ProcessingStatus.FAILED}
        )
    
    def update_processing_status(
        self, 
        session: Session, 
        *, 
        document: Document, 
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> Document:
        """更新文档处理状态"""
        document.processing_status = status
        document.processed_at = datetime.utcnow()
        
        if error_message:
            document.error_message = error_message
        
        session.add(document)
        session.commit()
        session.refresh(document)
        return document
    
    def get_document_stats(self, session: Session, *, owner_id: Optional[UUID] = None) -> Dict:
        """获取文档统计信息"""
        base_query = select(Document)
        if owner_id:
            base_query = base_query.where(Document.owner_id == owner_id)
        
        # 总文档数
        total_count = session.exec(
            select(func.count()).select_from(base_query.subquery())
        ).one()
        
        # 按状态统计
        status_stats = {}
        for status in DocumentStatus:
            count_query = base_query.where(Document.status == status)
            count = session.exec(
                select(func.count()).select_from(count_query.subquery())
            ).one()
            status_stats[status.value] = count
        
        # 按类型统计
        type_stats = {}
        for doc_type in DocumentType:
            count_query = base_query.where(Document.document_type == doc_type)
            count = session.exec(
                select(func.count()).select_from(count_query.subquery())
            ).one()
            type_stats[doc_type.value] = count
        
        return {
            "total": total_count,
            "by_status": status_stats,
            "by_type": type_stats
        }
    
    # 异步方法
    async def aget_by_title(self, session: AsyncSession, *, title: str) -> Optional[Document]:
        """异步根据标题获取文档"""
        query = select(Document).where(Document.title == title)
        result = await session.exec(query)
        return result.first()
    
    async def aget_by_hash(self, session: AsyncSession, *, file_hash: str) -> Optional[Document]:
        """异步根据文件哈希获取文档"""
        query = select(Document).where(Document.file_hash == file_hash)
        result = await session.exec(query)
        return result.first()
    
    async def aget_by_owner(
        self, 
        session: AsyncSession, 
        *, 
        owner_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """异步获取用户的文档列表"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"owner_id": owner_id},
            order_by="-created_at"
        )
    
    async def asearch_documents(
        self, 
        session: AsyncSession, 
        *, 
        query: str, 
        owner_id: Optional[UUID] = None,
        document_type: Optional[DocumentType] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """异步搜索文档"""
        conditions = [
            or_(
                Document.title.like(f"%{query}%"),
                Document.description.like(f"%{query}%")
            )
        ]
        
        if owner_id:
            conditions.append(Document.owner_id == owner_id)
        
        if document_type:
            conditions.append(Document.document_type == document_type)
        
        stmt = select(Document).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(desc(Document.created_at))
        
        result = await session.exec(stmt)
        return result.all()
    
    async def aupdate_processing_status(
        self, 
        session: AsyncSession, 
        *, 
        document: Document, 
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> Document:
        """异步更新文档处理状态"""
        document.processing_status = status
        document.processed_at = datetime.utcnow()
        
        if error_message:
            document.error_message = error_message
        
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return document


class DocumentChunkRepository(BaseRepository[DocumentChunk, DocumentChunkCreate, DocumentChunkUpdate]):
    """文档块仓库类"""
    
    def __init__(self):
        super().__init__(DocumentChunk)
    
    # 文档块查询方法
    def get_by_document(
        self, 
        session: Session, 
        *, 
        document_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[DocumentChunk]:
        """获取文档的所有块"""
        return self.get_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"document_id": document_id},
            order_by="chunk_index"
        )
    
    def get_by_vector_id(self, session: Session, *, vector_id: str) -> Optional[DocumentChunk]:
        """根据向量ID获取文档块"""
        query = select(DocumentChunk).where(DocumentChunk.vector_id == vector_id)
        return session.exec(query).first()
    
    def get_chunk_by_index(
        self, 
        session: Session, 
        *, 
        document_id: UUID, 
        chunk_index: int
    ) -> Optional[DocumentChunk]:
        """根据索引获取文档块"""
        query = select(DocumentChunk).where(
            and_(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_index == chunk_index
            )
        )
        return session.exec(query).first()
    
    def search_chunks(
        self, 
        session: Session, 
        *, 
        query: str, 
        document_id: Optional[UUID] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[DocumentChunk]:
        """搜索文档块"""
        conditions = [DocumentChunk.content.like(f"%{query}%")]
        
        if document_id:
            conditions.append(DocumentChunk.document_id == document_id)
        
        stmt = select(DocumentChunk).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(
            DocumentChunk.document_id, 
            DocumentChunk.chunk_index
        )
        
        return session.exec(stmt).all()
    
    def get_chunks_with_vectors(self, session: Session) -> List[DocumentChunk]:
        """获取已生成向量的文档块"""
        query = select(DocumentChunk).where(DocumentChunk.vector_id.isnot(None))
        return session.exec(query).all()
    
    def get_chunks_without_vectors(self, session: Session) -> List[DocumentChunk]:
        """获取未生成向量的文档块"""
        query = select(DocumentChunk).where(DocumentChunk.vector_id.is_(None))
        return session.exec(query).all()
    
    def update_vector_id(
        self, 
        session: Session, 
        *, 
        chunk: DocumentChunk, 
        vector_id: str
    ) -> DocumentChunk:
        """更新文档块的向量ID"""
        chunk.vector_id = vector_id
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
    
    def delete_by_document(self, session: Session, *, document_id: UUID) -> int:
        """删除文档的所有块"""
        return self.abulk_delete(
            session,
            filters={"document_id": document_id}
        )
    
    def get_chunk_stats(self, session: Session, *, document_id: Optional[UUID] = None) -> Dict:
        """获取文档块统计信息"""
        base_query = select(DocumentChunk)
        if document_id:
            base_query = base_query.where(DocumentChunk.document_id == document_id)
        
        # 总块数
        total_count = session.exec(
            select(func.count()).select_from(base_query.subquery())
        ).one()
        
        # 有向量的块数
        with_vector_query = base_query.where(DocumentChunk.vector_id.isnot(None))
        with_vector_count = session.exec(
            select(func.count()).select_from(with_vector_query.subquery())
        ).one()
        
        # 平均块大小
        avg_size = session.exec(
            select(func.avg(func.length(DocumentChunk.content))).select_from(
                base_query.subquery()
            )
        ).one() or 0
        
        return {
            "total": total_count,
            "with_vector": with_vector_count,
            "without_vector": total_count - with_vector_count,
            "avg_size": int(avg_size)
        }
    
    # 异步方法
    async def aget_by_document(
        self, 
        session: AsyncSession, 
        *, 
        document_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[DocumentChunk]:
        """异步获取文档的所有块"""
        return await self.aget_multi(
            session,
            skip=skip,
            limit=limit,
            filters={"document_id": document_id},
            order_by="chunk_index"
        )
    
    async def aget_by_vector_id(self, session: AsyncSession, *, vector_id: str) -> Optional[DocumentChunk]:
        """异步根据向量ID获取文档块"""
        query = select(DocumentChunk).where(DocumentChunk.vector_id == vector_id)
        result = await session.exec(query)
        return result.first()
    
    async def asearch_chunks(
        self, 
        session: AsyncSession, 
        *, 
        query: str, 
        document_id: Optional[UUID] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[DocumentChunk]:
        """异步搜索文档块"""
        conditions = [DocumentChunk.content.like(f"%{query}%")]
        
        if document_id:
            conditions.append(DocumentChunk.document_id == document_id)
        
        stmt = select(DocumentChunk).where(
            and_(*conditions)
        ).offset(skip).limit(limit).order_by(
            DocumentChunk.document_id, 
            DocumentChunk.chunk_index
        )
        
        result = await session.exec(stmt)
        return result.all()
    
    async def aupdate_vector_id(
        self, 
        session: AsyncSession, 
        *, 
        chunk: DocumentChunk, 
        vector_id: str
    ) -> DocumentChunk:
        """异步更新文档块的向量ID"""
        chunk.vector_id = vector_id
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        return chunk
    
    async def adelete_by_document(self, session: AsyncSession, *, document_id: UUID) -> int:
        """异步删除文档的所有块"""
        return await self.abulk_delete(
            session,
            filters={"document_id": document_id}
        )


# 创建全局仓库实例
document_repository = DocumentRepository()
document_chunk_repository = DocumentChunkRepository()