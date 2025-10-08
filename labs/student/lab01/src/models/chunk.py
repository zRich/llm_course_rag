"""
文档分块数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Optional, List

from .database import Base


class Chunk(Base):
    """文档分块模型"""
    
    __tablename__ = "chunks"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # 外键关联
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # 分块信息
    chunk_index = Column(Integer, nullable=False, comment="分块索引")
    content = Column(Text, nullable=False, comment="分块内容")
    content_hash = Column(String(64), unique=True, index=True, comment="内容哈希")
    
    # 位置信息
    start_pos = Column(Integer, comment="在原文档中的起始位置")
    end_pos = Column(Integer, comment="在原文档中的结束位置")
    
    # 统计信息
    token_count = Column(Integer, default=0, comment="token数量")
    char_count = Column(Integer, default=0, comment="字符数量")
    
    # 向量信息
    vector_id = Column(String(100), comment="向量数据库中的ID")
    embedding_model = Column(String(100), comment="使用的embedding模型")
    embedding_dimension = Column(Integer, comment="向量维度")
    
    # 处理状态
    is_embedded = Column(Integer, default=0, comment="是否已向量化: 0-未处理, 1-已处理")
    embedding_error = Column(Text, comment="向量化错误信息")
    
    # 元数据
    metadata_ = Column("metadata", Text, comment="分块元数据（JSON格式）")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    embedded_at = Column(DateTime(timezone=True), comment="向量化时间")
    
    # 关系
    # document = relationship("Document", back_populates="chunks")
    
    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "vector_id": self.vector_id,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "is_embedded": bool(self.is_embedded),
            "embedding_error": self.embedding_error,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "embedded_at": self.embedded_at.isoformat() if self.embedded_at else None,
        }
    
    @property
    def is_vector_stored(self) -> bool:
        """是否已存储向量"""
        return self.is_embedded == 1 and self.vector_id is not None
    
    @property
    def has_embedding_error(self) -> bool:
        """是否有向量化错误"""
        return self.embedding_error is not None
    
    def get_preview(self, max_length: int = 100) -> str:
        """获取内容预览"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."