"""
文档数据模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import List, Optional

from .database import Base


class Document(Base):
    """文档模型"""
    
    __tablename__ = "documents"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # 基础信息
    filename = Column(String(255), nullable=False, comment="文件名")
    original_filename = Column(String(255), nullable=False, comment="原始文件名")
    file_path = Column(String(500), nullable=False, comment="文件路径")
    file_size = Column(Integer, nullable=False, comment="文件大小（字节）")
    file_type = Column(String(50), nullable=False, comment="文件类型")
    
    # 内容信息
    title = Column(String(500), comment="文档标题")
    description = Column(Text, comment="文档描述")
    content = Column(Text, comment="文档内容")
    content_hash = Column(String(64), unique=True, index=True, comment="内容哈希")
    
    # 处理状态
    status = Column(String(20), default="pending", comment="处理状态: pending, processing, completed, failed")
    error_message = Column(Text, comment="错误信息")
    
    # 统计信息
    total_chunks = Column(Integer, default=0, comment="总分块数")
    total_tokens = Column(Integer, default=0, comment="总token数")
    
    # 元数据
    metadata_ = Column("metadata", Text, comment="文档元数据（JSON格式）")
    
    # 向量化状态
    is_vectorized = Column(Boolean, default=False, comment="是否已向量化")
    vectorized_at = Column(DateTime(timezone=True), comment="向量化完成时间")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    processed_at = Column(DateTime(timezone=True), comment="处理完成时间")
    
    # 关系
    # chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "error_message": self.error_message,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
    
    @property
    def is_processed(self) -> bool:
        """是否已处理完成"""
        return self.status == "completed"
    
    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        return self.status == "processing"
    
    @property
    def has_error(self) -> bool:
        """是否有错误"""
        return self.status == "failed"