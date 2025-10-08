"""
文档模型示例代码
演示关系映射和复杂字段类型的使用
"""

from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import Text, JSON
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum

if TYPE_CHECKING:
    from .user_model import User

class DocumentStatus(str, Enum):
    """文档状态枚举"""
    DRAFT = "draft"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class DocumentBase(SQLModel):
    """文档基础模型"""
    title: str = Field(
        index=True,
        min_length=1,
        max_length=200,
        description="文档标题"
    )
    content: str = Field(
        sa_column=Column(Text),
        description="文档内容"
    )
    summary: Optional[str] = Field(
        default=None,
        max_length=500,
        description="文档摘要"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="文件存储路径"
    )
    file_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="原始文件名"
    )
    file_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="文件大小(字节)"
    )
    mime_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="MIME类型"
    )
    status: DocumentStatus = Field(
        default=DocumentStatus.DRAFT,
        description="文档状态"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON),
        description="文档元数据"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="文档标签"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="创建时间"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="最后更新时间"
    )

class Document(DocumentBase, table=True):
    """文档表模型"""
    __tablename__ = "documents"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        description="文档唯一标识"
    )
    user_id: UUID = Field(
        foreign_key="users.id",
        description="所属用户ID"
    )
    
    # 关系定义
    user: Optional["User"] = Relationship(
        back_populates="documents"
    )
    vectors: List["DocumentVector"] = Relationship(
        back_populates="document",
        cascade_delete=True
    )
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "title": "示例文档",
                "content": "这是一个示例文档的内容...",
                "summary": "文档摘要",
                "status": "completed",
                "tags": ["技术", "教程"],
                "metadata": {
                    "source": "upload",
                    "language": "zh-CN",
                    "word_count": 1000
                }
            }
        }

class DocumentVector(SQLModel, table=True):
    """文档向量表模型"""
    __tablename__ = "document_vectors"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True
    )
    document_id: UUID = Field(
        foreign_key="documents.id",
        description="关联文档ID"
    )
    chunk_index: int = Field(
        ge=0,
        description="文档块索引"
    )
    chunk_text: str = Field(
        sa_column=Column(Text),
        description="文档块文本"
    )
    vector: List[float] = Field(
        description="向量数据"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    
    # 关系定义
    document: Optional[Document] = Relationship(
        back_populates="vectors"
    )

class DocumentCreate(DocumentBase):
    """文档创建模型"""
    pass

class DocumentUpdate(SQLModel):
    """文档更新模型"""
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200
    )
    content: Optional[str] = None
    summary: Optional[str] = Field(
        default=None,
        max_length=500
    )
    status: Optional[DocumentStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )

class DocumentRead(DocumentBase):
    """文档读取模型"""
    id: UUID
    user_id: UUID

class DocumentReadWithUser(DocumentRead):
    """包含用户信息的文档读取模型"""
    user: Optional["UserRead"] = None

class DocumentReadWithVectors(DocumentRead):
    """包含向量信息的文档读取模型"""
    vectors: List["DocumentVectorRead"] = []

class DocumentVectorRead(SQLModel):
    """文档向量读取模型"""
    id: UUID
    document_id: UUID
    chunk_index: int
    chunk_text: str
    created_at: datetime

# 避免循环导入的前向引用
from .user_model import UserRead
DocumentReadWithUser.model_rebuild()