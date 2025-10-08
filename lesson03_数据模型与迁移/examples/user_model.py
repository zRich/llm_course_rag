"""
用户模型示例代码
演示SQLModel的基础使用方法和最佳实践
"""

from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from .document_model import Document

class UserBase(SQLModel):
    """用户基础模型 - 定义共享字段"""
    email: str = Field(
        unique=True, 
        index=True, 
        description="用户邮箱地址",
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    username: str = Field(
        unique=True, 
        index=True, 
        min_length=3, 
        max_length=50,
        description="用户名"
    )
    full_name: Optional[str] = Field(
        default=None, 
        max_length=100,
        description="用户全名"
    )
    is_active: bool = Field(
        default=True, 
        description="账户是否激活"
    )
    is_superuser: bool = Field(
        default=False, 
        description="是否为超级用户"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="创建时间"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="最后更新时间"
    )

class User(UserBase, table=True):
    """用户表模型"""
    __tablename__ = "users"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4, 
        primary_key=True,
        description="用户唯一标识"
    )
    password_hash: str = Field(
        description="密码哈希值"
    )
    last_login: Optional[datetime] = Field(
        default=None,
        description="最后登录时间"
    )
    
    # 关系定义
    documents: List["Document"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "is_active": True,
                "is_superuser": False
            }
        }

class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(
        min_length=8,
        description="用户密码"
    )

class UserUpdate(SQLModel):
    """用户更新模型"""
    email: Optional[str] = Field(
        default=None,
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    username: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=50
    )
    full_name: Optional[str] = Field(
        default=None,
        max_length=100
    )
    is_active: Optional[bool] = None
    password: Optional[str] = Field(
        default=None,
        min_length=8
    )

class UserRead(UserBase):
    """用户读取模型"""
    id: UUID
    last_login: Optional[datetime] = None

class UserReadWithDocuments(UserRead):
    """包含文档的用户读取模型"""
    documents: List["DocumentRead"] = []

# 避免循环导入的前向引用
from .document_model import DocumentRead
UserReadWithDocuments.model_rebuild()