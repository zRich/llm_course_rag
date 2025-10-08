"""
用户模型模板 - Exercise 1
请根据注释提示完成用户模型的定义
"""

# TODO: 导入必要的模块
# 提示：需要从sqlmodel导入SQLModel, Field, Relationship
# 提示：需要从typing导入Optional, List, TYPE_CHECKING
# 提示：需要从datetime导入datetime
# 提示：需要从uuid导入UUID, uuid4

# 导入语句
from sqlmodel import _____, _____, _____
from typing import _____, _____, _____
from datetime import _____
from uuid import _____, _____

# TODO: 添加类型检查导入
if _____:
    from .document_model import Document

class UserBase(SQLModel):
    """用户基础模型 - 定义共享字段"""
    
    # TODO: 定义email字段
    # 提示：应该是str类型，需要unique=True和index=True
    email: _____ = Field(
        unique=_____,
        index=_____,
        description="用户邮箱地址"
    )
    
    # TODO: 定义username字段
    # 提示：应该是str类型，需要unique=True和index=True
    username: _____ = Field(
        unique=_____,
        index=_____,
        description="用户名"
    )
    
    # TODO: 定义is_active字段
    # 提示：应该是bool类型，默认值为True
    is_active: _____ = Field(
        default=_____,
        description="账户是否激活"
    )
    
    # TODO: 定义created_at字段
    # 提示：应该是datetime类型，使用default_factory=datetime.utcnow
    created_at: _____ = Field(
        default_factory=_____,
        description="创建时间"
    )

class User(UserBase, table=_____):
    """用户表模型"""
    
    # TODO: 定义id字段作为主键
    # 提示：应该是Optional[UUID]类型，使用uuid4作为默认值，设置primary_key=True
    id: _____[_____] = Field(
        default_factory=_____,
        primary_key=_____,
        description="用户唯一标识"
    )
    
    # TODO: 定义password_hash字段
    # 提示：应该是str类型，用于存储密码哈希值
    password_hash: _____ = Field(
        description="密码哈希值"
    )
    
    # TODO: 定义与文档的关系
    # 提示：应该是List["Document"]类型，使用Relationship，设置back_populates="user"
    documents: _____["Document"] = Relationship(
        back_populates="_____"
    )
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "username": "johndoe",
                "is_active": True
            }
        }

class UserCreate(UserBase):
    """用户创建模型"""
    
    # TODO: 添加password字段
    # 提示：应该是str类型，用于接收明文密码
    password: _____ = Field(
        description="用户密码"
    )

class UserRead(UserBase):
    """用户读取模型"""
    
    # TODO: 添加id字段
    # 提示：应该是UUID类型（不是Optional）
    id: _____

# Exercise检查清单：
# □ 所有TODO项目都已完成
# □ 导入语句正确
# □ 字段类型定义正确
# □ Field参数设置正确
# □ 关系定义正确
# □ 代码可以正常运行