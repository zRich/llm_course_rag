"""
文档模型模板 - Exercise 3
请根据注释提示完成文档模型和关系映射的定义
"""

# TODO: 导入必要的模块
from sqlmodel import _____, _____, _____
from typing import _____, _____, _____
from datetime import _____
from uuid import _____, _____

# TODO: 添加类型检查导入
if _____:
    from .user_model import User

class DocumentBase(SQLModel):
    """文档基础模型"""
    
    # TODO: 定义title字段
    # 提示：str类型，需要index=True
    title: _____ = Field(
        index=_____,
        description="文档标题"
    )
    
    # TODO: 定义content字段
    # 提示：str类型，存储文档内容
    content: _____ = Field(
        description="文档内容"
    )
    
    # TODO: 定义file_path字段（可选）
    # 提示：Optional[str]类型，默认值为None
    file_path: _____[_____] = Field(
        default=_____,
        description="文件存储路径"
    )
    
    # TODO: 定义file_size字段（可选）
    # 提示：Optional[int]类型，默认值为None
    file_size: _____[_____] = Field(
        default=_____,
        description="文件大小(字节)"
    )
    
    # TODO: 定义mime_type字段（可选）
    # 提示：Optional[str]类型，默认值为None
    mime_type: _____[_____] = Field(
        default=_____,
        description="MIME类型"
    )
    
    # TODO: 定义created_at字段
    # 提示：datetime类型，使用default_factory=datetime.utcnow
    created_at: _____ = Field(
        default_factory=_____,
        description="创建时间"
    )
    
    # TODO: 定义updated_at字段（可选）
    # 提示：Optional[datetime]类型，默认值为None
    updated_at: _____[_____] = Field(
        default=_____,
        description="最后更新时间"
    )

class Document(DocumentBase, table=_____):
    """文档表模型"""
    
    # TODO: 定义id字段作为主键
    # 提示：Optional[UUID]类型，使用uuid4作为默认值，设置primary_key=True
    id: _____[_____] = Field(
        default_factory=_____,
        primary_key=_____,
        description="文档唯一标识"
    )
    
    # TODO: 定义user_id外键字段
    # 提示：UUID类型，设置foreign_key="user.id"
    user_id: _____ = Field(
        foreign_key="_____",
        description="所属用户ID"
    )
    
    # TODO: 定义与用户的关系
    # 提示：Optional["User"]类型，使用Relationship，设置back_populates="documents"
    user: _____["User"] = Relationship(
        back_populates="_____"
    )
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "title": "示例文档",
                "content": "这是一个示例文档的内容...",
                "file_path": "/uploads/document.pdf",
                "file_size": 1024000,
                "mime_type": "application/pdf"
            }
        }

class DocumentCreate(DocumentBase):
    """文档创建模型"""
    # TODO: 添加user_id字段
    # 提示：UUID类型，用于指定文档所属用户
    user_id: _____ = Field(
        description="所属用户ID"
    )

class DocumentUpdate(SQLModel):
    """文档更新模型"""
    
    # TODO: 定义可更新的字段（都是可选的）
    # 提示：所有字段都应该是Optional类型，默认值为None
    title: _____[_____] = None
    content: _____[_____] = None
    file_path: _____[_____] = None
    file_size: _____[_____] = None
    mime_type: _____[_____] = None
    
    # TODO: 定义updated_at字段
    # 提示：datetime类型，使用default_factory=datetime.utcnow
    updated_at: _____ = Field(
        default_factory=_____,
        description="更新时间"
    )

class DocumentRead(DocumentBase):
    """文档读取模型"""
    
    # TODO: 添加必要的字段
    # 提示：需要id和user_id字段，都不是Optional类型
    id: _____
    user_id: _____

# TODO: 扩展Exercise - 创建包含用户信息的文档读取模型
class DocumentReadWithUser(DocumentRead):
    """包含用户信息的文档读取模型"""
    
    # TODO: 添加用户关系字段
    # 提示：Optional["UserRead"]类型，默认值为None
    user: _____["UserRead"] = None

# 避免循环导入的前向引用
# TODO: 导入UserRead并重建模型
# 提示：需要从.user_model导入UserRead，然后调用model_rebuild()
from .user_model import _____
DocumentReadWithUser._____()

# Exercise检查清单：
# □ 所有TODO项目都已完成
# □ 导入语句正确
# □ 字段类型定义正确
# □ 外键关系正确设置
# □ 关系映射正确定义
# □ 前向引用正确处理
# □ 扩展Exercise已完成
# □ 代码可以正常运行