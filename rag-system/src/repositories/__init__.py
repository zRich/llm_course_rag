"""仓库模块"""

# 基础仓库
from .base import BaseRepository

# 用户仓库
from .user import UserRepository, user_repository

# 文档仓库
from .document import (
    DocumentRepository,
    DocumentChunkRepository,
    document_repository,
    document_chunk_repository
)

# 查询仓库
from .query import (
    QueryHistoryRepository,
    SystemConfigRepository,
    query_history_repository,
    system_config_repository
)

__all__ = [
    # 基础仓库类
    "BaseRepository",
    
    # 用户仓库
    "UserRepository",
    "user_repository",
    
    # 文档仓库
    "DocumentRepository",
    "DocumentChunkRepository",
    "document_repository",
    "document_chunk_repository",
    
    # 查询仓库
    "QueryHistoryRepository",
    "SystemConfigRepository",
    "query_history_repository",
    "system_config_repository",
]

# 所有仓库实例的集合
REPOSITORIES = {
    "user": user_repository,
    "document": document_repository,
    "document_chunk": document_chunk_repository,
    "query_history": query_history_repository,
    "system_config": system_config_repository,
}