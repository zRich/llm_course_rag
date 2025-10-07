"""数据库模块"""
from .config import DatabaseConfig, db_config
from .connection import (
    DatabaseManager,
    db_manager,
    get_sync_session,
    get_async_session,
    init_database,
    close_database,
    check_database_health
)
from .init_db import (
    create_database_if_not_exists,
    create_extensions,
    create_indexes,
    create_default_admin,
    create_default_configs,
    init_database as init_db,
    reset_database
)

__all__ = [
    # 配置
    "DatabaseConfig",
    "db_config",
    
    # 连接管理
    "DatabaseManager",
    "db_manager",
    "get_sync_session",
    "get_async_session",
    "init_database",
    "close_database",
    "check_database_health",
    
    # 初始化
    "create_database_if_not_exists",
    "create_extensions",
    "create_indexes",
    "create_default_admin",
    "create_default_configs",
    "init_db",
    "reset_database",
]