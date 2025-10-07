"""数据库初始化脚本"""
import asyncio
import sys
from pathlib import Path
from typing import Optional
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from .connection import db_manager, get_async_session
from ..models import TABLE_MODELS, User, UserRole, SystemConfig
from ..config import get_settings
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


async def create_database_if_not_exists() -> None:
    """创建数据库（如果不存在）"""
    from .config import db_config
    
    # 连接到默认数据库
    temp_config = db_config
    temp_config.database = "postgres"
    
    try:
        # 创建临时引擎连接到postgres数据库
        from sqlalchemy.ext.asyncio import create_async_engine
        temp_engine = create_async_engine(temp_config.async_url)
        
        async with temp_engine.begin() as conn:
            # 检查数据库是否存在
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                {"db_name": db_config.database}
            )
            
            if not result.fetchone():
                # 创建数据库
                await conn.execute(text(f"CREATE DATABASE {db_config.database}"))
                logger.info(f"数据库 {db_config.database} 创建成功")
            else:
                logger.info(f"数据库 {db_config.database} 已存在")
        
        await temp_engine.dispose()
        
    except Exception as e:
        logger.error(f"创建数据库失败: {e}")
        raise


async def create_extensions() -> None:
    """创建数据库扩展"""
    extensions = [
        "uuid-ossp",  # UUID生成
        "pg_trgm",    # 文本相似度搜索
        "btree_gin",  # GIN索引支持
    ]
    
    try:
        async with db_manager.get_async_session() as session:
            for ext in extensions:
                try:
                    await session.execute(
                        text(f"CREATE EXTENSION IF NOT EXISTS \"{ext}\"")
                    )
                    logger.info(f"扩展 {ext} 创建成功")
                except ProgrammingError as e:
                    logger.warning(f"扩展 {ext} 创建失败: {e}")
            
            await session.commit()
    
    except Exception as e:
        logger.error(f"创建数据库扩展失败: {e}")
        raise


async def create_indexes() -> None:
    """创建数据库索引"""
    indexes = [
        # 用户表索引
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
        "CREATE INDEX IF NOT EXISTS idx_users_status ON users(status)",
        
        # 文档表索引
        "CREATE INDEX IF NOT EXISTS idx_documents_title ON documents USING gin(to_tsvector('english', title))",
        "CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents(owner_id)",
        "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
        "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)",
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)",
        "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
        
        # 文档块表索引
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON document_chunks(vector_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_content ON document_chunks USING gin(to_tsvector('english', content))",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk_index ON document_chunks(document_id, chunk_index)",
        
        # 查询历史表索引
        "CREATE INDEX IF NOT EXISTS idx_query_histories_user_id ON query_histories(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_query_histories_session_id ON query_histories(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_query_histories_type ON query_histories(query_type)",
        "CREATE INDEX IF NOT EXISTS idx_query_histories_status ON query_histories(status)",
        "CREATE INDEX IF NOT EXISTS idx_query_histories_created_at ON query_histories(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_query_histories_query_text ON query_histories USING gin(to_tsvector('english', query_text))",
        
        # 系统配置表索引
        "CREATE INDEX IF NOT EXISTS idx_system_configs_key ON system_configs(config_key)",
        "CREATE INDEX IF NOT EXISTS idx_system_configs_group ON system_configs(config_group)",
        "CREATE INDEX IF NOT EXISTS idx_system_configs_active ON system_configs(is_active)",
    ]
    
    try:
        async with db_manager.get_async_session() as session:
            for index_sql in indexes:
                try:
                    await session.execute(text(index_sql))
                    logger.debug(f"索引创建成功: {index_sql[:50]}...")
                except Exception as e:
                    logger.warning(f"索引创建失败: {e}")
            
            await session.commit()
            logger.info("数据库索引创建完成")
    
    except Exception as e:
        logger.error(f"创建数据库索引失败: {e}")
        raise


async def create_default_admin() -> None:
    """创建默认管理员用户"""
    try:
        async with db_manager.get_async_session() as session:
            # 检查是否已存在管理员
            from sqlalchemy import select
            result = await session.execute(
                select(User).where(User.role == UserRole.ADMIN)
            )
            admin_user = result.scalar_one_or_none()
            
            if not admin_user:
                # 创建默认管理员
                from werkzeug.security import generate_password_hash
                
                admin = User(
                    username="admin",
                    email="admin@example.com",
                    password_hash=generate_password_hash("admin123"),
                    full_name="系统管理员",
                    role=UserRole.ADMIN
                )
                
                session.add(admin)
                await session.commit()
                await session.refresh(admin)
                
                logger.info(f"默认管理员创建成功: {admin.username}")
            else:
                logger.info("管理员用户已存在")
    
    except Exception as e:
        logger.error(f"创建默认管理员失败: {e}")
        raise


async def create_default_configs() -> None:
    """创建默认系统配置"""
    default_configs = [
        {
            "config_key": "system.name",
            "config_value": "RAG System",
            "config_type": "string",
            "description": "系统名称",
            "config_group": "system",
            "sort_order": 1
        },
        {
            "config_key": "system.version",
            "config_value": "1.0.0",
            "config_type": "string",
            "description": "系统版本",
            "config_group": "system",
            "sort_order": 2
        },
        {
            "config_key": "embedding.model",
            "config_value": "text-embedding-ada-002",
            "config_type": "string",
            "description": "默认嵌入模型",
            "config_group": "embedding",
            "sort_order": 1
        },
        {
            "config_key": "embedding.dimension",
            "config_value": "1536",
            "config_type": "int",
            "description": "嵌入向量维度",
            "config_group": "embedding",
            "sort_order": 2
        },
        {
            "config_key": "llm.model",
            "config_value": "gpt-3.5-turbo",
            "config_type": "string",
            "description": "默认语言模型",
            "config_group": "llm",
            "sort_order": 1
        },
        {
            "config_key": "search.top_k",
            "config_value": "5",
            "config_type": "int",
            "description": "默认检索数量",
            "config_group": "search",
            "sort_order": 1
        },
        {
            "config_key": "search.score_threshold",
            "config_value": "0.7",
            "config_type": "float",
            "description": "相关性分数阈值",
            "config_group": "search",
            "sort_order": 2
        },
    ]
    
    try:
        async with db_manager.get_async_session() as session:
            from sqlalchemy import select
            
            for config_data in default_configs:
                # 检查配置是否已存在
                result = await session.execute(
                    select(SystemConfig).where(
                        SystemConfig.config_key == config_data["config_key"]
                    )
                )
                existing_config = result.scalar_one_or_none()
                
                if not existing_config:
                    config = SystemConfig(**config_data)
                    session.add(config)
                    logger.debug(f"创建配置: {config_data['config_key']}")
            
            await session.commit()
            logger.info("默认系统配置创建完成")
    
    except Exception as e:
        logger.error(f"创建默认系统配置失败: {e}")
        raise


async def init_database() -> None:
    """初始化数据库"""
    logger.info("开始初始化数据库...")
    
    try:
        # 1. 创建数据库（如果不存在）
        await create_database_if_not_exists()
        
        # 2. 初始化数据库管理器
        db_manager.initialize()
        
        # 3. 创建数据库表
        await db_manager.create_tables()
        
        # 4. 创建数据库扩展
        await create_extensions()
        
        # 5. 创建索引
        await create_indexes()
        
        # 6. 创建默认管理员
        await create_default_admin()
        
        # 7. 创建默认配置
        await create_default_configs()
        
        logger.info("数据库初始化完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


async def reset_database() -> None:
    """重置数据库"""
    logger.warning("开始重置数据库...")
    
    try:
        # 初始化数据库管理器
        db_manager.initialize()
        
        # 删除所有表
        await db_manager.drop_tables()
        
        # 重新初始化
        await init_database()
        
        logger.info("数据库重置完成")
        
    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库初始化脚本")
    parser.add_argument("--reset", action="store_true", help="重置数据库")
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.reset:
        asyncio.run(reset_database())
    else:
        asyncio.run(init_database())