"""数据库连接管理模块"""
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlmodel import SQLModel
from .config import db_config
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        """初始化数据库管理器"""
        self._sync_engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._sync_session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """初始化数据库连接"""
        if self._initialized:
            return
        
        try:
            # 验证配置
            db_config.validate()
            
            # 创建同步引擎
            self._sync_engine = create_engine(
                db_config.sync_url,
                **db_config.get_engine_kwargs()
            )
            
            # 创建异步引擎
            self._async_engine = create_async_engine(
                db_config.async_url,
                **db_config.get_engine_kwargs()
            )
            
            # 创建会话工厂
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                class_=Session,
                expire_on_commit=False
            )
            
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._initialized = True
            logger.info("数据库连接初始化成功")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    async def create_tables(self) -> None:
        """创建数据库表"""
        if not self._async_engine:
            raise RuntimeError("数据库未初始化")
        
        try:
            async with self._async_engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"数据库表创建失败: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """删除数据库表"""
        if not self._async_engine:
            raise RuntimeError("数据库未初始化")
        
        try:
            async with self._async_engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
            logger.info("数据库表删除成功")
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    async def check_connection(self) -> bool:
        """检查数据库连接"""
        if not self._async_engine:
            return False
        
        try:
            async with self._async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            return False
    
    def get_sync_session(self) -> Session:
        """获取同步会话"""
        if not self._sync_session_factory:
            raise RuntimeError("数据库未初始化")
        return self._sync_session_factory()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步会话上下文管理器"""
        if not self._async_session_factory:
            raise RuntimeError("数据库未初始化")
        
        async with self._async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_async_session_direct(self) -> AsyncSession:
        """直接获取异步会话（需要手动管理）"""
        if not self._async_session_factory:
            raise RuntimeError("数据库未初始化")
        return self._async_session_factory()
    
    async def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self._async_engine:
                await self._async_engine.dispose()
            if self._sync_engine:
                self._sync_engine.dispose()
            
            self._sync_engine = None
            self._async_engine = None
            self._sync_session_factory = None
            self._async_session_factory = None
            self._initialized = False
            
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")
            raise
    
    @property
    def sync_engine(self) -> Engine:
        """获取同步引擎"""
        if not self._sync_engine:
            raise RuntimeError("数据库未初始化")
        return self._sync_engine
    
    @property
    def async_engine(self) -> AsyncEngine:
        """获取异步引擎"""
        if not self._async_engine:
            raise RuntimeError("数据库未初始化")
        return self._async_engine
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 便捷函数
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """获取异步数据库会话"""
    async with db_manager.get_async_session() as session:
        yield session


def get_sync_session() -> Session:
    """获取同步数据库会话"""
    return db_manager.get_sync_session()


async def init_database() -> None:
    """初始化数据库"""
    db_manager.initialize()
    await db_manager.create_tables()


async def close_database() -> None:
    """关闭数据库连接"""
    await db_manager.close()


async def check_database_health() -> dict:
    """检查数据库健康状态"""
    try:
        is_connected = await db_manager.check_connection()
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "initialized": db_manager.is_initialized,
            "database": db_config.database,
            "host": db_config.host,
            "port": db_config.port
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "initialized": db_manager.is_initialized,
            "error": str(e)
        }