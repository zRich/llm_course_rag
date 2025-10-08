"""
数据库连接示例代码
演示SQLModel与PostgreSQL的连接配置和最佳实践
"""

from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from typing import Generator, Optional
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.echo = self._get_echo_setting()
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        self.pool_pre_ping = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
    
    def _get_database_url(self) -> str:
        """获取数据库连接URL"""
        # 优先使用环境变量
        if url := os.getenv("DATABASE_URL"):
            return url
        
        # 从单独的环境变量构建URL
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "password")
        database = os.getenv("DB_NAME", "rag_db")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    def _get_echo_setting(self) -> bool:
        """获取SQL回显设置"""
        env = os.getenv("ENVIRONMENT", "development")
        echo_env = os.getenv("DB_ECHO", "false")
        
        # 开发环境默认开启，生产环境默认关闭
        if env == "development":
            return echo_env.lower() != "false"
        else:
            return echo_env.lower() == "true"

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
    
    @property
    def engine(self) -> Engine:
        """获取数据库引擎（单例模式）"""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine
    
    def _create_engine(self) -> Engine:
        """创建数据库引擎"""
        logger.info(f"创建数据库引擎: {self._mask_password(self.config.database_url)}")
        
        # 根据环境选择连接池类型
        poolclass = QueuePool
        if "sqlite" in self.config.database_url:
            poolclass = StaticPool
        
        engine = create_engine(
            self.config.database_url,
            echo=self.config.echo,
            poolclass=poolclass,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            # 连接参数
            connect_args={
                "options": "-c timezone=utc"  # 设置时区为UTC
            } if "postgresql" in self.config.database_url else {}
        )
        
        return engine
    
    def _mask_password(self, url: str) -> str:
        """隐藏URL中的密码"""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)
    
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话"""
        with Session(self.engine) as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"数据库会话错误: {e}")
                session.rollback()
                raise
            finally:
                session.close()
    
    @contextmanager
    def get_session_context(self):
        """获取数据库会话上下文管理器"""
        with Session(self.engine) as session:
            try:
                yield session
                session.commit()
            except Exception as e:
                logger.error(f"数据库事务错误: {e}")
                session.rollback()
                raise
            finally:
                session.close()
    
    def create_tables(self):
        """创建所有数据库表"""
        logger.info("创建数据库表...")
        SQLModel.metadata.create_all(self.engine)
        logger.info("数据库表创建完成")
    
    def drop_tables(self):
        """删除所有数据库表（仅用于测试）"""
        logger.warning("删除所有数据库表...")
        SQLModel.metadata.drop_all(self.engine)
        logger.info("数据库表删除完成")
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_session_context() as session:
                result = session.exec("SELECT 1").first()
                logger.info(f"数据库连接测试成功: {result}")
                return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """获取连接信息"""
        return {
            "url": self._mask_password(self.config.database_url),
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "pool_recycle": self.config.pool_recycle,
            "pool_pre_ping": self.config.pool_pre_ping,
            "echo": self.config.echo
        }

# 全局数据库管理器实例
db_manager = DatabaseManager()

# 便捷函数
def get_session() -> Generator[Session, None, None]:
    """获取数据库会话的便捷函数"""
    yield from db_manager.get_session()

def get_session_context():
    """获取数据库会话上下文的便捷函数"""
    return db_manager.get_session_context()

def create_tables():
    """创建数据库表的便捷函数"""
    db_manager.create_tables()

def test_connection() -> bool:
    """测试数据库连接的便捷函数"""
    return db_manager.test_connection()

# 健康检查函数
async def health_check() -> dict:
    """异步健康检查函数"""
    try:
        connection_ok = test_connection()
        return {
            "database": "healthy" if connection_ok else "unhealthy",
            "connection_info": db_manager.get_connection_info()
        }
    except Exception as e:
        return {
            "database": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    # 测试脚本
    print("=== 数据库连接测试 ===")
    
    # 显示配置信息
    config = DatabaseConfig()
    print(f"数据库URL: {db_manager._mask_password(config.database_url)}")
    print(f"连接池大小: {config.pool_size}")
    print(f"最大溢出: {config.max_overflow}")
    print(f"SQL回显: {config.echo}")
    
    # 测试连接
    if test_connection():
        print("✅ 数据库连接成功")
        
        # 测试会话创建
        try:
            with get_session_context() as session:
                result = session.exec("SELECT current_timestamp").first()
                print(f"✅ 会话测试成功，当前时间: {result}")
        except Exception as e:
            print(f"❌ 会话测试失败: {e}")
    else:
        print("❌ 数据库连接失败")
    
    print("=== 测试完成 ===")