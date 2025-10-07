"""数据库配置模块"""
import os
from typing import Optional
from sqlalchemy.engine import URL


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        """初始化数据库配置"""
        # 基础配置
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database = os.getenv("DB_NAME", "rag_system")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "postgres")
        
        # 连接池配置
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # 连接配置
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"
        self.echo_pool = os.getenv("DB_ECHO_POOL", "false").lower() == "true"
        
        # SSL配置
        self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")
        self.ssl_cert = os.getenv("DB_SSL_CERT")
        self.ssl_key = os.getenv("DB_SSL_KEY")
        self.ssl_ca = os.getenv("DB_SSL_CA")
    
    @property
    def sync_url(self) -> str:
        """获取同步数据库URL"""
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        ).render_as_string(hide_password=False)
    
    @property
    def async_url(self) -> str:
        """获取异步数据库URL"""
        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        ).render_as_string(hide_password=False)
    
    @property
    def alembic_url(self) -> str:
        """获取Alembic迁移URL"""
        return self.sync_url
    
    def get_connect_args(self) -> dict:
        """获取连接参数"""
        connect_args = {}
        
        # SSL配置
        if self.ssl_mode != "disable":
            connect_args["sslmode"] = self.ssl_mode
            if self.ssl_cert:
                connect_args["sslcert"] = self.ssl_cert
            if self.ssl_key:
                connect_args["sslkey"] = self.ssl_key
            if self.ssl_ca:
                connect_args["sslrootcert"] = self.ssl_ca
        
        return connect_args
    
    def get_engine_kwargs(self) -> dict:
        """获取引擎参数"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "echo": self.echo,
            "echo_pool": self.echo_pool,
            "connect_args": self.get_connect_args(),
        }
    
    def validate(self) -> bool:
        """验证配置"""
        required_fields = ["host", "port", "database", "username", "password"]
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"数据库配置缺少必需字段: {field}")
        
        if self.port < 1 or self.port > 65535:
            raise ValueError("数据库端口必须在1-65535之间")
        
        if self.pool_size < 1:
            raise ValueError("连接池大小必须大于0")
        
        return True


# 全局数据库配置实例
db_config = DatabaseConfig()