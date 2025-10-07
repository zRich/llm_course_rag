from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = "RAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # API配置
    api_prefix: str = "/api/v1"
    
    # 数据库配置
    database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_pool_pre_ping: bool = True
    
    # 数据库连接组件配置
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "rag_user"
    db_password: str = "rag_password"
    db_name: str = "rag_db"
    db_ssl_mode: str = "prefer"
    
    # Alembic配置
    alembic_config_path: str = "alembic.ini"
    alembic_script_location: str = "alembic"
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10
    redis_socket_timeout: int = 5
    
    # 向量数据库配置
    qdrant_url: str = "http://localhost:6333"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_timeout: int = 30
    
    # MinIO对象存储配置
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_name: str = "rag-documents"
    
    # OpenAI配置（后续课程会用到）
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # 文件存储配置
    upload_dir: str = str(PROJECT_ROOT / "uploads")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# 创建全局配置实例
settings = Settings()

# 获取配置实例的函数
def get_settings() -> Settings:
    """获取配置实例"""
    return settings

# 确保上传目录存在
os.makedirs(settings.upload_dir, exist_ok=True)

# 配置验证函数
def validate_config() -> bool:
    """验证配置是否正确"""
    try:
        # 检查必要的目录
        if not os.path.exists(settings.upload_dir):
            print(f"警告: 上传目录不存在: {settings.upload_dir}")
            return False
            
        # 检查端口范围
        if not (1 <= settings.port <= 65535):
            print(f"错误: 端口号无效: {settings.port}")
            return False
            
        # 检查数据库端口范围
        if not (1 <= settings.db_port <= 65535):
            print(f"错误: 数据库端口号无效: {settings.db_port}")
            return False
            
        # 检查数据库连接池配置
        if settings.database_pool_size <= 0:
            print(f"错误: 数据库连接池大小无效: {settings.database_pool_size}")
            return False
            
        if settings.database_max_overflow < 0:
            print(f"错误: 数据库连接池最大溢出无效: {settings.database_max_overflow}")
            return False
            
        # 检查Alembic配置文件
        alembic_config_file = PROJECT_ROOT / settings.alembic_config_path
        if not alembic_config_file.exists():
            print(f"警告: Alembic配置文件不存在: {alembic_config_file}")
            
        print("配置验证通过")
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False

# 获取配置信息的辅助函数
def get_config_info() -> dict:
    """获取配置信息摘要"""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "api_prefix": settings.api_prefix,
        "upload_dir": settings.upload_dir,
        "log_level": settings.log_level
    }

def get_database_config() -> dict:
    """获取数据库配置信息"""
    return {
        "database_url": settings.database_url,
        "db_host": settings.db_host,
        "db_port": settings.db_port,
        "db_name": settings.db_name,
        "db_user": settings.db_user,
        "database_pool_size": settings.database_pool_size,
        "database_max_overflow": settings.database_max_overflow,
        "database_pool_timeout": settings.database_pool_timeout,
        "database_pool_recycle": settings.database_pool_recycle,
        "database_echo": settings.database_echo
    }

if __name__ == "__main__":
    # 测试配置
    print("=== RAG系统配置信息 ===")
    config_info = get_config_info()
    for key, value in config_info.items():
        print(f"{key}: {value}")
    
    print("\n=== 数据库配置信息 ===")
    db_config = get_database_config()
    for key, value in db_config.items():
        if "password" in key.lower():
            print(f"{key}: {'*' * len(str(value))}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== 配置验证 ===")
    validate_config()