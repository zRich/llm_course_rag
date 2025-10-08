"""
应用配置设置
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME: str = Field(
        default="RAG系统",
        description="应用名称"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="应用版本"
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="运行环境"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="应用监听地址"
    )
    PORT: int = Field(
        default=8000,
        description="应用监听端口"
    )
    WORKERS: int = Field(
        default=1,
        description="工作进程数"
    )
    ALLOWED_HOSTS: list = Field(
        default=["*"],
        description="允许的主机列表"
    )
    
    # 目录配置
    UPLOAD_DIR: str = Field(
        default="uploads",
        description="文件上传目录"
    )
    LOG_DIR: str = Field(
        default="logs",
        description="日志目录"
    )
    
    # 数据库配置
    database_url: str = Field(
        default="postgresql://rag_user:rag_password@localhost:15432/rag_db",
        description="数据库连接URL"
    )
    
    # Redis配置
    redis_url: str = Field(
        default="redis://localhost:16379/0",
        description="Redis连接URL"
    )
    
    # Qdrant配置
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant服务URL"
    )
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant主机地址"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant端口"
    )
    qdrant_collection_name: str = Field(
        default="rag_documents",
        description="Qdrant集合名称"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API密钥"
    )
    embedding_dimensions: int = Field(
        default=384,
        description="嵌入向量维度"
    )
    
    # 火山引擎配置
    volcengine_api_key: Optional[str] = Field(
        default=None,
        description="火山引擎API密钥"
    )
    volcengine_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="火山引擎API基础URL"
    )
    volcengine_model: str = Field(
        default="doubao-seed-1-6-250615",
        description="火山引擎模型名称"
    )
    volcengine_embedding_model: str = Field(
        default="doubao-embedding-v1",
        description="火山引擎Embedding模型名称"
    )
    
    # Embedding模型配置
    embedding_provider: str = Field(
        default="local",
        description="嵌入模型提供者类型 (local/api)"
    )
    
    # 本地嵌入模型配置
    local_embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="本地嵌入模型名称"
    )
    local_embedding_dimension: int = Field(
        default=384,
        description="本地嵌入模型向量维度"
    )
    
    # API嵌入模型配置 (向后兼容)
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="嵌入模型名称 (兼容字段)"
    )
    embedding_dimension: int = Field(
        default=384,
        description="嵌入模型向量维度 (兼容字段)"
    )
    
    # 支持的本地嵌入模型配置
    supported_local_models: dict = Field(
        default={
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
                "dimension": 384,
                "description": "多语言释义模型，优秀的中英文支持，推荐用于中文RAG"
            },
            "shibing624/text2vec-base-chinese": {
                "dimension": 768,
                "description": "专门的中文嵌入模型，中文语义理解能力强"
            },
            "BAAI/bge-small-zh-v1.5": {
                "dimension": 512,
                "description": "BGE中文小模型，高质量中文嵌入，性能优异"
            },
            "sentence-transformers/distiluse-base-multilingual-cased": {
                "dimension": 512,
                "description": "多语言通用模型，支持中文，平衡性能和速度"
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "dimension": 384,
                "description": "轻量级多语言模型，适合快速实验，基础中文支持"
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "dimension": 768,
                "description": "高质量英文模型，性能较好，有限中文支持"
            }
        },
        description="支持的本地嵌入模型配置"
    )
    
    # 应用配置
    app_host: str = Field(
        default="0.0.0.0",
        description="应用监听地址"
    )
    app_port: int = Field(
        default=8000,
        description="应用监听端口"
    )
    debug: bool = Field(
        default=True,
        description="调试模式"
    )
    
    # 文档处理配置
    chunk_size: int = Field(
        default=500,
        description="文本分块大小"
    )
    chunk_overlap: int = Field(
        default=50,
        description="文本分块重叠大小"
    )
    max_file_size: int = Field(
        default=10485760,  # 10MB
        description="最大文件大小（字节）"
    )
    
    # 检索配置
    top_k: int = Field(
        default=5,
        description="检索返回的文档数量"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="相似度阈值"
    )
    
    # LLM配置
    max_tokens: int = Field(
        default=2000,
        description="生成的最大token数"
    )
    temperature: float = Field(
        default=0.7,
        description="生成温度"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()