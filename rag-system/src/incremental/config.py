"""增量更新系统配置"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class IncrementalConfig:
    """增量更新配置类"""
    
    # 基础配置
    data_directory: str = "./data"
    metadata_directory: str = "./metadata"
    log_level: str = "INFO"
    
    # 变更检测配置
    change_detection_enabled: bool = True
    hash_algorithm: str = "md5"
    file_extensions: list = field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx"])
    exclude_patterns: list = field(default_factory=lambda: ["*.tmp", "*.log", "__pycache__", ".git"])
    
    # 版本管理配置
    version_management_enabled: bool = True
    max_versions_per_document: int = 10
    version_cleanup_enabled: bool = True
    version_cleanup_days: int = 30
    
    # 增量索引配置
    incremental_indexing_enabled: bool = True
    batch_size: int = 10
    max_queue_size: int = 1000
    processing_timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # 冲突解决配置
    conflict_resolution_enabled: bool = True
    default_conflict_strategy: str = "latest_wins"
    auto_resolve_conflicts: bool = True
    conflict_backup_enabled: bool = True
    
    # RAG集成配置
    rag_integration_enabled: bool = True
    vector_collection_name: str = "documents"
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # 性能配置
    max_concurrent_tasks: int = 5
    memory_limit_mb: int = 1024
    disk_space_threshold_mb: int = 1024
    
    # 监控配置
    monitoring_enabled: bool = True
    metrics_collection_enabled: bool = True
    health_check_interval: int = 60
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录路径是绝对路径
        self.data_directory = os.path.abspath(self.data_directory)
        self.metadata_directory = os.path.abspath(self.metadata_directory)
        
        # 创建必要的目录
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "data_directory": self.data_directory,
            "metadata_directory": self.metadata_directory,
            "log_level": self.log_level,
            "change_detection_enabled": self.change_detection_enabled,
            "hash_algorithm": self.hash_algorithm,
            "file_extensions": self.file_extensions,
            "exclude_patterns": self.exclude_patterns,
            "version_management_enabled": self.version_management_enabled,
            "max_versions_per_document": self.max_versions_per_document,
            "version_cleanup_enabled": self.version_cleanup_enabled,
            "version_cleanup_days": self.version_cleanup_days,
            "incremental_indexing_enabled": self.incremental_indexing_enabled,
            "batch_size": self.batch_size,
            "max_queue_size": self.max_queue_size,
            "processing_timeout": self.processing_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "conflict_resolution_enabled": self.conflict_resolution_enabled,
            "default_conflict_strategy": self.default_conflict_strategy,
            "auto_resolve_conflicts": self.auto_resolve_conflicts,
            "conflict_backup_enabled": self.conflict_backup_enabled,
            "rag_integration_enabled": self.rag_integration_enabled,
            "vector_collection_name": self.vector_collection_name,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "memory_limit_mb": self.memory_limit_mb,
            "disk_space_threshold_mb": self.disk_space_threshold_mb,
            "monitoring_enabled": self.monitoring_enabled,
            "metrics_collection_enabled": self.metrics_collection_enabled,
            "health_check_interval": self.health_check_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IncrementalConfig':
        """从字典创建配置"""
        return cls(**data)
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'IncrementalConfig':
        """从文件加载配置"""
        if not os.path.exists(file_path):
            # 如果配置文件不存在，创建默认配置
            config = cls()
            config.save_to_file(file_path)
            return config
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def update(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 验证目录路径
            if not os.path.exists(self.data_directory):
                Path(self.data_directory).mkdir(parents=True, exist_ok=True)
            
            if not os.path.exists(self.metadata_directory):
                Path(self.metadata_directory).mkdir(parents=True, exist_ok=True)
            
            # 验证数值范围
            assert self.batch_size > 0, "batch_size必须大于0"
            assert self.max_queue_size > 0, "max_queue_size必须大于0"
            assert self.processing_timeout > 0, "processing_timeout必须大于0"
            assert self.retry_attempts >= 0, "retry_attempts必须大于等于0"
            assert self.retry_delay >= 0, "retry_delay必须大于等于0"
            assert self.max_versions_per_document > 0, "max_versions_per_document必须大于0"
            assert self.version_cleanup_days > 0, "version_cleanup_days必须大于0"
            assert self.chunk_size > 0, "chunk_size必须大于0"
            assert self.chunk_overlap >= 0, "chunk_overlap必须大于等于0"
            assert self.max_concurrent_tasks > 0, "max_concurrent_tasks必须大于0"
            assert self.memory_limit_mb > 0, "memory_limit_mb必须大于0"
            assert self.disk_space_threshold_mb > 0, "disk_space_threshold_mb必须大于0"
            assert self.health_check_interval > 0, "health_check_interval必须大于0"
            
            # 验证哈希算法
            assert self.hash_algorithm in ["md5", "sha1", "sha256"], "不支持的哈希算法"
            
            # 验证冲突解决策略
            valid_strategies = ["latest_wins", "oldest_wins", "auto_merge", "backup_and_replace", "skip"]
            assert self.default_conflict_strategy in valid_strategies, "不支持的冲突解决策略"
            
            # 验证日志级别
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            assert self.log_level in valid_log_levels, "不支持的日志级别"
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 全局配置实例
_config: Optional[IncrementalConfig] = None

def get_config(config_file: str = "incremental_config.json") -> IncrementalConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = IncrementalConfig.load_from_file(config_file)
        if not _config.validate():
            raise ValueError("配置验证失败")
    return _config

def set_config(config: IncrementalConfig) -> None:
    """设置全局配置实例"""
    global _config
    if not config.validate():
        raise ValueError("配置验证失败")
    _config = config

def reset_config() -> None:
    """重置全局配置实例"""
    global _config
    _config = None

# 环境变量配置覆盖
def load_config_from_env() -> Dict[str, Any]:
    """从环境变量加载配置覆盖"""
    env_config = {}
    
    # 基础配置
    if os.getenv("INCREMENTAL_DATA_DIR"):
        env_config["data_directory"] = os.getenv("INCREMENTAL_DATA_DIR")
    if os.getenv("INCREMENTAL_METADATA_DIR"):
        env_config["metadata_directory"] = os.getenv("INCREMENTAL_METADATA_DIR")
    if os.getenv("INCREMENTAL_LOG_LEVEL"):
        env_config["log_level"] = os.getenv("INCREMENTAL_LOG_LEVEL")
    
    # 批处理配置
    if os.getenv("INCREMENTAL_BATCH_SIZE"):
        env_config["batch_size"] = int(os.getenv("INCREMENTAL_BATCH_SIZE"))
    if os.getenv("INCREMENTAL_MAX_QUEUE_SIZE"):
        env_config["max_queue_size"] = int(os.getenv("INCREMENTAL_MAX_QUEUE_SIZE"))
    
    # RAG集成配置
    if os.getenv("INCREMENTAL_VECTOR_COLLECTION"):
        env_config["vector_collection_name"] = os.getenv("INCREMENTAL_VECTOR_COLLECTION")
    if os.getenv("INCREMENTAL_EMBEDDING_MODEL"):
        env_config["embedding_model"] = os.getenv("INCREMENTAL_EMBEDDING_MODEL")
    
    # 性能配置
    if os.getenv("INCREMENTAL_MAX_CONCURRENT_TASKS"):
        env_config["max_concurrent_tasks"] = int(os.getenv("INCREMENTAL_MAX_CONCURRENT_TASKS"))
    if os.getenv("INCREMENTAL_MEMORY_LIMIT_MB"):
        env_config["memory_limit_mb"] = int(os.getenv("INCREMENTAL_MEMORY_LIMIT_MB"))
    
    return env_config

def create_config_with_env_override(base_config_file: str = "incremental_config.json") -> IncrementalConfig:
    """创建带环境变量覆盖的配置"""
    # 加载基础配置
    config = IncrementalConfig.load_from_file(base_config_file)
    
    # 应用环境变量覆盖
    env_overrides = load_config_from_env()
    config.update(**env_overrides)
    
    # 验证配置
    if not config.validate():
        raise ValueError("配置验证失败")
    
    return config