#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗配置管理模块

提供文本清洗过程中的各种配置参数管理，包括：
- 清洗规则配置
- 质量评估阈值
- API服务配置
- 日志配置等
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class CleaningConfig:
    """文本清洗配置类"""
    
    # 基础清洗配置
    remove_extra_spaces: bool = True
    remove_special_chars: bool = True
    normalize_unicode: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    
    # 噪声检测配置
    min_text_length: int = 10
    max_repetition_ratio: float = 0.3
    encoding_confidence_threshold: float = 0.8
    
    # 质量评估配置
    min_quality_score: float = 0.6
    readability_weight: float = 0.25
    completeness_weight: float = 0.25
    consistency_weight: float = 0.2
    density_weight: float = 0.15
    structure_weight: float = 0.15
    
    # 批处理配置
    batch_size: int = 100
    max_workers: int = 4
    
    # 保留的特殊字符
    keep_chars: List[str] = field(default_factory=lambda: ['.', ',', '!', '?', ';', ':', '-', '_'])
    
    # 需要移除的特殊字符模式
    remove_patterns: List[str] = field(default_factory=lambda: [
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # 控制字符
        r'[\u200B-\u200D\uFEFF]',  # 零宽字符
        r'[\u2000-\u206F]',  # 通用标点符号
    ])


@dataclass
class APIConfig:
    """API服务配置类"""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    
    # 请求限制
    max_text_length: int = 100000  # 最大文本长度
    rate_limit: int = 100  # 每分钟请求限制
    
    # CORS配置
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LogConfig:
    """日志配置类"""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "text_cleaning.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.cleaning_config = CleaningConfig()
        self.api_config = APIConfig()
        self.log_config = LogConfig()
        
        # 从环境变量加载配置
        self._load_from_env()
        
        # 从配置文件加载配置（如果存在）
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # API配置
        if os.getenv('API_HOST'):
            self.api_config.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api_config.port = int(os.getenv('API_PORT'))
        if os.getenv('API_DEBUG'):
            self.api_config.debug = os.getenv('API_DEBUG').lower() == 'true'
        
        # 清洗配置
        if os.getenv('MIN_TEXT_LENGTH'):
            self.cleaning_config.min_text_length = int(os.getenv('MIN_TEXT_LENGTH'))
        if os.getenv('MIN_QUALITY_SCORE'):
            self.cleaning_config.min_quality_score = float(os.getenv('MIN_QUALITY_SCORE'))
        if os.getenv('BATCH_SIZE'):
            self.cleaning_config.batch_size = int(os.getenv('BATCH_SIZE'))
        
        # 日志配置
        if os.getenv('LOG_LEVEL'):
            self.log_config.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.log_config.file_path = os.getenv('LOG_FILE')
    
    def _load_from_file(self, config_file: str):
        """从配置文件加载配置"""
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新清洗配置
            if 'cleaning' in config_data:
                cleaning_data = config_data['cleaning']
                for key, value in cleaning_data.items():
                    if hasattr(self.cleaning_config, key):
                        setattr(self.cleaning_config, key, value)
            
            # 更新API配置
            if 'api' in config_data:
                api_data = config_data['api']
                for key, value in api_data.items():
                    if hasattr(self.api_config, key):
                        setattr(self.api_config, key, value)
            
            # 更新日志配置
            if 'logging' in config_data:
                log_data = config_data['logging']
                for key, value in log_data.items():
                    if hasattr(self.log_config, key):
                        setattr(self.log_config, key, value)
                        
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        try:
            import json
            from dataclasses import asdict
            
            config_data = {
                'cleaning': asdict(self.cleaning_config),
                'api': asdict(self.api_config),
                'logging': asdict(self.log_config)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_cleaning_config(self) -> CleaningConfig:
        """获取清洗配置"""
        return self.cleaning_config
    
    def get_api_config(self) -> APIConfig:
        """获取API配置"""
        return self.api_config
    
    def get_log_config(self) -> LogConfig:
        """获取日志配置"""
        return self.log_config
    
    def update_config(self, section: str, **kwargs):
        """更新配置"""
        if section == 'cleaning':
            config_obj = self.cleaning_config
        elif section == 'api':
            config_obj = self.api_config
        elif section == 'logging':
            config_obj = self.log_config
        else:
            raise ValueError(f"未知的配置节: {section}")
        
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                print(f"警告: 配置项 {key} 不存在于 {section} 配置中")


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器"""
    return config_manager


def create_sample_config(file_path: str = "config.json"):
    """创建示例配置文件"""
    sample_config = {
        "cleaning": {
            "remove_extra_spaces": True,
            "remove_special_chars": True,
            "normalize_unicode": True,
            "min_text_length": 10,
            "max_repetition_ratio": 0.3,
            "min_quality_score": 0.6,
            "batch_size": 100
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "max_text_length": 100000,
            "rate_limit": 100
        },
        "logging": {
            "level": "INFO",
            "file_path": "text_cleaning.log",
            "max_file_size": 10485760,
            "backup_count": 5
        }
    }
    
    try:
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        print(f"示例配置文件已创建: {file_path}")
    except Exception as e:
        print(f"创建配置文件失败: {e}")


if __name__ == "__main__":
    # 创建示例配置文件
    create_sample_config()
    
    # 测试配置管理器
    config = get_config()
    print(f"API端口: {config.get_api_config().port}")
    print(f"最小文本长度: {config.get_cleaning_config().min_text_length}")
    print(f"日志级别: {config.get_log_config().level}")