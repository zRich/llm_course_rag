from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataConnector(ABC):
    """
    数据连接器基类
    定义了所有数据连接器必须实现的抽象接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据连接器
        
        Args:
            config: 连接器配置参数
        """
        self.config = config
        self.connection = None
        self.is_connected = False
        self.last_sync_time = None
        
    @abstractmethod
    def connect(self) -> bool:
        """
        建立数据源连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开数据源连接
        
        Returns:
            bool: 断开是否成功
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        测试数据源连接
        
        Returns:
            bool: 连接是否正常
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        获取数据源结构信息
        
        Returns:
            Dict[str, Any]: 数据源结构信息
        """
        pass
    
    @abstractmethod
    def fetch_data(self, 
                   query: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        获取数据
        
        Args:
            query: 查询条件或SQL语句
            limit: 限制返回记录数
            offset: 偏移量
            
        Returns:
            Iterator[Dict[str, Any]]: 数据记录迭代器
        """
        pass
    
    @abstractmethod
    def fetch_incremental_data(self, 
                              last_sync_time: datetime,
                              query: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        获取增量数据
        
        Args:
            last_sync_time: 上次同步时间
            query: 查询条件
            
        Returns:
            Iterator[Dict[str, Any]]: 增量数据记录迭代器
        """
        pass
    
    @abstractmethod
    def get_total_count(self, query: Optional[str] = None) -> int:
        """
        获取数据总数
        
        Args:
            query: 查询条件
            
        Returns:
            int: 数据总数
        """
        pass
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证配置是否有效
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
        """
        # 创建临时实例来获取必需字段
        temp_instance = cls(config)
        required_fields = temp_instance.get_required_config_fields()
        return all(field in config for field in required_fields)
    
    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            List[str]: 必需的配置字段列表
        """
        pass
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        获取连接信息
        
        Returns:
            Dict[str, Any]: 连接信息
        """
        return {
            "is_connected": self.is_connected,
            "last_sync_time": self.last_sync_time,
            "config": {k: v for k, v in self.config.items() if k not in ['password', 'token', 'api_key']}
        }
    
    def update_last_sync_time(self, sync_time: datetime = None):
        """
        更新最后同步时间
        
        Args:
            sync_time: 同步时间，默认为当前时间
        """
        self.last_sync_time = sync_time or datetime.now()
        logger.info(f"Updated last sync time to: {self.last_sync_time}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()