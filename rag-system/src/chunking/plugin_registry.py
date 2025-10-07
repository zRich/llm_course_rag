"""插件注册系统

实现切分策略插件的注册、发现、管理和调用机制。
这是第19节课插件化架构的核心管理组件。
"""

from typing import Dict, List, Optional, Type, Any, Callable
import logging
import inspect
from functools import wraps
import threading

from .strategy_interface import ChunkingStrategy, StrategyError
from .chunker import ChunkingConfig

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """策略注册器
    
    单例模式的策略注册和管理系统，支持策略的动态注册、发现和调用。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._strategies: Dict[str, Type[ChunkingStrategy]] = {}
            self._strategy_instances: Dict[str, ChunkingStrategy] = {}
            self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
            self._execution_stats: Dict[str, Dict[str, Any]] = {}
            self.logger = logging.getLogger(self.__class__.__name__)
            self._initialized = True
    
    def register_strategy(self, strategy_class: Type[ChunkingStrategy], 
                         name: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """注册策略类
        
        Args:
            strategy_class: 策略类
            name: 策略名称（可选，默认使用类名）
            metadata: 策略元数据
            
        Raises:
            ValueError: 如果策略类无效或名称已存在
        """
        if not issubclass(strategy_class, ChunkingStrategy):
            raise ValueError(f"策略类必须继承自ChunkingStrategy: {strategy_class}")
        
        strategy_name = name or strategy_class.__name__.lower().replace('strategy', '')
        
        if strategy_name in self._strategies:
            self.logger.warning(f"策略 '{strategy_name}' 已存在，将被覆盖")
        
        self._strategies[strategy_name] = strategy_class
        self._strategy_metadata[strategy_name] = metadata or {}
        self._execution_stats[strategy_name] = {
            'total_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'last_execution_time': None,
            'error_count': 0,
            'success_rate': 1.0
        }
        
        self.logger.info(f"策略 '{strategy_name}' 注册成功: {strategy_class.__name__}")
    
    def get_strategy(self, name: str, config: Optional[ChunkingConfig] = None, 
                    **kwargs) -> ChunkingStrategy:
        """获取策略实例
        
        Args:
            name: 策略名称
            config: 策略配置
            **kwargs: 策略特定参数
            
        Returns:
            ChunkingStrategy: 策略实例
            
        Raises:
            ValueError: 如果策略不存在
        """
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(f"策略 '{name}' 不存在。可用策略: {available}")
        
        # 创建新实例（每次调用都创建新实例以避免状态污染）
        strategy_class = self._strategies[name]
        try:
            instance = strategy_class(config=config, **kwargs)
            return instance
        except Exception as e:
            self.logger.error(f"创建策略实例失败 '{name}': {e}")
            raise StrategyError(f"创建策略实例失败: {e}")
    
    def get_cached_strategy(self, name: str, config: Optional[ChunkingConfig] = None,
                           **kwargs) -> ChunkingStrategy:
        """获取缓存的策略实例
        
        Args:
            name: 策略名称
            config: 策略配置
            **kwargs: 策略特定参数
            
        Returns:
            ChunkingStrategy: 缓存的策略实例
        """
        cache_key = f"{name}_{hash(str(config))}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in self._strategy_instances:
            self._strategy_instances[cache_key] = self.get_strategy(name, config, **kwargs)
        
        return self._strategy_instances[cache_key]
    
    def list_strategies(self) -> List[str]:
        """列出所有已注册的策略
        
        Returns:
            List[str]: 策略名称列表
        """
        return list(self._strategies.keys())
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """获取策略信息
        
        Args:
            name: 策略名称
            
        Returns:
            Dict[str, Any]: 策略信息
            
        Raises:
            ValueError: 如果策略不存在
        """
        if name not in self._strategies:
            raise ValueError(f"策略 '{name}' 不存在")
        
        strategy_class = self._strategies[name]
        metadata = self._strategy_metadata[name]
        stats = self._execution_stats[name]
        
        # 创建临时实例获取基本信息
        temp_instance = strategy_class()
        
        return {
            'name': name,
            'class_name': strategy_class.__name__,
            'description': temp_instance.get_strategy_description(),
            'module': strategy_class.__module__,
            'metadata': metadata,
            'execution_stats': stats.copy(),
            'parameters': self._get_strategy_parameters(strategy_class)
        }
    
    def _get_strategy_parameters(self, strategy_class: Type[ChunkingStrategy]) -> Dict[str, Any]:
        """获取策略参数信息
        
        Args:
            strategy_class: 策略类
            
        Returns:
            Dict[str, Any]: 参数信息
        """
        try:
            sig = inspect.signature(strategy_class.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'config'):
                    continue
                
                params[param_name] = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
            
            return params
        except Exception as e:
            self.logger.warning(f"获取策略参数失败: {e}")
            return {}
    
    def search_strategies(self, keyword: str = None, **filters) -> List[str]:
        """搜索策略
        
        Args:
            keyword: 关键词
            **filters: 过滤条件
            
        Returns:
            List[str]: 匹配的策略名称列表
        """
        strategies = self.list_strategies()
        
        if keyword:
            keyword = keyword.lower()
            strategies = [name for name in strategies 
                         if keyword in name.lower() or 
                         keyword in self._strategy_metadata.get(name, {}).get('description', '').lower()]
        
        return strategies


# 全局注册器实例
registry = StrategyRegistry()