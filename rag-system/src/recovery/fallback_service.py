"""降级服务模块

实现服务降级和备用方案，在主服务不可用时提供基本功能。
"""

import time
import logging
from typing import Callable, Any, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """降级策略"""
    CACHE = "cache"              # 缓存策略
    DEFAULT_VALUE = "default"    # 默认值策略
    ALTERNATIVE_SERVICE = "alt"  # 备用服务策略
    SIMPLIFIED = "simplified"    # 简化服务策略
    FAIL_FAST = "fail_fast"      # 快速失败策略


@dataclass
class FallbackConfig:
    """降级配置"""
    strategy: FallbackStrategy = FallbackStrategy.DEFAULT_VALUE
    default_value: Any = None
    cache_ttl: float = 300.0  # 缓存TTL（秒）
    max_cache_size: int = 1000
    enable_logging: bool = True
    timeout: float = 5.0  # 降级服务超时时间


class FallbackException(Exception):
    """降级服务异常"""
    def __init__(self, message: str, strategy: FallbackStrategy):
        self.strategy = strategy
        super().__init__(message)


class FallbackStatistics:
    """降级统计"""
    def __init__(self):
        self.total_calls = 0
        self.fallback_calls = 0
        self.successful_fallbacks = 0
        self.failed_fallbacks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.strategy_usage = {strategy: 0 for strategy in FallbackStrategy}
        self.call_history = []
        self._lock = threading.RLock()
    
    def record_call(self, used_fallback: bool, strategy: Optional[FallbackStrategy] = None, 
                   success: bool = True, cache_hit: bool = False):
        """记录调用"""
        with self._lock:
            self.total_calls += 1
            
            if used_fallback:
                self.fallback_calls += 1
                if success:
                    self.successful_fallbacks += 1
                else:
                    self.failed_fallbacks += 1
                
                if strategy:
                    self.strategy_usage[strategy] += 1
            
            if cache_hit:
                self.cache_hits += 1
            elif used_fallback and strategy == FallbackStrategy.CACHE:
                self.cache_misses += 1
            
            # 记录调用历史
            self.call_history.append({
                'timestamp': time.time(),
                'used_fallback': used_fallback,
                'strategy': strategy.value if strategy else None,
                'success': success,
                'cache_hit': cache_hit
            })
            
            # 保持历史记录在合理范围内
            if len(self.call_history) > 1000:
                self.call_history = self.call_history[-500:]
    
    def get_fallback_rate(self) -> float:
        """获取降级率"""
        with self._lock:
            if self.total_calls == 0:
                return 0.0
            return self.fallback_calls / self.total_calls
    
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        with self._lock:
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests == 0:
                return 0.0
            return self.cache_hits / total_cache_requests
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        with self._lock:
            return {
                'total_calls': self.total_calls,
                'fallback_calls': self.fallback_calls,
                'successful_fallbacks': self.successful_fallbacks,
                'failed_fallbacks': self.failed_fallbacks,
                'fallback_rate': self.get_fallback_rate(),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.get_cache_hit_rate(),
                'strategy_usage': dict(self.strategy_usage),
                'recent_calls': self.call_history[-10:] if self.call_history else []
            }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # 检查是否过期
            if current_time > entry['expires_at']:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            # 更新访问时间
            self._access_times[key] = current_time
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            current_time = time.time()
            expires_at = current_time + (ttl or self.default_ttl)
            
            # 如果缓存已满，删除最久未访问的条目
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'created_at': current_time,
                'expires_at': expires_at
            }
            self._access_times[key] = current_time
    
    def _evict_lru(self):
        """删除最久未访问的条目"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self._cache)


class FallbackService:
    """降级服务
    
    提供多种降级策略，在主服务不可用时提供备用方案。
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None, name: str = "default"):
        """
        初始化降级服务
        
        Args:
            config: 降级配置
            name: 服务名称
        """
        self.config = config or FallbackConfig()
        self.name = name
        self.cache_manager = CacheManager(self.config.max_cache_size, self.config.cache_ttl)
        self.statistics = FallbackStatistics()
        self.alternative_services: List[Callable] = []
        self._lock = threading.RLock()
        
        logger.info(f"Fallback service '{name}' initialized with strategy: {self.config.strategy.value}")
    
    def add_alternative_service(self, service: Callable):
        """添加备用服务
        
        Args:
            service: 备用服务函数
        """
        with self._lock:
            self.alternative_services.append(service)
            logger.info(f"Added alternative service to '{self.name}'. Total: {len(self.alternative_services)}")
    
    def execute_with_fallback(self, primary_func: Callable, *args, **kwargs) -> Any:
        """执行主函数，失败时使用降级策略
        
        Args:
            primary_func: 主函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果或降级结果
        """
        # 首先尝试执行主函数
        try:
            result = primary_func(*args, **kwargs)
            self.statistics.record_call(used_fallback=False)
            return result
        except Exception as e:
            if self.config.enable_logging:
                logger.warning(f"Primary function failed in '{self.name}': {str(e)}. Using fallback strategy: {self.config.strategy.value}")
            
            # 执行降级策略
            return self._execute_fallback(*args, **kwargs)
    
    def _execute_fallback(self, *args, **kwargs) -> Any:
        """执行降级策略"""
        try:
            if self.config.strategy == FallbackStrategy.CACHE:
                return self._cache_fallback(*args, **kwargs)
            elif self.config.strategy == FallbackStrategy.DEFAULT_VALUE:
                return self._default_value_fallback()
            elif self.config.strategy == FallbackStrategy.ALTERNATIVE_SERVICE:
                return self._alternative_service_fallback(*args, **kwargs)
            elif self.config.strategy == FallbackStrategy.SIMPLIFIED:
                return self._simplified_fallback(*args, **kwargs)
            elif self.config.strategy == FallbackStrategy.FAIL_FAST:
                return self._fail_fast_fallback()
            else:
                raise FallbackException(f"Unknown fallback strategy: {self.config.strategy}", self.config.strategy)
        
        except Exception as e:
            self.statistics.record_call(used_fallback=True, strategy=self.config.strategy, success=False)
            raise FallbackException(f"Fallback execution failed: {str(e)}", self.config.strategy)
    
    def _cache_fallback(self, *args, **kwargs) -> Any:
        """缓存降级策略"""
        cache_key = self._generate_cache_key(*args, **kwargs)
        cached_value = self.cache_manager.get(cache_key)
        
        if cached_value is not None:
            self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.CACHE, 
                                      success=True, cache_hit=True)
            return cached_value
        else:
            self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.CACHE, 
                                      success=False, cache_hit=False)
            # 缓存未命中，使用默认值
            return self.config.default_value
    
    def _default_value_fallback(self) -> Any:
        """默认值降级策略"""
        self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.DEFAULT_VALUE, success=True)
        return self.config.default_value
    
    def _alternative_service_fallback(self, *args, **kwargs) -> Any:
        """备用服务降级策略"""
        if not self.alternative_services:
            raise FallbackException("No alternative services available", FallbackStrategy.ALTERNATIVE_SERVICE)
        
        # 尝试备用服务
        for i, alt_service in enumerate(self.alternative_services):
            try:
                result = alt_service(*args, **kwargs)
                self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.ALTERNATIVE_SERVICE, success=True)
                return result
            except Exception as e:
                logger.warning(f"Alternative service {i} failed in '{self.name}': {str(e)}")
                continue
        
        # 所有备用服务都失败
        raise FallbackException("All alternative services failed", FallbackStrategy.ALTERNATIVE_SERVICE)
    
    def _simplified_fallback(self, *args, **kwargs) -> Any:
        """简化服务降级策略"""
        # 这里可以实现简化版本的服务逻辑
        # 例如：返回基本信息而不是完整结果
        simplified_result = {
            'status': 'simplified',
            'message': 'Service is running in simplified mode',
            'timestamp': time.time(),
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        }
        
        self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.SIMPLIFIED, success=True)
        return simplified_result
    
    def _fail_fast_fallback(self) -> Any:
        """快速失败降级策略"""
        self.statistics.record_call(used_fallback=True, strategy=FallbackStrategy.FAIL_FAST, success=False)
        raise FallbackException("Fail-fast strategy activated", FallbackStrategy.FAIL_FAST)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 简单的缓存键生成策略
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return f"{self.name}:" + ":".join(key_parts)
    
    def cache_result(self, key: str, value: Any, ttl: Optional[float] = None):
        """手动缓存结果
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间
        """
        self.cache_manager.set(key, value, ttl)
        logger.debug(f"Cached result for key '{key}' in service '{self.name}'")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """获取缓存结果
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或None
        """
        return self.cache_manager.get(key)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_manager.clear()
        logger.info(f"Cache cleared for service '{self.name}'")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self.statistics.get_summary()
            stats.update({
                'name': self.name,
                'strategy': self.config.strategy.value,
                'cache_size': self.cache_manager.size(),
                'alternative_services_count': len(self.alternative_services),
                'config': {
                    'cache_ttl': self.config.cache_ttl,
                    'max_cache_size': self.config.max_cache_size,
                    'timeout': self.config.timeout
                }
            })
            return stats


def fallback_decorator(config: Optional[FallbackConfig] = None, name: str = "default"):
    """降级服务装饰器
    
    Args:
        config: 降级配置
        name: 服务名称
    """
    fallback_service = FallbackService(config, name)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return fallback_service.execute_with_fallback(func, *args, **kwargs)
        
        # 将降级服务实例附加到装饰后的函数上
        wrapper.fallback_service = fallback_service
        return wrapper
    
    return decorator


# 全局降级服务管理器
class FallbackServiceManager:
    """降级服务管理器"""
    
    def __init__(self):
        self._services: Dict[str, FallbackService] = {}
        self._lock = threading.RLock()
    
    def get_service(self, name: str, config: Optional[FallbackConfig] = None) -> FallbackService:
        """获取或创建降级服务
        
        Args:
            name: 服务名称
            config: 降级配置
            
        Returns:
            降级服务实例
        """
        with self._lock:
            if name not in self._services:
                self._services[name] = FallbackService(config, name)
            return self._services[name]
    
    def get_all_services(self) -> Dict[str, FallbackService]:
        """获取所有降级服务"""
        with self._lock:
            return self._services.copy()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        with self._lock:
            return {
                'total_services': len(self._services),
                'services': {
                    name: service.get_statistics()
                    for name, service in self._services.items()
                }
            }
    
    def clear_all_caches(self):
        """清空所有服务的缓存"""
        with self._lock:
            for service in self._services.values():
                service.clear_cache()


# 全局降级服务管理器实例
fallback_service_manager = FallbackServiceManager()