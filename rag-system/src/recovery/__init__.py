"""恢复机制模块

本模块提供自动恢复机制，包括重试策略、熔断器模式和降级服务。
用于提升系统的容错能力和可靠性。
"""

from .retry_mechanism import retry_with_backoff, RetryConfig, RetryStrategy
from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .fallback_service import FallbackService, FallbackStrategy
from typing import Callable
import functools

# 简化的重试装饰器
def retry_with_backoff(max_attempts: int = 3, base_delay: float = 0.5, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
    """简化的重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        strategy: 重试策略
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .retry_mechanism import retry_with_backoff as _retry_with_backoff, RetryConfig
            
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                strategy=strategy
            )
            
            return _retry_with_backoff(config, func.__name__)(func)(*args, **kwargs)
        
        return wrapper
    return decorator

__all__ = [
    'retry_with_backoff', 
    'RetryConfig', 
    'RetryStrategy',
    'CircuitBreaker', 
    'CircuitBreakerState',
    'FallbackService',
    'FallbackStrategy'
]