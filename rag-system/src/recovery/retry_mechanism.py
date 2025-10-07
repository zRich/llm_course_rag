"""重试机制模块

提供灵活的重试策略，包括指数退避、固定延迟等多种重试模式。
"""

import time
import random
import logging
from typing import Callable, Any, Optional, Type, Union, Tuple
from dataclasses import dataclass
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    
    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")


class RetryExhausted(Exception):
    """重试次数耗尽异常"""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last exception: {type(last_exception).__name__}: {last_exception}"
        )


class RetryStatistics:
    """重试统计"""
    def __init__(self):
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.total_delay = 0.0
        self.operation_stats = {}
    
    def record_attempt(self, operation: str, attempt: int, success: bool, delay: float = 0.0):
        """记录重试尝试"""
        self.total_attempts += 1
        self.total_delay += delay
        
        if success:
            self.successful_retries += 1
        else:
            self.failed_retries += 1
        
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'total_delay': 0.0
            }
        
        stats = self.operation_stats[operation]
        stats['attempts'] += 1
        stats['total_delay'] += delay
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
    
    def get_summary(self):
        """获取统计摘要"""
        return {
            'total_attempts': self.total_attempts,
            'successful_retries': self.successful_retries,
            'failed_retries': self.failed_retries,
            'success_rate': self.successful_retries / max(self.total_attempts, 1),
            'average_delay': self.total_delay / max(self.total_attempts, 1),
            'operation_stats': self.operation_stats
        }


# 全局重试统计实例
retry_statistics = RetryStatistics()


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """计算重试延迟时间
    
    Args:
        attempt: 当前尝试次数（从1开始）
        config: 重试配置
        
    Returns:
        延迟时间（秒）
    """
    if config.strategy == RetryStrategy.FIXED_DELAY:
        delay = config.base_delay
    elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = config.base_delay * attempt
    elif config.strategy == RetryStrategy.RANDOM_JITTER:
        delay = random.uniform(config.base_delay, config.base_delay * 2)
    else:
        delay = config.base_delay
    
    # 限制最大延迟
    delay = min(delay, config.max_delay)
    
    # 添加随机抖动
    if config.jitter and config.strategy != RetryStrategy.RANDOM_JITTER:
        jitter_range = delay * 0.1  # 10% 抖动
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # 确保延迟不为负数
    
    return delay


def is_retryable_exception(exception: Exception, config: RetryConfig) -> bool:
    """判断异常是否可重试
    
    Args:
        exception: 异常实例
        config: 重试配置
        
    Returns:
        是否可重试
    """
    # 检查非重试异常
    if config.non_retryable_exceptions:
        for exc_type in config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
    
    # 检查可重试异常
    for exc_type in config.retryable_exceptions:
        if isinstance(exception, exc_type):
            return True
    
    return False


def retry_with_backoff(config: Optional[RetryConfig] = None, operation_name: str = "unknown"):
    """重试装饰器
    
    Args:
        config: 重试配置，如果为None则使用默认配置
        operation_name: 操作名称，用于统计
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    logger.debug(f"Attempting {operation_name}, attempt {attempt}/{config.max_attempts}")
                    result = func(*args, **kwargs)
                    
                    # 记录成功的重试
                    if attempt > 1:
                        retry_statistics.record_attempt(operation_name, attempt, True)
                        logger.info(f"{operation_name} succeeded on attempt {attempt}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否可重试
                    if not is_retryable_exception(e, config):
                        logger.error(f"{operation_name} failed with non-retryable exception: {e}")
                        retry_statistics.record_attempt(operation_name, attempt, False)
                        raise e
                    
                    # 如果是最后一次尝试，不再重试
                    if attempt == config.max_attempts:
                        logger.error(f"{operation_name} failed after {attempt} attempts: {e}")
                        retry_statistics.record_attempt(operation_name, attempt, False)
                        break
                    
                    # 计算延迟时间
                    delay = calculate_delay(attempt, config)
                    
                    logger.warning(
                        f"{operation_name} failed on attempt {attempt}/{config.max_attempts}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # 记录失败的重试
                    retry_statistics.record_attempt(operation_name, attempt, False, delay)
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 所有重试都失败了
            raise RetryExhausted(config.max_attempts, last_exception)
        
        return wrapper
    return decorator


class RetryableOperation:
    """可重试操作类
    
    提供更灵活的重试控制，支持动态配置和状态跟踪。
    """
    
    def __init__(self, config: Optional[RetryConfig] = None, operation_name: str = "operation"):
        self.config = config or RetryConfig()
        self.operation_name = operation_name
        self.attempt_count = 0
        self.last_exception = None
        self.start_time = None
        self.total_delay = 0.0
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行可重试操作
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            RetryExhausted: 重试次数耗尽
        """
        self.start_time = time.time()
        self.attempt_count = 0
        self.total_delay = 0.0
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.attempt_count = attempt
            
            try:
                logger.debug(f"Executing {self.operation_name}, attempt {attempt}/{self.config.max_attempts}")
                result = func(*args, **kwargs)
                
                # 记录成功
                if attempt > 1:
                    retry_statistics.record_attempt(self.operation_name, attempt, True, self.total_delay)
                    logger.info(f"{self.operation_name} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                self.last_exception = e
                
                # 检查是否可重试
                if not is_retryable_exception(e, self.config):
                    logger.error(f"{self.operation_name} failed with non-retryable exception: {e}")
                    retry_statistics.record_attempt(self.operation_name, attempt, False, self.total_delay)
                    raise e
                
                # 如果是最后一次尝试，不再重试
                if attempt == self.config.max_attempts:
                    logger.error(f"{self.operation_name} failed after {attempt} attempts: {e}")
                    retry_statistics.record_attempt(self.operation_name, attempt, False, self.total_delay)
                    break
                
                # 计算延迟时间
                delay = calculate_delay(attempt, self.config)
                self.total_delay += delay
                
                logger.warning(
                    f"{self.operation_name} failed on attempt {attempt}/{self.config.max_attempts}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # 记录失败的重试
                retry_statistics.record_attempt(self.operation_name, attempt, False, delay)
                
                # 等待后重试
                time.sleep(delay)
        
        # 所有重试都失败了
        raise RetryExhausted(self.config.max_attempts, self.last_exception)
    
    def get_status(self) -> dict:
        """获取当前状态
        
        Returns:
            状态信息字典
        """
        return {
            'operation_name': self.operation_name,
            'attempt_count': self.attempt_count,
            'max_attempts': self.config.max_attempts,
            'total_delay': self.total_delay,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'last_exception': str(self.last_exception) if self.last_exception else None
        }


# 便捷函数
def simple_retry(func: Callable, max_attempts: int = 3, delay: float = 1.0, 
                operation_name: str = "operation") -> Any:
    """简单重试函数
    
    Args:
        func: 要重试的函数
        max_attempts: 最大尝试次数
        delay: 固定延迟时间
        operation_name: 操作名称
        
    Returns:
        函数执行结果
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=delay,
        strategy=RetryStrategy.FIXED_DELAY
    )
    
    operation = RetryableOperation(config, operation_name)
    return operation.execute(func)


def exponential_retry(func: Callable, max_attempts: int = 3, base_delay: float = 1.0,
                     max_delay: float = 60.0, operation_name: str = "operation") -> Any:
    """指数退避重试函数
    
    Args:
        func: 要重试的函数
        max_attempts: 最大尝试次数
        base_delay: 基础延迟时间
        max_delay: 最大延迟时间
        operation_name: 操作名称
        
    Returns:
        函数执行结果
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    )
    
    operation = RetryableOperation(config, operation_name)
    return operation.execute(func)