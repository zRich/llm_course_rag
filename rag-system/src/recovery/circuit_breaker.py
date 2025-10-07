"""熔断器模块

实现熔断器模式，防止系统在故障时继续调用失败的服务。
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 关闭状态，正常工作
    OPEN = "open"            # 开启状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，尝试恢复


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败阈值
    timeout: float = 60.0               # 超时时间（秒）
    half_open_max_calls: int = 3        # 半开状态最大调用次数
    success_threshold: int = 2          # 半开状态成功阈值
    monitoring_window: float = 60.0     # 监控窗口（秒）
    
    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")


class CircuitBreakerException(Exception):
    """熔断器异常"""
    def __init__(self, message: str, state: CircuitBreakerState):
        self.state = state
        super().__init__(message)


class CircuitBreakerStatistics:
    """熔断器统计"""
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.state_transitions = []
        self.last_failure_time = None
        self.last_success_time = None
        self.failure_rate = 0.0
        self.call_history = []  # 最近调用历史
    
    def record_call(self, success: bool, rejected: bool = False):
        """记录调用"""
        current_time = time.time()
        
        if rejected:
            self.rejected_calls += 1
        else:
            self.total_calls += 1
            if success:
                self.successful_calls += 1
                self.last_success_time = current_time
            else:
                self.failed_calls += 1
                self.last_failure_time = current_time
        
        # 记录调用历史
        self.call_history.append({
            'timestamp': current_time,
            'success': success,
            'rejected': rejected
        })
        
        # 保持历史记录在合理范围内
        if len(self.call_history) > 1000:
            self.call_history = self.call_history[-500:]
        
        # 更新失败率
        self._update_failure_rate()
    
    def record_state_transition(self, from_state: CircuitBreakerState, to_state: CircuitBreakerState):
        """记录状态转换"""
        self.state_transitions.append({
            'timestamp': time.time(),
            'from_state': from_state.value,
            'to_state': to_state.value
        })
    
    def _update_failure_rate(self):
        """更新失败率"""
        if self.total_calls > 0:
            self.failure_rate = self.failed_calls / self.total_calls
        else:
            self.failure_rate = 0.0
    
    def get_recent_failure_rate(self, window_seconds: float = 60.0) -> float:
        """获取最近时间窗口内的失败率"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_calls = [call for call in self.call_history if call['timestamp'] >= cutoff_time and not call['rejected']]
        
        if not recent_calls:
            return 0.0
        
        failed_calls = sum(1 for call in recent_calls if not call['success'])
        return failed_calls / len(recent_calls)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'rejected_calls': self.rejected_calls,
            'failure_rate': self.failure_rate,
            'recent_failure_rate': self.get_recent_failure_rate(),
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'state_transitions_count': len(self.state_transitions),
            'recent_state_transitions': self.state_transitions[-5:] if self.state_transitions else []
        }


class CircuitBreaker:
    """熔断器实现
    
    实现熔断器模式，在服务故障时自动切断请求，避免级联故障。
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None, name: str = "default"):
        """
        初始化熔断器
        
        Args:
            config: 熔断器配置
            name: 熔断器名称
        """
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.statistics = CircuitBreakerStatistics()
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            CircuitBreakerException: 熔断器开启时拒绝调用
        """
        with self._lock:
            current_state = self._get_current_state()
            
            if current_state == CircuitBreakerState.OPEN:
                self.statistics.record_call(success=False, rejected=True)
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is OPEN. Calls are rejected.",
                    CircuitBreakerState.OPEN
                )
            
            if current_state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self.statistics.record_call(success=False, rejected=True)
                    raise CircuitBreakerException(
                        f"Circuit breaker '{self.name}' is HALF_OPEN and max calls exceeded.",
                        CircuitBreakerState.HALF_OPEN
                    )
                
                self.half_open_calls += 1
        
        # 执行函数调用
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _get_current_state(self) -> CircuitBreakerState:
        """获取当前状态"""
        if self.state == CircuitBreakerState.OPEN:
            # 检查是否应该转换到半开状态
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.config.timeout:
                self._transition_to_half_open()
        
        return self.state
    
    def _on_success(self):
        """处理成功调用"""
        with self._lock:
            self.statistics.record_call(success=True)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                logger.debug(f"Circuit breaker '{self.name}' success in HALF_OPEN: {self.success_count}/{self.config.success_threshold}")
                
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # 重置失败计数
                self.failure_count = 0
    
    def _on_failure(self):
        """处理失败调用"""
        with self._lock:
            self.statistics.record_call(success=False)
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.warning(f"Circuit breaker '{self.name}' failure count: {self.failure_count}/{self.config.failure_threshold}")
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # 半开状态下的失败直接转换到开启状态
                self._transition_to_open()
    
    def _transition_to_open(self):
        """转换到开启状态"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()
        self.half_open_calls = 0
        self.success_count = 0
        
        self.statistics.record_state_transition(old_state, self.state)
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")
    
    def _transition_to_half_open(self):
        """转换到半开状态"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        
        self.statistics.record_state_transition(old_state, self.state)
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _transition_to_closed(self):
        """转换到关闭状态"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        
        self.statistics.record_state_transition(old_state, self.state)
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def get_state(self) -> CircuitBreakerState:
        """获取当前状态"""
        with self._lock:
            return self._get_current_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self.statistics.get_summary()
            stats.update({
                'name': self.name,
                'current_state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'half_open_calls': self.half_open_calls,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'timeout': self.config.timeout,
                    'half_open_max_calls': self.config.half_open_max_calls,
                    'success_threshold': self.config.success_threshold
                }
            })
            return stats
    
    def reset(self):
        """重置熔断器"""
        with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            
            if old_state != self.state:
                self.statistics.record_state_transition(old_state, self.state)
            
            logger.info(f"Circuit breaker '{self.name}' has been reset")
    
    def force_open(self):
        """强制开启熔断器"""
        with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.OPEN
            self.last_failure_time = time.time()
            
            if old_state != self.state:
                self.statistics.record_state_transition(old_state, self.state)
            
            logger.warning(f"Circuit breaker '{self.name}' has been forced to OPEN")
    
    def force_close(self):
        """强制关闭熔断器"""
        with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            
            if old_state != self.state:
                self.statistics.record_state_transition(old_state, self.state)
            
            logger.info(f"Circuit breaker '{self.name}' has been forced to CLOSED")


def circuit_breaker_decorator(config: Optional[CircuitBreakerConfig] = None, name: str = "default"):
    """熔断器装饰器
    
    Args:
        config: 熔断器配置
        name: 熔断器名称
    """
    breaker = CircuitBreaker(config, name)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # 将熔断器实例附加到装饰后的函数上
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


# 全局熔断器管理器
class CircuitBreakerManager:
    """熔断器管理器"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """获取或创建熔断器
        
        Args:
            name: 熔断器名称
            config: 熔断器配置
            
        Returns:
            熔断器实例
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(config, name)
            return self._breakers[name]
    
    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """获取所有熔断器"""
        with self._lock:
            return self._breakers.copy()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        with self._lock:
            return {
                'total_breakers': len(self._breakers),
                'breakers': {
                    name: breaker.get_statistics()
                    for name, breaker in self._breakers.items()
                }
            }
    
    def reset_all(self):
        """重置所有熔断器"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# 全局熔断器管理器实例
circuit_breaker_manager = CircuitBreakerManager()