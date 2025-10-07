"""故障注入模块

本模块提供故障注入框架，用于测试系统的容错能力和恢复机制。
包含故障类型定义、故障注入器和统计功能。
"""

from .fault_injector import FaultInjector, FaultType
from .fault_statistics import FaultStatistics
from typing import Callable
import functools

# 全局故障注入器实例
global_fault_injector = FaultInjector(enabled=True, global_failure_rate=0.1)

def fault_injection_decorator(fault_type: FaultType, probability: float = 0.1):
    """简化的故障注入装饰器
    
    Args:
        fault_type: 故障类型
        probability: 故障概率
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 配置故障类型
            global_fault_injector.configure_fault(fault_type, probability=probability)
            
            # 尝试注入故障
            try:
                if global_fault_injector.should_inject_fault(func.__name__):
                    global_fault_injector.inject_fault(fault_type, func.__name__)
            except Exception:
                # 如果故障注入失败，重新抛出异常
                raise
            
            # 执行原函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

__all__ = ['FaultInjector', 'FaultType', 'FaultStatistics', 'fault_injection_decorator', 'global_fault_injector']