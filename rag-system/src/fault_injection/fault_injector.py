"""故障注入器模块

提供可配置的故障注入功能，支持多种故障类型和注入策略。
"""

import random
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """故障类型枚举"""
    NETWORK_TIMEOUT = "network_timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_ERROR = "memory_error"
    DISK_FULL = "disk_full"
    SLOW_RESPONSE = "slow_response"


@dataclass
class FaultConfig:
    """故障配置"""
    fault_type: FaultType
    probability: float = 0.1  # 故障概率
    delay_range: tuple = (1.0, 5.0)  # 延迟范围（秒）
    error_message: str = "Injected fault"
    enabled: bool = True


class FaultException(Exception):
    """故障注入异常"""
    def __init__(self, fault_type: FaultType, message: str):
        self.fault_type = fault_type
        self.message = message
        super().__init__(f"[{fault_type.value}] {message}")


class FaultInjector:
    """故障注入器
    
    用于在系统运行时注入各种类型的故障，测试系统的容错能力。
    """
    
    def __init__(self, enabled: bool = True, global_failure_rate: float = 0.1):
        """
        初始化故障注入器
        
        Args:
            enabled: 是否启用故障注入
            global_failure_rate: 全局故障率
        """
        self.enabled = enabled
        self.global_failure_rate = global_failure_rate
        self.fault_configs: Dict[str, FaultConfig] = {}
        self.fault_statistics = {}
        self._setup_default_faults()
    
    def _setup_default_faults(self):
        """设置默认故障配置"""
        default_faults = [
            FaultConfig(
                FaultType.NETWORK_TIMEOUT,
                probability=0.05,
                delay_range=(2.0, 8.0),
                error_message="Network timeout occurred"
            ),
            FaultConfig(
                FaultType.SERVICE_UNAVAILABLE,
                probability=0.03,
                error_message="Service temporarily unavailable"
            ),
            FaultConfig(
                FaultType.DATA_CORRUPTION,
                probability=0.02,
                error_message="Data corruption detected"
            ),
            FaultConfig(
                FaultType.SLOW_RESPONSE,
                probability=0.1,
                delay_range=(3.0, 10.0),
                error_message="Slow response simulated"
            )
        ]
        
        for fault_config in default_faults:
            self.fault_configs[fault_config.fault_type.value] = fault_config
    
    def configure_fault(self, fault_type: FaultType, **kwargs):
        """配置特定故障类型
        
        Args:
            fault_type: 故障类型
            **kwargs: 故障配置参数
        """
        if fault_type.value in self.fault_configs:
            config = self.fault_configs[fault_type.value]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            self.fault_configs[fault_type.value] = FaultConfig(fault_type, **kwargs)
    
    def should_inject_fault(self, operation: str = "default") -> bool:
        """判断是否应该注入故障
        
        Args:
            operation: 操作名称
            
        Returns:
            是否注入故障
        """
        if not self.enabled:
            return False
        
        return random.random() < self.global_failure_rate
    
    def inject_fault(self, fault_type: Optional[FaultType] = None, operation: str = "default"):
        """注入故障
        
        Args:
            fault_type: 指定故障类型，如果为None则随机选择
            operation: 操作名称
            
        Raises:
            FaultException: 故障注入异常
        """
        if not self.enabled:
            return
        
        # 选择故障类型
        if fault_type is None:
            # 根据概率随机选择故障类型
            available_faults = [
                config for config in self.fault_configs.values() 
                if config.enabled and random.random() < config.probability
            ]
            if not available_faults:
                return
            fault_config = random.choice(available_faults)
        else:
            fault_config = self.fault_configs.get(fault_type.value)
            if not fault_config or not fault_config.enabled:
                return
        
        # 记录故障统计
        self._record_fault(fault_config.fault_type, operation)
        
        # 执行故障注入
        self._execute_fault(fault_config)
    
    def _execute_fault(self, fault_config: FaultConfig):
        """执行故障注入
        
        Args:
            fault_config: 故障配置
        """
        logger.warning(f"Injecting fault: {fault_config.fault_type.value}")
        
        if fault_config.fault_type == FaultType.SLOW_RESPONSE:
            # 模拟慢响应
            delay = random.uniform(*fault_config.delay_range)
            logger.info(f"Simulating slow response: {delay:.2f}s delay")
            time.sleep(delay)
        elif fault_config.fault_type == FaultType.NETWORK_TIMEOUT:
            # 模拟网络超时
            delay = random.uniform(*fault_config.delay_range)
            time.sleep(delay)
            raise FaultException(fault_config.fault_type, fault_config.error_message)
        else:
            # 其他故障类型直接抛出异常
            raise FaultException(fault_config.fault_type, fault_config.error_message)
    
    def _record_fault(self, fault_type: FaultType, operation: str):
        """记录故障统计
        
        Args:
            fault_type: 故障类型
            operation: 操作名称
        """
        key = f"{operation}:{fault_type.value}"
        if key not in self.fault_statistics:
            self.fault_statistics[key] = {
                'count': 0,
                'last_occurrence': None,
                'fault_type': fault_type.value,
                'operation': operation
            }
        
        self.fault_statistics[key]['count'] += 1
        self.fault_statistics[key]['last_occurrence'] = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取故障统计信息
        
        Returns:
            故障统计数据
        """
        return {
            'enabled': self.enabled,
            'global_failure_rate': self.global_failure_rate,
            'fault_configs': {
                name: {
                    'fault_type': config.fault_type.value,
                    'probability': config.probability,
                    'enabled': config.enabled
                }
                for name, config in self.fault_configs.items()
            },
            'fault_statistics': self.fault_statistics
        }
    
    def reset_statistics(self):
        """重置故障统计"""
        self.fault_statistics.clear()
    
    def enable(self):
        """启用故障注入"""
        self.enabled = True
        logger.info("Fault injection enabled")
    
    def disable(self):
        """禁用故障注入"""
        self.enabled = False
        logger.info("Fault injection disabled")
    
    def set_global_failure_rate(self, rate: float):
        """设置全局故障率
        
        Args:
            rate: 故障率 (0.0 - 1.0)
        """
        if 0.0 <= rate <= 1.0:
            self.global_failure_rate = rate
            logger.info(f"Global failure rate set to {rate}")
        else:
            raise ValueError("Failure rate must be between 0.0 and 1.0")


def fault_injection_decorator(injector: FaultInjector, operation: str = "default"):
    """故障注入装饰器
    
    Args:
        injector: 故障注入器实例
        operation: 操作名称
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # 检查是否应该注入故障
            if injector.should_inject_fault(operation):
                injector.inject_fault(operation=operation)
            
            # 执行原函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator