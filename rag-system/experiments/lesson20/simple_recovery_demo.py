#!/usr/bin/env python3
"""
实验二：恢复机制演示（简化版）

本实验演示重试机制、熔断器和降级服务的使用。
"""

import time
import random
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

# 重试策略
class RetryStrategy(Enum):
    FIXED_DELAY = "fixed"
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    RANDOM_JITTER = "jitter"

# 熔断器状态
class CircuitBreakerState(Enum):
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 熔断状态
    HALF_OPEN = "half_open" # 半开状态

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True

class SimpleRetryMechanism:
    """简化的重试机制"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.statistics = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'operations': {}
        }
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        else:  # RANDOM_JITTER
            delay = self.config.base_delay * random.uniform(0.5, 1.5)
        
        # 添加抖动
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOM_JITTER:
            jitter = delay * 0.1 * random.uniform(-1, 1)
            delay += jitter
        
        return min(delay, self.config.max_delay)
    
    def retry_with_backoff(self, operation_name: str, func: Callable, *args, **kwargs):
        """带退避的重试"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.statistics['total_attempts'] += 1
            
            try:
                print(f"🔄 尝试 {attempt}/{self.config.max_attempts}: {operation_name}")
                result = func(*args, **kwargs)
                
                if attempt > 1:
                    self.statistics['successful_retries'] += 1
                    print(f"✅ 重试成功: {operation_name}")
                
                return result
                
            except Exception as e:
                last_exception = e
                print(f"❌ 尝试 {attempt} 失败: {str(e)}")
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    print(f"⏳ 等待 {delay:.2f} 秒后重试...")
                    time.sleep(delay)
                else:
                    self.statistics['failed_retries'] += 1
        
        print(f"💥 所有重试失败: {operation_name}")
        raise last_exception

class SimpleCircuitBreaker:
    """简化的熔断器"""
    
    def __init__(self, failure_threshold: int = 3, timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.statistics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'state_changes': []
        }
    
    def _change_state(self, new_state: CircuitBreakerState):
        """改变状态"""
        if self.state != new_state:
            print(f"🔌 熔断器状态变更: {self.state.value} -> {new_state.value}")
            self.statistics['state_changes'].append({
                'from': self.state.value,
                'to': new_state.value,
                'timestamp': time.time()
            })
            self.state = new_state
            
            if new_state == CircuitBreakerState.OPEN:
                self.statistics['circuit_opens'] += 1
    
    def call(self, operation_name: str, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        self.statistics['total_calls'] += 1
        
        # 检查熔断器状态
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self._change_state(CircuitBreakerState.HALF_OPEN)
            else:
                print(f"⚡ 熔断器开启，拒绝调用: {operation_name}")
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            # 成功调用
            self.statistics['successful_calls'] += 1
            if self.state == CircuitBreakerState.HALF_OPEN:
                print(f"🔄 熔断器恢复正常")
                self._change_state(CircuitBreakerState.CLOSED)
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            # 失败调用
            self.statistics['failed_calls'] += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
            
            raise e

class SimpleFallbackService:
    """简化的降级服务"""
    
    def __init__(self):
        self.cache = {}
        self.statistics = {
            'total_calls': 0,
            'fallback_calls': 0,
            'cache_hits': 0,
            'default_responses': 0
        }
    
    def call_with_fallback(self, operation_name: str, primary_func: Callable, 
                          fallback_func: Optional[Callable] = None, 
                          default_value: Any = None, *args, **kwargs):
        """带降级的调用"""
        self.statistics['total_calls'] += 1
        
        try:
            # 尝试主服务
            result = primary_func(*args, **kwargs)
            # 缓存成功结果
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"🔄 主服务失败，启用降级: {operation_name} - {str(e)}")
            self.statistics['fallback_calls'] += 1
            
            # 尝试缓存
            cache_key = f"{operation_name}_{hash(str(args) + str(kwargs))}"
            if cache_key in self.cache:
                print(f"📦 使用缓存结果")
                self.statistics['cache_hits'] += 1
                return self.cache[cache_key]
            
            # 尝试降级服务
            if fallback_func:
                try:
                    print(f"🔧 使用降级服务")
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    print(f"❌ 降级服务也失败: {str(fallback_error)}")
            
            # 使用默认值
            if default_value is not None:
                print(f"🎯 使用默认值")
                self.statistics['default_responses'] += 1
                return default_value
            
            # 所有降级方案都失败
            raise e

# 模拟不稳定的服务
class UnstableService:
    """不稳定的服务（用于测试）"""
    
    def __init__(self, failure_rate: float = 0.7):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def unreliable_operation(self, data: str):
        """不可靠的操作"""
        self.call_count += 1
        
        if random.random() < self.failure_rate:
            raise Exception(f"Service failure #{self.call_count}")
        
        return f"Success: {data} (call #{self.call_count})"
    
    def fallback_operation(self, data: str):
        """降级操作"""
        return f"Fallback: {data} (simplified response)"

def run_recovery_demo():
    """运行恢复机制演示"""
    print("=" * 60)
    print("🛠️ 恢复机制演示实验")
    print("=" * 60)
    
    # 创建组件
    retry_config = RetryConfig(max_attempts=3, base_delay=0.5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    retry_mechanism = SimpleRetryMechanism(retry_config)
    circuit_breaker = SimpleCircuitBreaker(failure_threshold=2, timeout=5.0)
    fallback_service = SimpleFallbackService()
    unstable_service = UnstableService(failure_rate=0.6)
    
    print("\n🧪 测试1: 重试机制")
    print("-" * 30)
    
    try:
        result = retry_mechanism.retry_with_backoff(
            "unstable_operation",
            unstable_service.unreliable_operation,
            "test data 1"
        )
        print(f"✅ 最终结果: {result}")
    except Exception as e:
        print(f"❌ 重试失败: {str(e)}")
    
    print("\n🧪 测试2: 熔断器")
    print("-" * 30)
    
    # 多次调用触发熔断器
    for i in range(6):
        try:
            result = circuit_breaker.call(
                f"circuit_test_{i+1}",
                unstable_service.unreliable_operation,
                f"test data {i+1}"
            )
            print(f"✅ 调用成功: {result}")
        except Exception as e:
            print(f"❌ 调用失败: {str(e)}")
        
        time.sleep(0.5)
    
    print("\n🧪 测试3: 降级服务")
    print("-" * 30)
    
    # 测试降级服务
    test_cases = [
        "first call",
        "second call",
        "third call (should use cache)",
        "fourth call"
    ]
    
    for i, test_data in enumerate(test_cases, 1):
        try:
            result = fallback_service.call_with_fallback(
                "fallback_test",
                unstable_service.unreliable_operation,
                unstable_service.fallback_operation,
                "Default response",
                test_data
            )
            print(f"✅ 结果 {i}: {result}")
        except Exception as e:
            print(f"❌ 完全失败 {i}: {str(e)}")
        
        time.sleep(0.3)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("📊 恢复机制统计")
    print("=" * 60)
    
    print("\n🔄 重试统计:")
    retry_stats = retry_mechanism.statistics
    print(f"  总尝试次数: {retry_stats['total_attempts']}")
    print(f"  成功重试: {retry_stats['successful_retries']}")
    print(f"  失败重试: {retry_stats['failed_retries']}")
    
    print("\n⚡ 熔断器统计:")
    cb_stats = circuit_breaker.statistics
    print(f"  总调用次数: {cb_stats['total_calls']}")
    print(f"  成功调用: {cb_stats['successful_calls']}")
    print(f"  失败调用: {cb_stats['failed_calls']}")
    print(f"  熔断次数: {cb_stats['circuit_opens']}")
    print(f"  当前状态: {circuit_breaker.state.value}")
    
    print("\n🔧 降级服务统计:")
    fb_stats = fallback_service.statistics
    print(f"  总调用次数: {fb_stats['total_calls']}")
    print(f"  降级调用: {fb_stats['fallback_calls']}")
    print(f"  缓存命中: {fb_stats['cache_hits']}")
    print(f"  默认响应: {fb_stats['default_responses']}")
    
    print("\n✅ 恢复机制演示完成！")

if __name__ == "__main__":
    run_recovery_demo()