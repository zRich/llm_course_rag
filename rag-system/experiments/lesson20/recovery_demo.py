#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恢复机制演示
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.recovery import (
    retry_with_backoff, RetryStrategy, 
    CircuitBreaker, CircuitBreakerState,
    FallbackService, FallbackStrategy
)
import time
import random

def demo_retry_mechanism():
    """演示重试机制"""
    print("=== 重试机制演示 ===")
    
    # 模拟不稳定的服务
    call_count = 0
    
    @retry_with_backoff(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay=0.5
    )
    def unreliable_service(service_name):
        nonlocal call_count
        call_count += 1
        
        print(f"  🔄 尝试调用 {service_name} (第 {call_count} 次)")
        
        # 70%的失败率
        if random.random() < 0.7:
            raise Exception(f"{service_name} 服务临时不可用")
        
        return f"{service_name} 调用成功"
    
    # 测试重试机制
    services = ["用户服务", "订单服务", "支付服务"]
    
    for service in services:
        call_count = 0
        try:
            start_time = time.time()
            result = unreliable_service(service)
            duration = time.time() - start_time
            print(f"✅ {service}: {result} (耗时: {duration:.2f}s, 尝试次数: {call_count})")
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {service}: 最终失败 - {str(e)} (耗时: {duration:.2f}s, 尝试次数: {call_count})")
        print()

def demo_circuit_breaker():
    """演示熔断器模式"""
    print("\n=== 熔断器模式演示 ===")
    
    # 创建熔断器
    breaker = CircuitBreaker(
        failure_threshold=3,
        timeout_seconds=5,
        half_open_max_calls=2
    )
    
    def failing_service(request_id):
        """经常失败的服务"""
        # 80%的失败率
        if random.random() < 0.8:
            raise Exception(f"服务错误 (请求 {request_id})")
        return f"请求 {request_id} 处理成功"
    
    print("开始测试熔断器...")
    
    # 测试熔断器
    for i in range(15):
        try:
            result = breaker.call(lambda: failing_service(i+1))
            print(f"✅ 调用 {i+1}: {result} (状态: {breaker.state.value})")
        except Exception as e:
            print(f"❌ 调用 {i+1}: {str(e)} (状态: {breaker.state.value})")
        
        # 显示熔断器统计
        stats = breaker.get_statistics()
        print(f"   📊 成功率: {stats['success_rate']:.1f}%, 失败率: {stats['failure_rate']:.1f}%")
        
        time.sleep(0.5)
        
        # 在熔断器打开时，等待一段时间让它进入半开状态
        if breaker.state == CircuitBreakerState.OPEN and i == 8:
            print("\n⏳ 等待熔断器超时，进入半开状态...")
            time.sleep(6)

def demo_fallback_service():
    """演示降级服务"""
    print("\n=== 降级服务演示 ===")
    
    # 创建降级服务
    fallback = FallbackService()
    fallback.configure(FallbackStrategy.CACHE, cache_ttl=10)
    
    def get_user_profile(user_id):
        """获取用户资料"""
        # 50%的失败率
        if random.random() < 0.5:
            raise Exception(f"用户服务不可用 (用户 {user_id})")
        
        return {
            "id": user_id,
            "name": f"User{user_id}",
            "email": f"user{user_id}@example.com",
            "created_at": time.time()
        }
    
    # 测试降级服务
    user_ids = [1, 2, 1, 3, 2, 4, 1]  # 重复访问用户1和2测试缓存
    
    for user_id in user_ids:
        try:
            result = fallback.execute_with_fallback(
                f"user_profile_{user_id}",
                lambda: get_user_profile(user_id),
                default_value={
                    "id": user_id,
                    "name": f"默认用户{user_id}",
                    "email": "default@example.com",
                    "created_at": time.time()
                }
            )
            
            # 检查是否来自缓存
            cache_hit = "created_at" in result and time.time() - result["created_at"] > 1
            cache_status = "(缓存命中)" if cache_hit else "(实时数据)"
            
            print(f"👤 用户 {user_id}: {result['name']} {cache_status}")
            
        except Exception as e:
            print(f"❌ 用户 {user_id}: 获取失败 - {str(e)}")
        
        time.sleep(0.5)
    
    # 显示降级服务统计
    stats = fallback.get_statistics()
    print(f"\n📊 降级服务统计:")
    print(f"   总调用: {stats['total_calls']}")
    print(f"   成功: {stats['successful_calls']}")
    print(f"   降级: {stats['fallback_calls']}")
    print(f"   缓存命中: {stats['cache_hits']}")

if __name__ == "__main__":
    demo_retry_mechanism()
    demo_circuit_breaker()
    demo_fallback_service()
    
    print("\n🎯 实验二完成！请观察不同恢复机制的效果。")