#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
故障注入演示
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.fault_injection import FaultInjector, FaultType, fault_injection_decorator
import time
import random

def demo_basic_fault_injection():
    """演示基础故障注入"""
    print("=== 基础故障注入演示 ===")
    
    # 创建故障注入器
    injector = FaultInjector()
    
    # 配置网络超时故障
    injector.configure_fault(FaultType.NETWORK_TIMEOUT, probability=0.3)
    
    def simulate_api_call():
        """模拟API调用"""
        time.sleep(0.1)  # 正常响应时间
        return {"status": "success", "data": f"response_{random.randint(1, 100)}"}
    
    # 测试故障注入
    success_count = 0
    failure_count = 0
    
    for i in range(10):
        try:
            result = injector.inject_fault("api_call", simulate_api_call)
            print(f"✅ 调用 {i+1}: 成功 - {result}")
            success_count += 1
        except Exception as e:
            print(f"❌ 调用 {i+1}: 失败 - {str(e)}")
            failure_count += 1
    
    print(f"\n📊 统计结果: 成功 {success_count} 次, 失败 {failure_count} 次")
    print(f"📈 故障率: {failure_count/10*100:.1f}%")

def demo_decorator_fault_injection():
    """演示装饰器故障注入"""
    print("\n=== 装饰器故障注入演示 ===")
    
    @fault_injection_decorator(FaultType.SLOW_RESPONSE, probability=0.4)
    def process_request(data):
        """处理请求"""
        return f"处理完成: {data}"
    
    # 测试装饰器故障注入
    for i in range(5):
        start_time = time.time()
        try:
            result = process_request(f"request_{i}")
            duration = time.time() - start_time
            print(f"🔄 请求 {i+1}: {result}, 耗时: {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            print(f"⏱️ 请求 {i+1}: 失败 - {str(e)}, 耗时: {duration:.2f}s")

def demo_multiple_fault_types():
    """演示多种故障类型"""
    print("\n=== 多种故障类型演示 ===")
    
    injector = FaultInjector()
    
    # 配置多种故障
    injector.configure_fault(FaultType.SERVICE_UNAVAILABLE, probability=0.2)
    injector.configure_fault(FaultType.DATA_CORRUPTION, probability=0.1)
    injector.configure_fault(FaultType.MEMORY_ERROR, probability=0.1)
    
    def database_operation(operation_type):
        """模拟数据库操作"""
        operations = {
            "read": lambda: {"data": ["item1", "item2", "item3"]},
            "write": lambda: {"status": "written", "id": random.randint(1, 1000)},
            "delete": lambda: {"status": "deleted", "count": 1}
        }
        return operations[operation_type]()
    
    operations = ["read", "write", "delete"] * 3
    
    for i, op in enumerate(operations):
        try:
            result = injector.inject_fault(f"db_{op}", lambda: database_operation(op))
            print(f"💾 操作 {i+1} ({op}): 成功 - {result}")
        except Exception as e:
            print(f"💥 操作 {i+1} ({op}): 失败 - {str(e)}")

if __name__ == "__main__":
    demo_basic_fault_injection()
    demo_decorator_fault_injection()
    demo_multiple_fault_types()
    
    print("\n🎯 实验一完成！请观察不同故障类型的表现。")