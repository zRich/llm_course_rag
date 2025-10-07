#!/usr/bin/env python3
"""
实验一：故障注入演示（简化版）

本实验演示如何使用故障注入框架模拟各种故障场景。
"""

import sys
import os
import time
import random
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

# 简化的故障类型定义
class FaultType(Enum):
    NETWORK_TIMEOUT = "network_timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_ERROR = "memory_error"
    DISK_FULL = "disk_full"
    SLOW_RESPONSE = "slow_response"

@dataclass
class FaultConfig:
    """故障配置"""
    probability: float = 0.1
    delay_range: tuple = (0.1, 2.0)
    error_message: str = "Simulated fault"

class SimpleFaultInjector:
    """简化的故障注入器"""
    
    def __init__(self):
        self.enabled = True
        self.global_fault_rate = 0.1
        self.fault_configs = {
            FaultType.NETWORK_TIMEOUT: FaultConfig(0.05, (1.0, 3.0), "Network timeout"),
            FaultType.SERVICE_UNAVAILABLE: FaultConfig(0.03, (0.1, 0.5), "Service unavailable"),
            FaultType.SLOW_RESPONSE: FaultConfig(0.1, (0.5, 2.0), "Slow response")
        }
        self.statistics = {
            'total_calls': 0,
            'faults_injected': 0,
            'fault_types': {ft: 0 for ft in FaultType}
        }
    
    def should_inject_fault(self) -> bool:
        """判断是否应该注入故障"""
        if not self.enabled:
            return False
        return random.random() < self.global_fault_rate
    
    def inject_fault(self, operation: str, fault_type: Optional[FaultType] = None):
        """注入故障"""
        self.statistics['total_calls'] += 1
        
        if not self.should_inject_fault():
            return
        
        # 选择故障类型
        if fault_type is None:
            fault_type = random.choice(list(self.fault_configs.keys()))
        
        config = self.fault_configs.get(fault_type)
        if not config:
            return
        
        # 记录统计
        self.statistics['faults_injected'] += 1
        self.statistics['fault_types'][fault_type] += 1
        
        print(f"🔥 故障注入: {operation} - {fault_type.value}")
        
        # 执行故障
        if fault_type == FaultType.SLOW_RESPONSE:
            delay = random.uniform(*config.delay_range)
            print(f"   延迟 {delay:.2f} 秒...")
            time.sleep(delay)
        elif fault_type == FaultType.NETWORK_TIMEOUT:
            delay = random.uniform(*config.delay_range)
            print(f"   网络超时，延迟 {delay:.2f} 秒后抛出异常")
            time.sleep(delay)
            raise TimeoutError(config.error_message)
        elif fault_type == FaultType.SERVICE_UNAVAILABLE:
            raise ConnectionError(config.error_message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.statistics.copy()

# 全局故障注入器实例
fault_injector = SimpleFaultInjector()

def fault_injection_decorator(operation_name: str, fault_type: Optional[FaultType] = None):
    """故障注入装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                fault_injector.inject_fault(operation_name, fault_type)
                return func(*args, **kwargs)
            except Exception as e:
                print(f"❌ 操作失败: {operation_name} - {str(e)}")
                raise
        return wrapper
    return decorator

# 模拟的RAG服务组件
class MockVectorStore:
    """模拟向量存储"""
    
    @fault_injection_decorator("vector_search")
    def search(self, query: str, top_k: int = 5):
        """搜索向量"""
        print(f"🔍 向量搜索: {query} (top_k={top_k})")
        time.sleep(0.1)  # 模拟搜索时间
        return [f"文档{i}" for i in range(top_k)]

class MockLLMService:
    """模拟LLM服务"""
    
    @fault_injection_decorator("llm_generate")
    def generate(self, prompt: str):
        """生成回答"""
        print(f"🤖 LLM生成: {prompt[:50]}...")
        time.sleep(0.2)  # 模拟生成时间
        return f"基于提供的文档，回答是：{prompt}的相关信息"

class MockEmbeddingService:
    """模拟嵌入服务"""
    
    @fault_injection_decorator("embedding_encode")
    def encode(self, text: str):
        """编码文本"""
        print(f"📊 文本编码: {text[:30]}...")
        time.sleep(0.05)  # 模拟编码时间
        return [random.random() for _ in range(768)]  # 模拟768维向量

def run_fault_injection_demo():
    """运行故障注入演示"""
    print("=" * 60)
    print("🧪 故障注入演示实验")
    print("=" * 60)
    
    # 创建模拟服务
    vector_store = MockVectorStore()
    llm_service = MockLLMService()
    embedding_service = MockEmbeddingService()
    
    # 设置故障注入参数
    fault_injector.global_fault_rate = 0.3  # 30%故障率
    
    print(f"\n📋 配置信息:")
    print(f"   全局故障率: {fault_injector.global_fault_rate * 100}%")
    print(f"   故障类型: {list(fault_injector.fault_configs.keys())}")
    
    # 模拟多次操作
    queries = [
        "什么是机器学习？",
        "深度学习的原理",
        "自然语言处理应用",
        "计算机视觉技术",
        "人工智能发展历史"
    ]
    
    print(f"\n🚀 开始执行 {len(queries)} 个查询...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- 查询 {i}: {query} ---")
        
        try:
            # 1. 文本编码
            embedding = embedding_service.encode(query)
            print(f"✅ 编码完成，向量维度: {len(embedding)}")
            
            # 2. 向量搜索
            documents = vector_store.search(query)
            print(f"✅ 搜索完成，找到 {len(documents)} 个文档")
            
            # 3. LLM生成
            response = llm_service.generate(query)
            print(f"✅ 生成完成: {response[:50]}...")
            
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            continue
        
        # 短暂延迟
        time.sleep(0.1)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("📊 故障注入统计")
    print("=" * 60)
    
    stats = fault_injector.get_statistics()
    print(f"总调用次数: {stats['total_calls']}")
    print(f"故障注入次数: {stats['faults_injected']}")
    print(f"故障注入率: {stats['faults_injected']/stats['total_calls']*100:.1f}%")
    
    print("\n故障类型分布:")
    for fault_type, count in stats['fault_types'].items():
        if count > 0:
            print(f"  {fault_type.value}: {count} 次")
    
    print("\n✅ 故障注入演示完成！")

if __name__ == "__main__":
    run_fault_injection_demo()