#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统容错集成演示
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.fault_injection import fault_injection_decorator, FaultType
from src.recovery import (
    retry_with_backoff, RetryStrategy,
    CircuitBreaker, FallbackService, FallbackStrategy
)
from src.monitoring import global_metrics_collector, global_alert_manager
import time
import random

class RobustRAGService:
    """容错的RAG服务"""
    
    def __init__(self):
        self.vector_circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.llm_circuit_breaker = CircuitBreaker(failure_threshold=2)
        self.fallback_service = FallbackService()
        self.fallback_service.configure(FallbackStrategy.CACHE, cache_ttl=300)
        
        # 初始化指标收集
        self.metrics = global_metrics_collector
        
        print("🤖 RobustRAGService 初始化完成")
    
    @fault_injection_decorator(FaultType.NETWORK_TIMEOUT, probability=0.1)
    @retry_with_backoff(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    def query_vector_store(self, query):
        """查询向量存储"""
        start_time = time.time()
        
        try:
            # 模拟向量查询
            if random.random() < 0.2:  # 20%失败率
                raise Exception("向量存储查询失败")
            
            # 模拟查询延迟
            time.sleep(random.uniform(0.1, 0.5))
            
            docs = [
                f"文档1: 关于{query}的相关内容...",
                f"文档2: {query}的详细说明...",
                f"文档3: {query}的应用案例..."
            ]
            
            # 记录成功指标
            duration = time.time() - start_time
            self.metrics.record_timer("vector_query_duration", duration)
            self.metrics.record_counter("vector_queries_total", 1, {"status": "success"})
            
            return docs
            
        except Exception as e:
            # 记录失败指标
            duration = time.time() - start_time
            self.metrics.record_timer("vector_query_duration", duration)
            self.metrics.record_counter("vector_queries_total", 1, {"status": "error"})
            raise e
    
    @fault_injection_decorator(FaultType.SLOW_RESPONSE, probability=0.15)
    def generate_response(self, context, query):
        """生成回答"""
        start_time = time.time()
        
        try:
            # 使用熔断器保护LLM调用
            def llm_call():
                if random.random() < 0.25:  # 25%失败率
                    raise Exception("LLM服务不可用")
                
                # 模拟LLM处理时间
                time.sleep(random.uniform(0.2, 1.0))
                
                return f"基于提供的上下文，关于'{query}'的回答是：这是一个详细的解答，结合了相关文档的信息。"
            
            response = self.llm_circuit_breaker.call(llm_call)
            
            # 记录成功指标
            duration = time.time() - start_time
            self.metrics.record_timer("llm_response_duration", duration)
            self.metrics.record_counter("llm_calls_total", 1, {"status": "success"})
            
            return response
            
        except Exception as e:
            # 记录失败指标
            duration = time.time() - start_time
            self.metrics.record_timer("llm_response_duration", duration)
            self.metrics.record_counter("llm_calls_total", 1, {"status": "error"})
            raise e
    
    def search_with_fallback(self, query):
        """带降级的搜索"""
        def primary_search():
            # 查询向量存储
            docs = self.query_vector_store(query)
            
            # 生成回答
            response = self.generate_response(docs, query)
            
            return {
                "success": True,
                "response": response,
                "source": "primary",
                "documents": len(docs)
            }
        
        # 使用降级服务
        try:
            result = self.fallback_service.execute_with_fallback(
                f"search_{hash(query) % 1000}",  # 简单的缓存键
                primary_search,
                default_value={
                    "success": True,
                    "response": f"抱歉，由于系统繁忙，无法提供关于'{query}'的详细回答。请稍后重试。",
                    "source": "fallback",
                    "documents": 0
                }
            )
            
            # 记录搜索指标
            self.metrics.record_counter("searches_total", 1, {"source": result["source"]})
            
            return result
            
        except Exception as e:
            # 最终降级
            self.metrics.record_counter("searches_total", 1, {"source": "error"})
            return {
                "success": False,
                "error": str(e),
                "response": "系统暂时不可用，请稍后重试。",
                "source": "error"
            }

def demo_rag_integration():
    """演示RAG系统集成"""
    print("=== RAG系统容错集成演示 ===")
    
    # 创建容错RAG服务
    rag_service = RobustRAGService()
    
    # 测试查询
    queries = [
        "什么是机器学习？",
        "深度学习的应用场景",
        "自然语言处理技术",
        "推荐系统算法",
        "计算机视觉发展",
        "什么是机器学习？",  # 重复查询测试缓存
        "人工智能伦理",
        "大数据分析方法"
    ]
    
    print(f"🔍 开始执行 {len(queries)} 个查询...\n")
    
    for i, query in enumerate(queries):
        print(f"📝 查询 {i+1}: {query}")
        
        start_time = time.time()
        result = rag_service.search_with_fallback(query)
        duration = time.time() - start_time
        
        if result["success"]:
            source_icon = {
                "primary": "🎯",
                "fallback": "🔄",
                "cache": "💾"
            }.get(result["source"], "❓")
            
            print(f"   {source_icon} 成功 ({result['source']}) - 耗时: {duration:.2f}s")
            print(f"   📄 文档数: {result.get('documents', 0)}")
            print(f"   💬 回答: {result['response'][:100]}...")
        else:
            print(f"   ❌ 失败 - {result['error']} (耗时: {duration:.2f}s)")
            print(f"   💬 降级回答: {result['response']}")
        
        print()
        time.sleep(0.5)
    
    # 显示服务统计
    print("📊 服务统计:")
    
    # 熔断器状态
    print(f"   🔌 向量存储熔断器: {rag_service.vector_circuit_breaker.state.value}")
    print(f"   🤖 LLM熔断器: {rag_service.llm_circuit_breaker.state.value}")
    
    # 降级服务统计
    fallback_stats = rag_service.fallback_service.get_statistics()
    print(f"   🔄 降级服务调用: {fallback_stats['total_calls']}")
    print(f"   💾 缓存命中: {fallback_stats['cache_hits']}")
    
    # 指标统计
    metrics = rag_service.metrics.get_metrics_summary()
    print(f"\n📈 性能指标:")
    for name, data in metrics.items():
        if 'count' in data:
            print(f"   {name}: {data['count']} 次")
        elif 'avg' in data:
            print(f"   {name}: 平均 {data['avg']:.2f}s")

if __name__ == "__main__":
    demo_rag_integration()
    
    print("\n🎯 实验四完成！RAG系统容错机制集成演示结束。")