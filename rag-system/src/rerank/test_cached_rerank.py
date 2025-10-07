#!/usr/bin/env python3
"""
缓存重排序功能测试脚本

测试CachedRerankService的缓存功能：
1. 缓存命中和未命中
2. 缓存性能提升
3. 缓存统计信息
4. 缓存管理功能
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.cached_rerank import CachedRerankService

def test_cached_rerank_initialization():
    """测试缓存重排序服务初始化"""
    print("\n=== 测试缓存重排序服务初始化 ===")
    
    try:
        # 使用较小的缓存大小进行测试
        cached_rerank = CachedRerankService(max_cache_size=100)
        print("✓ 缓存重排序服务初始化成功")
        
        # 检查缓存统计信息
        cache_stats = cached_rerank.get_cache_stats()
        print(f"✓ 初始缓存统计: {cache_stats}")
        
        return cached_rerank
    except Exception as e:
        print(f"✗ 缓存重排序服务初始化失败: {e}")
        return None

def test_cache_miss_and_hit(cached_rerank: CachedRerankService):
    """测试缓存未命中和命中"""
    print("\n=== 测试缓存未命中和命中 ===")
    
    query = "什么是机器学习？"
    documents = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。",
        "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。",
        "今天天气很好，适合出去散步。",
        "机器学习算法可以从数据中学习模式，并做出预测或决策。",
        "Python是一种流行的编程语言，广泛用于数据科学和机器学习。"
    ]
    
    # 第一次查询 - 缓存未命中
    print("\n--- 第一次查询（缓存未命中）---")
    start_time = time.time()
    result1 = cached_rerank.rerank(query, documents, top_k=3)
    first_time = time.time() - start_time
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 第一次查询完成，耗时: {first_time:.3f}秒")
    print(f"✓ 缓存统计: 命中率={cache_stats['hit_rate']:.2%}, 总查询={cache_stats['total_queries']}")
    
    # 第二次相同查询 - 缓存命中
    print("\n--- 第二次相同查询（缓存命中）---")
    start_time = time.time()
    result2 = cached_rerank.rerank(query, documents, top_k=3)
    second_time = time.time() - start_time
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 第二次查询完成，耗时: {second_time:.3f}秒")
    print(f"✓ 缓存统计: 命中率={cache_stats['hit_rate']:.2%}, 总查询={cache_stats['total_queries']}")
    
    # 性能提升计算
    if second_time > 0:
        speedup = first_time / second_time
        print(f"✓ 缓存带来的性能提升: {speedup:.2f}x")
    
    # 验证结果一致性
    if result1.documents == result2.documents and result1.scores == result2.scores:
        print("✓ 缓存结果与原始结果一致")
    else:
        print("✗ 缓存结果与原始结果不一致")
    
    return result1, result2

def test_cache_performance(cached_rerank: CachedRerankService):
    """测试缓存性能"""
    print("\n=== 测试缓存性能 ===")
    
    queries = [
        "什么是深度学习？",
        "Python编程语言的特点",
        "人工智能的应用领域",
        "机器学习算法分类"
    ]
    
    documents = [
        "深度学习是机器学习的一个子集，使用多层神经网络。",
        "Python是一种高级编程语言，语法简洁易读。",
        "人工智能在医疗、金融、交通等领域有广泛应用。",
        "机器学习算法包括监督学习、无监督学习和强化学习。",
        "神经网络是深度学习的基础架构。",
        "数据预处理是机器学习项目的重要步骤。"
    ]
    
    # 第一轮：填充缓存
    print("\n--- 第一轮：填充缓存 ---")
    first_round_times = []
    for i, query in enumerate(queries):
        start_time = time.time()
        result = cached_rerank.rerank(query, documents, top_k=3)
        elapsed = time.time() - start_time
        first_round_times.append(elapsed)
        print(f"  查询 {i+1}: {elapsed:.3f}秒")
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 第一轮完成，缓存统计: {cache_stats}")
    
    # 第二轮：利用缓存
    print("\n--- 第二轮：利用缓存 ---")
    second_round_times = []
    for i, query in enumerate(queries):
        start_time = time.time()
        result = cached_rerank.rerank(query, documents, top_k=3)
        elapsed = time.time() - start_time
        second_round_times.append(elapsed)
        print(f"  查询 {i+1}: {elapsed:.3f}秒")
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 第二轮完成，缓存统计: {cache_stats}")
    
    # 性能对比
    avg_first = sum(first_round_times) / len(first_round_times)
    avg_second = sum(second_round_times) / len(second_round_times)
    if avg_second > 0:
        overall_speedup = avg_first / avg_second
        print(f"✓ 整体性能提升: {overall_speedup:.2f}x")
        print(f"✓ 平均查询时间: 第一轮={avg_first:.3f}秒, 第二轮={avg_second:.3f}秒")

def test_cache_management(cached_rerank: CachedRerankService):
    """测试缓存管理功能"""
    print("\n=== 测试缓存管理功能 ===")
    
    # 获取缓存信息
    cache_info = cached_rerank.get_cache_info()
    print(f"✓ 缓存信息: {cache_info}")
    
    # 估算内存使用
    memory_usage = cached_rerank.estimate_memory_usage()
    print(f"✓ 估算内存使用: {memory_usage:.2f} MB")
    
    # 缓存预热
    print("\n--- 测试缓存预热 ---")
    warmup_queries = [
        "机器学习基础概念",
        "深度学习网络结构"
    ]
    warmup_documents = [
        "机器学习是一种让计算机从数据中学习的方法。",
        "深度学习使用多层神经网络进行特征学习。",
        "数据是机器学习的燃料。"
    ]
    
    try:
        cached_rerank.warmup_cache(warmup_queries, [warmup_documents] * len(warmup_queries))
        print("✓ 缓存预热完成")
        
        cache_stats = cached_rerank.get_cache_stats()
        print(f"✓ 预热后缓存统计: {cache_stats}")
    except Exception as e:
        print(f"✗ 缓存预热失败: {e}")
    
    # 缓存导出和导入
    print("\n--- 测试缓存导出和导入 ---")
    try:
        # 导出缓存
        cache_data = cached_rerank.export_cache()
        print(f"✓ 缓存导出成功，数据大小: {len(str(cache_data))} 字符")
        
        # 清空缓存
        cached_rerank.clear_cache()
        stats_after_clear = cached_rerank.get_cache_stats()
        print(f"✓ 缓存清空完成: {stats_after_clear}")
        
        # 导入缓存
        cached_rerank.import_cache(cache_data)
        stats_after_import = cached_rerank.get_cache_stats()
        print(f"✓ 缓存导入完成: {stats_after_import}")
        
    except Exception as e:
        print(f"✗ 缓存导出/导入失败: {e}")

def test_cache_limits(cached_rerank: CachedRerankService):
    """测试缓存限制"""
    print("\n=== 测试缓存限制 ===")
    
    # 创建一个小缓存的服务进行测试
    small_cache_service = CachedRerankService(max_cache_size=3)
    
    queries = [f"查询{i}" for i in range(5)]
    documents = [f"文档{i}内容" for i in range(3)]
    
    # 填充超过缓存大小的查询
    for i, query in enumerate(queries):
        result = small_cache_service.rerank(query, documents, top_k=2)
        cache_stats = small_cache_service.get_cache_stats()
        print(f"  查询 {i+1}: 缓存大小={cache_stats['cache_size']}, 命中率={cache_stats['hit_rate']:.2%}")
    
    final_stats = small_cache_service.get_cache_stats()
    print(f"✓ 最终缓存统计: {final_stats}")
    
    # 验证缓存大小限制
    if final_stats['cache_size'] <= 3:
        print("✓ 缓存大小限制正常工作")
    else:
        print("✗ 缓存大小限制未生效")

def test_different_parameters(cached_rerank: CachedRerankService):
    """测试不同参数的缓存行为"""
    print("\n=== 测试不同参数的缓存行为 ===")
    
    query = "测试查询"
    documents = ["文档1", "文档2", "文档3", "文档4"]
    
    # 相同查询，不同top_k
    result1 = cached_rerank.rerank(query, documents, top_k=2)
    result2 = cached_rerank.rerank(query, documents, top_k=3)
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 不同top_k参数测试完成，缓存统计: {cache_stats}")
    
    # 相同查询和top_k，不同文档顺序
    shuffled_documents = documents[::-1]  # 反转文档顺序
    result3 = cached_rerank.rerank(query, shuffled_documents, top_k=2)
    
    cache_stats = cached_rerank.get_cache_stats()
    print(f"✓ 不同文档顺序测试完成，缓存统计: {cache_stats}")

def main():
    """主测试函数"""
    print("开始缓存重排序功能测试...")
    
    # 初始化服务
    cached_rerank = test_cached_rerank_initialization()
    if not cached_rerank:
        print("\n测试终止：缓存重排序服务初始化失败")
        return
    
    # 执行各项测试
    test_cache_miss_and_hit(cached_rerank)
    test_cache_performance(cached_rerank)
    test_cache_management(cached_rerank)
    test_cache_limits(cached_rerank)
    test_different_parameters(cached_rerank)
    
    # 最终统计信息
    print("\n=== 最终统计信息 ===")
    final_stats = cached_rerank.get_cache_stats()
    final_rerank_stats = cached_rerank.get_stats()
    print(f"✓ 最终缓存统计: {final_stats}")
    print(f"✓ 最终重排序统计: {final_rerank_stats}")
    
    print("\n=== 缓存重排序功能测试完成 ===")

if __name__ == "__main__":
    main()