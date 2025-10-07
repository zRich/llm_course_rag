#!/usr/bin/env python3
"""
基础重排序功能测试脚本

测试RerankService的基本功能：
1. 模型加载
2. 文档重排序
3. 批量重排序
4. 统计信息
"""

import sys
import os
import time
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.rerank import RerankService, RerankResult

def test_rerank_service_initialization():
    """测试重排序服务初始化"""
    print("\n=== 测试重排序服务初始化 ===")
    
    try:
        rerank_service = RerankService()
        print("✓ 重排序服务初始化成功")
        
        # 检查统计信息
        stats = rerank_service.get_stats()
        print(f"✓ 初始统计信息: {stats}")
        
        return rerank_service
    except Exception as e:
        print(f"✗ 重排序服务初始化失败: {e}")
        return None

def test_single_rerank(rerank_service: RerankService):
    """测试单次重排序"""
    print("\n=== 测试单次重排序 ===")
    
    query = "什么是机器学习？"
    documents = [
        "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。",
        "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。",
        "今天天气很好，适合出去散步。",
        "机器学习算法可以从数据中学习模式，并做出预测或决策。",
        "Python是一种流行的编程语言，广泛用于数据科学和机器学习。"
    ]
    
    try:
        start_time = time.time()
        result = rerank_service.rerank(query, documents, top_k=3)
        end_time = time.time()
        
        print(f"✓ 重排序完成，耗时: {end_time - start_time:.3f}秒")
        print(f"✓ 返回文档数量: {len(result.documents)}")
        
        for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
            print(f"  {i+1}. 分数: {score:.4f} - {doc[:50]}...")
        
        # 检查统计信息
        stats = rerank_service.get_stats()
        print(f"✓ 更新后统计信息: {stats}")
        
        return result
    except Exception as e:
        print(f"✗ 单次重排序失败: {e}")
        return None

def test_batch_rerank(rerank_service: RerankService):
    """测试批量重排序"""
    print("\n=== 测试批量重排序 ===")
    
    queries = [
        "什么是深度学习？",
        "Python编程语言的特点"
    ]
    
    documents_list = [
        [
            "深度学习是机器学习的一个子集，使用多层神经网络。",
            "机器学习包括监督学习、无监督学习和强化学习。",
            "今天是个好天气。"
        ],
        [
            "Python是一种高级编程语言，语法简洁易读。",
            "Java是一种面向对象的编程语言。",
            "Python广泛应用于数据科学、Web开发和人工智能。"
        ]
    ]
    
    try:
        start_time = time.time()
        results = rerank_service.batch_rerank(queries, documents_list, top_k=2)
        end_time = time.time()
        
        print(f"✓ 批量重排序完成，耗时: {end_time - start_time:.3f}秒")
        print(f"✓ 处理查询数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n  查询 {i+1}: {queries[i]}")
            for j, (doc, score) in enumerate(zip(result.documents, result.scores)):
                print(f"    {j+1}. 分数: {score:.4f} - {doc[:40]}...")
        
        # 检查统计信息
        stats = rerank_service.get_stats()
        print(f"\n✓ 批量处理后统计信息: {stats}")
        
        return results
    except Exception as e:
        print(f"✗ 批量重排序失败: {e}")
        return None

def test_performance_metrics(rerank_service: RerankService):
    """测试性能指标"""
    print("\n=== 测试性能指标 ===")
    
    query = "人工智能的应用领域"
    documents = [
        "人工智能在医疗诊断中发挥重要作用。",
        "自动驾驶汽车是人工智能的重要应用。",
        "今天吃什么好呢？",
        "人工智能在金融风控中有广泛应用。",
        "机器翻译是自然语言处理的重要应用。",
        "天气预报说明天会下雨。",
        "人工智能助手可以帮助用户解答问题。"
    ]
    
    # 多次测试以获得平均性能
    times = []
    for i in range(5):
        start_time = time.time()
        result = rerank_service.rerank(query, documents, top_k=3)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"✓ 平均重排序时间: {avg_time:.3f}秒")
    print(f"✓ 最快时间: {min(times):.3f}秒")
    print(f"✓ 最慢时间: {max(times):.3f}秒")
    
    # 最终统计信息
    stats = rerank_service.get_stats()
    print(f"✓ 最终统计信息: {stats}")

def test_edge_cases(rerank_service: RerankService):
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试空文档列表
    try:
        result = rerank_service.rerank("测试查询", [], top_k=3)
        print("✓ 空文档列表处理正常")
    except Exception as e:
        print(f"✗ 空文档列表处理失败: {e}")
    
    # 测试单个文档
    try:
        result = rerank_service.rerank("测试查询", ["单个文档内容"], top_k=3)
        print(f"✓ 单个文档处理正常，返回{len(result.documents)}个文档")
    except Exception as e:
        print(f"✗ 单个文档处理失败: {e}")
    
    # 测试top_k大于文档数量
    try:
        documents = ["文档1", "文档2"]
        result = rerank_service.rerank("测试查询", documents, top_k=5)
        print(f"✓ top_k大于文档数量处理正常，返回{len(result.documents)}个文档")
    except Exception as e:
        print(f"✗ top_k大于文档数量处理失败: {e}")

def main():
    """主测试函数"""
    print("开始基础重排序功能测试...")
    
    # 初始化服务
    rerank_service = test_rerank_service_initialization()
    if not rerank_service:
        print("\n测试终止：重排序服务初始化失败")
        return
    
    # 执行各项测试
    test_single_rerank(rerank_service)
    test_batch_rerank(rerank_service)
    test_performance_metrics(rerank_service)
    test_edge_cases(rerank_service)
    
    # 重置统计信息测试
    print("\n=== 测试统计信息重置 ===")
    rerank_service.reset_stats()
    stats = rerank_service.get_stats()
    print(f"✓ 重置后统计信息: {stats}")
    
    print("\n=== 基础重排序功能测试完成 ===")

if __name__ == "__main__":
    main()