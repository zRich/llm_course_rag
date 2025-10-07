#!/usr/bin/env python3
"""
增强RAG查询测试脚本

测试EnhancedRAGSystem的功能：
1. 基本RAG查询与重排序对比
2. 批量查询处理
3. 元数据过滤功能
4. 性能统计和监控
5. 健康检查
"""

import sys
import os
import time
import json
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.enhanced_rag import EnhancedRAGSystem
from app.rag.rag_service import RAGService
from app.config import get_settings

def initialize_services():
    """初始化RAG服务"""
    print("\n=== 初始化服务 ===")
    
    try:
        # 获取配置
        settings = get_settings()
        
        # 初始化标准RAG服务
        standard_rag = RAGService()
        print("✓ 标准RAG服务初始化成功")
        
        # 初始化增强RAG服务
        enhanced_rag = EnhancedRAGSystem(rag_service=standard_rag)
        print("✓ 增强RAG服务初始化成功")
        
        return standard_rag, enhanced_rag
        
    except Exception as e:
        print(f"✗ 服务初始化失败: {e}")
        return None, None

def test_basic_query_comparison(standard_rag: RAGService, enhanced_rag: EnhancedRAGSystem):
    """测试基本查询对比"""
    print("\n=== 测试基本查询对比 ===")
    
    query = "什么是机器学习？"
    
    try:
        # 标准RAG查询
        print("\n--- 标准RAG查询 ---")
        start_time = time.time()
        standard_result = standard_rag.query(
            query=query,
            top_k=5,
            collection_name="default"
        )
        standard_time = time.time() - start_time
        
        print(f"✓ 标准RAG查询完成，耗时: {standard_time:.3f}秒")
        print(f"✓ 返回文档数量: {len(standard_result.documents)}")
        
        if standard_result.documents:
            print("✓ 标准RAG前3个结果:")
            for i, (doc, score) in enumerate(zip(standard_result.documents[:3], standard_result.scores[:3])):
                print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:100]}...")
        
        # 增强RAG查询（启用重排序）
        print("\n--- 增强RAG查询（启用重排序）---")
        start_time = time.time()
        enhanced_result = enhanced_rag.query(
            query=query,
            top_k=5,
            collection_name="default",
            enable_rerank=True,
            rerank_top_k=3
        )
        enhanced_time = time.time() - start_time
        
        print(f"✓ 增强RAG查询完成，耗时: {enhanced_time:.3f}秒")
        print(f"✓ 返回文档数量: {len(enhanced_result.documents)}")
        print(f"✓ 重排序处理时间: {enhanced_result.rerank_processing_time:.3f}秒")
        
        if enhanced_result.documents:
            print("✓ 增强RAG前3个结果:")
            for i, (doc, score) in enumerate(zip(enhanced_result.documents[:3], enhanced_result.scores[:3])):
                print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:100]}...")
        
        # 对比分析
        print("\n--- 对比分析 ---")
        print(f"✓ 查询时间对比: 标准={standard_time:.3f}秒, 增强={enhanced_time:.3f}秒")
        
        if enhanced_result.original_scores and enhanced_result.rerank_scores:
            print("✓ 分数变化分析:")
            for i, (orig, rerank) in enumerate(zip(enhanced_result.original_scores[:3], enhanced_result.rerank_scores[:3])):
                change = rerank - orig
                print(f"  文档{i+1}: 原始={orig:.4f}, 重排序={rerank:.4f}, 变化={change:+.4f}")
        
        return standard_result, enhanced_result
        
    except Exception as e:
        print(f"✗ 查询对比测试失败: {e}")
        return None, None

def test_batch_query(enhanced_rag: EnhancedRAGSystem):
    """测试批量查询"""
    print("\n=== 测试批量查询 ===")
    
    queries = [
        "什么是深度学习？",
        "Python编程语言有什么特点？",
        "人工智能的应用领域有哪些？",
        "机器学习算法如何分类？"
    ]
    
    try:
        # 批量查询（不启用重排序）
        print("\n--- 批量查询（不启用重排序）---")
        start_time = time.time()
        batch_results_standard = enhanced_rag.batch_query(
            queries=queries,
            top_k=3,
            collection_name="default",
            enable_rerank=False
        )
        batch_time_standard = time.time() - start_time
        
        print(f"✓ 标准批量查询完成，耗时: {batch_time_standard:.3f}秒")
        print(f"✓ 处理查询数量: {len(batch_results_standard)}")
        
        # 批量查询（启用重排序）
        print("\n--- 批量查询（启用重排序）---")
        start_time = time.time()
        batch_results_enhanced = enhanced_rag.batch_query(
            queries=queries,
            top_k=3,
            collection_name="default",
            enable_rerank=True,
            rerank_top_k=2
        )
        batch_time_enhanced = time.time() - start_time
        
        print(f"✓ 增强批量查询完成，耗时: {batch_time_enhanced:.3f}秒")
        print(f"✓ 处理查询数量: {len(batch_results_enhanced)}")
        
        # 结果分析
        print("\n--- 批量查询结果分析 ---")
        for i, (query, std_result, enh_result) in enumerate(zip(queries, batch_results_standard, batch_results_enhanced)):
            print(f"\n查询 {i+1}: {query}")
            print(f"  标准结果文档数: {len(std_result.documents)}")
            print(f"  增强结果文档数: {len(enh_result.documents)}")
            if enh_result.rerank_processing_time:
                print(f"  重排序时间: {enh_result.rerank_processing_time:.3f}秒")
        
        return batch_results_standard, batch_results_enhanced
        
    except Exception as e:
        print(f"✗ 批量查询测试失败: {e}")
        return None, None

def test_metadata_filtering(enhanced_rag: EnhancedRAGSystem):
    """测试元数据过滤"""
    print("\n=== 测试元数据过滤 ===")
    
    query = "机器学习算法"
    
    try:
        # 不使用过滤器的查询
        print("\n--- 无过滤器查询 ---")
        result_no_filter = enhanced_rag.query(
            query=query,
            top_k=5,
            collection_name="default",
            enable_rerank=True
        )
        print(f"✓ 无过滤器查询完成，返回文档数: {len(result_no_filter.documents)}")
        
        # 使用元数据过滤器的查询
        print("\n--- 使用元数据过滤器查询 ---")
        metadata_filter = {
            "source": "tutorial",  # 假设的元数据字段
            "difficulty": "beginner"
        }
        
        result_with_filter = enhanced_rag.query(
            query=query,
            top_k=5,
            collection_name="default",
            enable_rerank=True,
            metadata_filter=metadata_filter
        )
        print(f"✓ 元数据过滤查询完成，返回文档数: {len(result_with_filter.documents)}")
        
        # 对比结果
        print("\n--- 过滤结果对比 ---")
        print(f"✓ 无过滤器结果数: {len(result_no_filter.documents)}")
        print(f"✓ 有过滤器结果数: {len(result_with_filter.documents)}")
        
        if result_with_filter.documents:
            print("✓ 过滤后的前3个结果:")
            for i, doc in enumerate(result_with_filter.documents[:3]):
                print(f"  {i+1}. 内容: {doc.content[:80]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"      元数据: {doc.metadata}")
        
        return result_no_filter, result_with_filter
        
    except Exception as e:
        print(f"✗ 元数据过滤测试失败: {e}")
        return None, None

def test_performance_stats(enhanced_rag: EnhancedRAGSystem):
    """测试性能统计"""
    print("\n=== 测试性能统计 ===")
    
    try:
        # 获取当前统计信息
        stats = enhanced_rag.get_stats()
        print(f"✓ 当前统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        # 执行一些查询来更新统计
        test_queries = [
            "测试查询1",
            "测试查询2",
            "测试查询3"
        ]
        
        for query in test_queries:
            enhanced_rag.query(
                query=query,
                top_k=3,
                collection_name="default",
                enable_rerank=True
            )
        
        # 获取更新后的统计信息
        updated_stats = enhanced_rag.get_stats()
        print(f"\n✓ 更新后统计信息: {json.dumps(updated_stats, indent=2, ensure_ascii=False)}")
        
        # 分析统计变化
        if 'total_queries' in stats and 'total_queries' in updated_stats:
            query_increase = updated_stats['total_queries'] - stats['total_queries']
            print(f"✓ 查询数量增加: {query_increase}")
        
        return updated_stats
        
    except Exception as e:
        print(f"✗ 性能统计测试失败: {e}")
        return None

def test_health_check(enhanced_rag: EnhancedRAGSystem):
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    
    try:
        health_status = enhanced_rag.health_check()
        print(f"✓ 健康检查结果: {json.dumps(health_status, indent=2, ensure_ascii=False)}")
        
        # 检查各组件状态
        if health_status.get('status') == 'healthy':
            print("✓ 系统整体状态健康")
        else:
            print("⚠ 系统状态异常")
        
        # 检查各个服务组件
        components = health_status.get('components', {})
        for component, status in components.items():
            if status.get('status') == 'healthy':
                print(f"✓ {component}: 健康")
            else:
                print(f"⚠ {component}: {status.get('status', '未知')}")
        
        return health_status
        
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return None

def test_compare_before_after_rerank(enhanced_rag: EnhancedRAGSystem):
    """测试重排序前后对比"""
    print("\n=== 测试重排序前后对比 ===")
    
    query = "深度学习神经网络"
    
    try:
        comparison_result = enhanced_rag.compare_with_without_rerank(
            query=query,
            top_k=5,
            collection_name="default",
            rerank_top_k=3
        )
        
        print(f"✓ 重排序对比完成")
        print(f"✓ 原始结果数量: {len(comparison_result['without_rerank']['documents'])}")
        print(f"✓ 重排序结果数量: {len(comparison_result['with_rerank']['documents'])}")
        print(f"✓ 重排序处理时间: {comparison_result['rerank_time']:.3f}秒")
        
        # 显示前3个结果的对比
        print("\n--- 前3个结果对比 ---")
        without_docs = comparison_result['without_rerank']['documents'][:3]
        without_scores = comparison_result['without_rerank']['scores'][:3]
        with_docs = comparison_result['with_rerank']['documents'][:3]
        with_scores = comparison_result['with_rerank']['scores'][:3]
        
        print("\n原始排序:")
        for i, (doc, score) in enumerate(zip(without_docs, without_scores)):
            print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:60]}...")
        
        print("\n重排序后:")
        for i, (doc, score) in enumerate(zip(with_docs, with_scores)):
            print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:60]}...")
        
        return comparison_result
        
    except Exception as e:
        print(f"✗ 重排序对比测试失败: {e}")
        return None

def main():
    """主测试函数"""
    print("开始增强RAG查询功能测试...")
    
    # 初始化服务
    standard_rag, enhanced_rag = initialize_services()
    if not standard_rag or not enhanced_rag:
        print("\n测试终止：服务初始化失败")
        return
    
    # 执行各项测试
    test_basic_query_comparison(standard_rag, enhanced_rag)
    test_batch_query(enhanced_rag)
    test_metadata_filtering(enhanced_rag)
    test_performance_stats(enhanced_rag)
    test_health_check(enhanced_rag)
    test_compare_before_after_rerank(enhanced_rag)
    
    # 最终统计信息
    print("\n=== 最终系统状态 ===")
    final_stats = enhanced_rag.get_stats()
    final_health = enhanced_rag.health_check()
    
    print(f"✓ 最终统计信息: {json.dumps(final_stats, indent=2, ensure_ascii=False)}")
    print(f"✓ 最终健康状态: {final_health.get('status', '未知')}")
    
    print("\n=== 增强RAG查询功能测试完成 ===")

if __name__ == "__main__":
    main()