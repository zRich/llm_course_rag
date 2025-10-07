#!/usr/bin/env python3
"""
Rerank A/B测试脚本

测试RerankABTest的功能：
1. A/B测试配置和用户分组
2. 对照组和实验组查询对比
3. 用户反馈收集和分析
4. 统计显著性检验
5. 测试报告生成
"""

import sys
import os
import time
import json
import random
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.rerank_ab_test import RerankABTest
from app.rag.rag_service import RAGService
from app.config import get_settings

def initialize_ab_test():
    """初始化A/B测试"""
    print("\n=== 初始化A/B测试 ===")
    
    try:
        # 获取配置
        settings = get_settings()
        
        # 初始化RAG服务
        rag_service = RAGService()
        print("✓ RAG服务初始化成功")
        
        # 初始化A/B测试
        ab_test = RerankABTest(
            rag_service=rag_service,
            test_name="rerank_effectiveness_test",
            description="测试重排序功能对查询结果质量的影响",
            control_ratio=0.5,  # 50%对照组，50%实验组
            enable_rerank_in_treatment=True
        )
        print("✓ A/B测试初始化成功")
        
        return ab_test
        
    except Exception as e:
        print(f"✗ A/B测试初始化失败: {e}")
        return None

def test_user_assignment(ab_test: RerankABTest):
    """测试用户分组"""
    print("\n=== 测试用户分组 ===")
    
    # 测试多个用户的分组
    test_users = [f"user_{i}" for i in range(20)]
    group_counts = {"control": 0, "treatment": 0}
    
    print("\n--- 用户分组结果 ---")
    for user_id in test_users:
        group = ab_test.get_user_group(user_id)
        group_counts[group] += 1
        print(f"  {user_id}: {group}")
    
    print(f"\n✓ 分组统计: 对照组={group_counts['control']}, 实验组={group_counts['treatment']}")
    
    # 验证分组一致性
    print("\n--- 验证分组一致性 ---")
    for user_id in test_users[:5]:
        group1 = ab_test.get_user_group(user_id)
        group2 = ab_test.get_user_group(user_id)
        if group1 == group2:
            print(f"✓ {user_id}: 分组一致 ({group1})")
        else:
            print(f"✗ {user_id}: 分组不一致 ({group1} vs {group2})")
    
    return test_users, group_counts

def test_ab_queries(ab_test: RerankABTest, test_users: List[str]):
    """测试A/B查询"""
    print("\n=== 测试A/B查询 ===")
    
    test_queries = [
        "什么是机器学习？",
        "深度学习的基本原理",
        "Python编程语言特点",
        "人工智能应用领域",
        "神经网络结构设计"
    ]
    
    query_results = []
    
    # 为每个用户执行查询
    for user_id in test_users[:10]:  # 只测试前10个用户
        user_group = ab_test.get_user_group(user_id)
        query = random.choice(test_queries)
        
        print(f"\n--- 用户 {user_id} ({user_group}组) 查询: {query} ---")
        
        try:
            start_time = time.time()
            result = ab_test.query(
                user_id=user_id,
                query=query,
                top_k=5,
                collection_name="default"
            )
            query_time = time.time() - start_time
            
            print(f"✓ 查询完成，耗时: {query_time:.3f}秒")
            print(f"✓ 返回文档数: {len(result.documents)}")
            print(f"✓ 使用重排序: {result.rerank_enabled}")
            
            if result.rerank_enabled and result.rerank_processing_time:
                print(f"✓ 重排序时间: {result.rerank_processing_time:.3f}秒")
            
            # 显示前2个结果
            if result.documents:
                print("✓ 前2个结果:")
                for i, (doc, score) in enumerate(zip(result.documents[:2], result.scores[:2])):
                    print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:60]}...")
            
            query_results.append({
                'user_id': user_id,
                'group': user_group,
                'query': query,
                'result': result,
                'query_time': query_time
            })
            
        except Exception as e:
            print(f"✗ 查询失败: {e}")
    
    return query_results

def test_batch_ab_test(ab_test: RerankABTest):
    """测试批量A/B测试"""
    print("\n=== 测试批量A/B测试 ===")
    
    # 准备测试数据
    test_data = [
        {"user_id": f"batch_user_{i}", "query": f"批量测试查询{i}"}
        for i in range(8)
    ]
    
    try:
        start_time = time.time()
        batch_results = ab_test.batch_test(
            test_data=test_data,
            top_k=3,
            collection_name="default"
        )
        batch_time = time.time() - start_time
        
        print(f"✓ 批量测试完成，耗时: {batch_time:.3f}秒")
        print(f"✓ 处理用户数: {len(batch_results)}")
        
        # 分析批量结果
        control_count = sum(1 for r in batch_results if r['group'] == 'control')
        treatment_count = sum(1 for r in batch_results if r['group'] == 'treatment')
        
        print(f"✓ 对照组查询数: {control_count}")
        print(f"✓ 实验组查询数: {treatment_count}")
        
        # 显示部分结果
        print("\n--- 部分批量测试结果 ---")
        for i, result in enumerate(batch_results[:4]):
            print(f"  用户{i+1}: {result['user_id']} ({result['group']}组)")
            print(f"    查询: {result['query']}")
            print(f"    结果数: {len(result['result'].documents)}")
            print(f"    重排序: {result['result'].rerank_enabled}")
        
        return batch_results
        
    except Exception as e:
        print(f"✗ 批量A/B测试失败: {e}")
        return None

def test_user_feedback(ab_test: RerankABTest, query_results: List[Dict]):
    """测试用户反馈"""
    print("\n=== 测试用户反馈 ===")
    
    # 模拟用户反馈
    feedback_types = ['relevance', 'satisfaction', 'usefulness']
    
    for result in query_results[:6]:  # 为前6个查询添加反馈
        user_id = result['user_id']
        query = result['query']
        
        # 模拟不同类型的反馈
        for feedback_type in feedback_types:
            # 实验组通常有更高的评分（模拟重排序效果）
            if result['group'] == 'treatment':
                score = random.uniform(3.5, 5.0)  # 3.5-5.0分
            else:
                score = random.uniform(2.5, 4.5)  # 2.5-4.5分
            
            try:
                ab_test.add_feedback(
                    user_id=user_id,
                    query=query,
                    feedback_type=feedback_type,
                    score=score,
                    comment=f"模拟{feedback_type}反馈"
                )
                print(f"✓ 添加反馈: {user_id} - {feedback_type}: {score:.2f}")
                
            except Exception as e:
                print(f"✗ 添加反馈失败: {e}")
    
    print("✓ 用户反馈添加完成")

def test_metrics_calculation(ab_test: RerankABTest):
    """测试指标计算"""
    print("\n=== 测试指标计算 ===")
    
    try:
        metrics = ab_test.calculate_metrics()
        print(f"✓ 指标计算完成")
        print(f"✓ 指标详情: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
        
        # 分析关键指标
        if 'control' in metrics and 'treatment' in metrics:
            control_metrics = metrics['control']
            treatment_metrics = metrics['treatment']
            
            print("\n--- 关键指标对比 ---")
            print(f"对照组查询数: {control_metrics.get('total_queries', 0)}")
            print(f"实验组查询数: {treatment_metrics.get('total_queries', 0)}")
            
            if 'avg_query_time' in control_metrics and 'avg_query_time' in treatment_metrics:
                print(f"对照组平均查询时间: {control_metrics['avg_query_time']:.3f}秒")
                print(f"实验组平均查询时间: {treatment_metrics['avg_query_time']:.3f}秒")
            
            # 反馈指标对比
            feedback_types = ['relevance', 'satisfaction', 'usefulness']
            for feedback_type in feedback_types:
                control_score = control_metrics.get('feedback', {}).get(feedback_type, {}).get('avg_score')
                treatment_score = treatment_metrics.get('feedback', {}).get(feedback_type, {}).get('avg_score')
                
                if control_score is not None and treatment_score is not None:
                    improvement = treatment_score - control_score
                    print(f"{feedback_type}: 对照组={control_score:.2f}, 实验组={treatment_score:.2f}, 提升={improvement:+.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ 指标计算失败: {e}")
        return None

def test_statistical_significance(ab_test: RerankABTest):
    """测试统计显著性"""
    print("\n=== 测试统计显著性 ===")
    
    try:
        significance_results = ab_test.test_statistical_significance()
        print(f"✓ 统计显著性检验完成")
        print(f"✓ 检验结果: {json.dumps(significance_results, indent=2, ensure_ascii=False)}")
        
        # 分析显著性结果
        for metric, result in significance_results.items():
            if isinstance(result, dict) and 'p_value' in result:
                p_value = result['p_value']
                is_significant = result.get('is_significant', False)
                
                print(f"\n{metric}:")
                print(f"  p值: {p_value:.4f}")
                print(f"  统计显著: {'是' if is_significant else '否'}")
                
                if 'effect_size' in result:
                    print(f"  效应大小: {result['effect_size']:.4f}")
        
        return significance_results
        
    except Exception as e:
        print(f"✗ 统计显著性检验失败: {e}")
        return None

def test_report_generation(ab_test: RerankABTest):
    """测试报告生成"""
    print("\n=== 测试报告生成 ===")
    
    try:
        # 生成汇总报告
        summary_report = ab_test.generate_summary_report()
        print(f"✓ 汇总报告生成完成")
        print(f"\n--- 汇总报告 ---")
        print(summary_report)
        
        # 导出详细结果
        export_data = ab_test.export_results()
        print(f"\n✓ 结果导出完成，数据大小: {len(str(export_data))} 字符")
        
        # 显示导出数据的结构
        if isinstance(export_data, dict):
            print(f"✓ 导出数据包含: {list(export_data.keys())}")
            
            if 'test_config' in export_data:
                config = export_data['test_config']
                print(f"  测试配置: {config.get('test_name', 'N/A')}")
                print(f"  对照组比例: {config.get('control_ratio', 'N/A')}")
            
            if 'results' in export_data:
                results = export_data['results']
                print(f"  结果记录数: {len(results)}")
        
        return summary_report, export_data
        
    except Exception as e:
        print(f"✗ 报告生成失败: {e}")
        return None, None

def test_reset_and_cleanup(ab_test: RerankABTest):
    """测试重置和清理"""
    print("\n=== 测试重置和清理 ===")
    
    try:
        # 获取重置前的统计
        pre_reset_metrics = ab_test.calculate_metrics()
        pre_reset_queries = sum([
            pre_reset_metrics.get('control', {}).get('total_queries', 0),
            pre_reset_metrics.get('treatment', {}).get('total_queries', 0)
        ])
        
        print(f"✓ 重置前总查询数: {pre_reset_queries}")
        
        # 执行重置
        ab_test.reset_test()
        print("✓ A/B测试重置完成")
        
        # 获取重置后的统计
        post_reset_metrics = ab_test.calculate_metrics()
        post_reset_queries = sum([
            post_reset_metrics.get('control', {}).get('total_queries', 0),
            post_reset_metrics.get('treatment', {}).get('total_queries', 0)
        ])
        
        print(f"✓ 重置后总查询数: {post_reset_queries}")
        
        if post_reset_queries == 0:
            print("✓ 重置功能正常工作")
        else:
            print("⚠ 重置可能未完全清理数据")
        
    except Exception as e:
        print(f"✗ 重置和清理失败: {e}")

def main():
    """主测试函数"""
    print("开始Rerank A/B测试功能测试...")
    
    # 初始化A/B测试
    ab_test = initialize_ab_test()
    if not ab_test:
        print("\n测试终止：A/B测试初始化失败")
        return
    
    # 执行各项测试
    test_users, group_counts = test_user_assignment(ab_test)
    query_results = test_ab_queries(ab_test, test_users)
    batch_results = test_batch_ab_test(ab_test)
    test_user_feedback(ab_test, query_results)
    metrics = test_metrics_calculation(ab_test)
    significance_results = test_statistical_significance(ab_test)
    summary_report, export_data = test_report_generation(ab_test)
    
    # 最终总结
    print("\n=== 测试总结 ===")
    print(f"✓ 测试用户数: {len(test_users)}")
    print(f"✓ 对照组用户: {group_counts['control']}")
    print(f"✓ 实验组用户: {group_counts['treatment']}")
    print(f"✓ 单独查询数: {len(query_results)}")
    if batch_results:
        print(f"✓ 批量查询数: {len(batch_results)}")
    
    # 重置测试（可选）
    print("\n--- 是否执行重置测试？---")
    # test_reset_and_cleanup(ab_test)  # 取消注释以执行重置测试
    
    print("\n=== Rerank A/B测试功能测试完成 ===")

if __name__ == "__main__":
    main()