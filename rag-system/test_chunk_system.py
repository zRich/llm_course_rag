#!/usr/bin/env python3
"""Chunk实验系统功能测试脚本"""

import sys
import time
from pathlib import Path

# 添加实验目录到Python路径
exp_dir = Path(__file__).parent / "experiments" / "chunk_optimization"
sys.path.append(str(exp_dir))

try:
    from chunk_optimizer import ChunkOptimizer, ExperimentResult
    from experiment_visualizer import ExperimentVisualizer
    from mock_rag_system import MockRAGSystem, MockDocumentGenerator
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有必要的文件都已创建")
    sys.exit(1)

def test_mock_rag_system():
    """测试模拟RAG系统"""
    print("\n🧪 测试模拟RAG系统...")
    
    try:
        # 创建RAG系统
        rag_system = MockRAGSystem()
        
        # 生成测试文档
        documents = MockDocumentGenerator.generate_test_documents(3, 1000)
        print(f"  ✅ 生成了 {len(documents)} 个测试文档")
        
        # 添加文档
        for doc_id, content in documents.items():
            rag_system.add_document(doc_id, content)
        
        print(f"  ✅ 添加了 {len(documents)} 个文档到RAG系统")
        
        # 测试搜索
        results = rag_system.search("测试查询", top_k=2)
        print(f"  ✅ 搜索返回了 {len(results)} 个结果")
        
        # 获取统计信息
        stats = rag_system.get_statistics()
        print(f"  ✅ 系统统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模拟RAG系统测试失败: {e}")
        return False

def test_chunk_optimizer():
    """测试Chunk优化器"""
    print("\n🔧 测试Chunk优化器...")
    
    try:
        # 创建RAG系统和测试数据
        rag_system = MockRAGSystem()
        documents = MockDocumentGenerator.generate_test_documents(3, 1000)
        
        for doc_id, content in documents.items():
            rag_system.add_document(doc_id, content)
        
        # 生成测试查询
        test_queries = MockDocumentGenerator.generate_test_queries(documents, 5)
        print(f"  ✅ 生成了 {len(test_queries)} 个测试查询")
        
        # 创建优化器
        optimizer = ChunkOptimizer(
            rag_system=rag_system,
            test_documents=list(documents.keys()),
            evaluation_queries=test_queries
        )
        
        print("  ✅ Chunk优化器创建成功")
        
        # 运行单个实验
        result = optimizer._run_single_experiment(500, 0.2)
        print(f"  ✅ 单个实验完成: 准确率={result.retrieval_accuracy:.3f}, 召回率={result.retrieval_recall:.3f}")
        
        # 运行小规模网格搜索
        results = optimizer.run_grid_search(
            chunk_sizes=[400, 600],
            overlap_ratios=[0.1, 0.2]
        )
        
        print(f"  ✅ 网格搜索完成，共 {len(results)} 个实验")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Chunk优化器测试失败: {e}")
        return False

def test_experiment_visualizer():
    """测试实验可视化器"""
    print("\n📊 测试实验可视化器...")
    
    try:
        # 创建一些模拟实验结果
        results = []
        for chunk_size in [400, 600]:
            for overlap_ratio in [0.1, 0.2]:
                result = ExperimentResult(
                    chunk_size=chunk_size,
                    overlap_ratio=overlap_ratio,
                    avg_chunk_length=chunk_size * 0.8,
                    total_chunks=100,
                    retrieval_accuracy=0.7 + (chunk_size / 1000) * 0.2,
                    retrieval_recall=0.6 + (overlap_ratio * 0.3),
                    response_time=50 + (chunk_size / 10),
                    storage_overhead=1.0 + overlap_ratio
                )
                results.append(result)
        
        print(f"  ✅ 创建了 {len(results)} 个模拟实验结果")
        
        # 创建可视化器
        visualizer = ExperimentVisualizer(results)
        print("  ✅ 实验可视化器创建成功")
        
        # 生成分析报告
        report = visualizer.generate_summary_report()
        print("  ✅ 分析报告生成成功")
        
        # 检查报告内容
        assert 'experiment_summary' in report
        assert 'performance_statistics' in report
        assert 'best_configurations' in report
        
        print(f"  ✅ 报告包含 {len(report)} 个主要部分")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 实验可视化器测试失败: {e}")
        return False

def test_integration():
    """集成测试"""
    print("\n🔗 运行集成测试...")
    
    try:
        start_time = time.time()
        
        # 创建完整的实验流程
        rag_system = MockRAGSystem()
        documents = MockDocumentGenerator.generate_test_documents(5, 800)
        
        for doc_id, content in documents.items():
            rag_system.add_document(doc_id, content)
        
        test_queries = MockDocumentGenerator.generate_test_queries(documents, 8)
        
        optimizer = ChunkOptimizer(
            rag_system=rag_system,
            test_documents=list(documents.keys()),
            evaluation_queries=test_queries
        )
        
        # 运行小规模实验
        results = optimizer.run_grid_search(
            chunk_sizes=[300, 500],
            overlap_ratios=[0.1, 0.2]
        )
        
        # 分析结果
        visualizer = ExperimentVisualizer(results)
        report = visualizer.generate_summary_report()
        
        end_time = time.time()
        
        print(f"  ✅ 集成测试完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"  ✅ 实验结果数量: {len(results)}")
        print(f"  ✅ 最佳准确率: {report['best_configurations']['highest_accuracy']['accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始Chunk实验系统功能测试")
    print("=" * 50)
    
    tests = [
        ("模拟RAG系统", test_mock_rag_system),
        ("Chunk优化器", test_chunk_optimizer),
        ("实验可视化器", test_experiment_visualizer),
        ("集成测试", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统功能正常")
        return True
    else:
        print(f"⚠️ {total - passed} 个测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)