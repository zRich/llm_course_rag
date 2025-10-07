#!/usr/bin/env python3
"""Chunk参数优化实验主脚本"""

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Optional

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from chunk_optimizer import ChunkOptimizer, ExperimentResult
from experiment_visualizer import ExperimentVisualizer
from mock_rag_system import MockRAGSystem, MockDocumentGenerator

class ChunkExperimentRunner:
    """Chunk实验运行器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rag_system = None
        self.optimizer = None
        self.results = []
        
    def setup_system(self):
        """设置RAG系统和测试数据"""
        print("🚀 正在初始化RAG系统...")
        
        # 创建RAG系统
        self.rag_system = MockRAGSystem()
        
        # 生成测试文档
        print(f"📚 生成 {self.config['num_documents']} 个测试文档...")
        documents = MockDocumentGenerator.generate_test_documents(
            self.config['num_documents'],
            self.config['document_length']
        )
        
        # 添加文档到系统
        for doc_id, content in documents.items():
            self.rag_system.add_document(doc_id, content)
        
        # 生成测试查询
        print(f"❓ 生成 {self.config['num_queries']} 个测试查询...")
        test_queries = MockDocumentGenerator.generate_test_queries(
            documents, self.config['num_queries']
        )
        
        # 创建优化器
        self.optimizer = ChunkOptimizer(
            rag_system=self.rag_system,
            test_documents=list(documents.keys()),
            evaluation_queries=test_queries
        )
        
        print(f"✅ 系统初始化完成！文档数: {len(documents)}, 查询数: {len(test_queries)}")
        
    def run_grid_search(self) -> List[ExperimentResult]:
        """运行网格搜索实验"""
        chunk_sizes = self.config['chunk_sizes']
        overlap_ratios = self.config['overlap_ratios']
        
        total_experiments = len(chunk_sizes) * len(overlap_ratios)
        print(f"🔬 开始网格搜索实验，共 {total_experiments} 个实验...")
        
        results = []
        current_experiment = 0
        start_time = time.time()
        
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                current_experiment += 1
                
                print(f"\n[{current_experiment}/{total_experiments}] "
                      f"Chunk大小: {chunk_size}, 重叠比例: {overlap_ratio:.2f}")
                
                # 运行单个实验
                experiment_start = time.time()
                result = self.optimizer._run_single_experiment(chunk_size, overlap_ratio)
                experiment_time = time.time() - experiment_start
                
                results.append(result)
                
                # 显示结果
                print(f"  ✅ 准确率: {result.retrieval_accuracy:.3f}, "
                      f"召回率: {result.retrieval_recall:.3f}, "
                      f"响应时间: {result.response_time:.2f}ms, "
                      f"实验耗时: {experiment_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\n🎉 网格搜索完成！总耗时: {total_time:.2f}s")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """分析实验结果"""
        if not self.results:
            raise ValueError("没有实验结果可分析")
        
        print("\n📊 正在分析实验结果...")
        
        # 创建可视化器
        visualizer = ExperimentVisualizer(self.results)
        
        # 生成分析报告
        report = visualizer.generate_summary_report()
        
        # 显示关键结果
        print("\n🏆 关键结果:")
        
        best_configs = report['best_configurations']
        
        print(f"  最高准确率: {best_configs['highest_accuracy']['accuracy']:.3f} "
              f"(Chunk: {best_configs['highest_accuracy']['chunk_size']}, "
              f"重叠: {best_configs['highest_accuracy']['overlap_ratio']:.2f})")
        
        print(f"  最高召回率: {best_configs['highest_recall']['recall']:.3f} "
              f"(Chunk: {best_configs['highest_recall']['chunk_size']}, "
              f"重叠: {best_configs['highest_recall']['overlap_ratio']:.2f})")
        
        print(f"  最高F1分数: {best_configs['highest_f1']['f1_score']:.3f} "
              f"(Chunk: {best_configs['highest_f1']['chunk_size']}, "
              f"重叠: {best_configs['highest_f1']['overlap_ratio']:.2f})")
        
        print(f"  最快响应: {best_configs['fastest_response']['response_time']:.2f}ms "
              f"(Chunk: {best_configs['fastest_response']['chunk_size']}, "
              f"重叠: {best_configs['fastest_response']['overlap_ratio']:.2f})")
        
        return report
    
    def save_results(self, output_dir: Path):
        """保存实验结果"""
        if not self.results:
            print("⚠️ 没有结果可保存")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始结果
        results_file = output_dir / f"experiment_results_{int(time.time())}.json"
        
        results_data = [
            {
                'chunk_size': r.chunk_size,
                'overlap_ratio': r.overlap_ratio,
                'retrieval_accuracy': r.retrieval_accuracy,
                'retrieval_recall': r.retrieval_recall,
                'response_time': r.response_time,
                'storage_overhead': r.storage_overhead,
                'chunk_count': r.chunk_count,
                'total_tokens': r.total_tokens
            }
            for r in self.results
        ]
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 实验结果已保存到: {results_file}")
        
        # 生成并保存分析报告
        report = self.analyze_results()
        report_file = output_dir / f"analysis_report_{int(time.time())}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 分析报告已保存到: {report_file}")
        
        # 生成可视化图表
        if self.config.get('generate_plots', True):
            self.generate_visualizations(output_dir)
    
    def generate_visualizations(self, output_dir: Path):
        """生成可视化图表"""
        print("\n📈 正在生成可视化图表...")
        
        visualizer = ExperimentVisualizer(self.results)
        
        try:
            # 生成热力图
            heatmap_file = output_dir / "accuracy_heatmap.png"
            visualizer.create_heatmap('retrieval_accuracy', save_path=str(heatmap_file))
            print(f"  ✅ 准确率热力图: {heatmap_file}")
            
            # 生成性能曲线
            curves_file = output_dir / "performance_curves.png"
            visualizer.create_performance_curves(save_path=str(curves_file))
            print(f"  ✅ 性能曲线图: {curves_file}")
            
            # 生成3D表面图
            surface_file = output_dir / "3d_surface.png"
            visualizer.create_3d_surface('retrieval_accuracy', save_path=str(surface_file))
            print(f"  ✅ 3D表面图: {surface_file}")
            
            # 生成相关性矩阵
            corr_file = output_dir / "correlation_matrix.png"
            visualizer.create_correlation_matrix(save_path=str(corr_file))
            print(f"  ✅ 相关性矩阵: {corr_file}")
            
        except Exception as e:
            print(f"⚠️ 生成可视化图表时出错: {str(e)}")
    
    def run_experiment(self, output_dir: Optional[Path] = None):
        """运行完整实验流程"""
        try:
            # 设置系统
            self.setup_system()
            
            # 运行网格搜索
            self.run_grid_search()
            
            # 分析结果
            self.analyze_results()
            
            # 保存结果
            if output_dir:
                self.save_results(output_dir)
            
            print("\n🎉 实验完成！")
            
        except Exception as e:
            print(f"❌ 实验失败: {str(e)}")
            raise

def load_config(config_file: Optional[Path] = None) -> Dict:
    """加载配置文件"""
    default_config = {
        'num_documents': 20,
        'document_length': 2000,
        'num_queries': 30,
        'chunk_sizes': [300, 500, 800, 1000, 1200],
        'overlap_ratios': [0.1, 0.15, 0.2, 0.25, 0.3],
        'generate_plots': True
    }
    
    if config_file and config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config

def create_sample_config(config_file: Path):
    """创建示例配置文件"""
    sample_config = {
        "num_documents": 20,
        "document_length": 2000,
        "num_queries": 30,
        "chunk_sizes": [300, 500, 800, 1000, 1200],
        "overlap_ratios": [0.1, 0.15, 0.2, 0.25, 0.3],
        "generate_plots": True,
        "description": "Chunk参数优化实验配置文件"
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"📝 示例配置文件已创建: {config_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Chunk参数优化实验')
    parser.add_argument('--config', type=Path, help='配置文件路径')
    parser.add_argument('--output', type=Path, default=Path('./results'), help='输出目录')
    parser.add_argument('--create-config', type=Path, help='创建示例配置文件')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    
    args = parser.parse_args()
    
    # 创建示例配置文件
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # 加载配置
    config = load_config(args.config)
    
    # 快速测试模式
    if args.quick:
        config.update({
            'num_documents': 5,
            'num_queries': 10,
            'chunk_sizes': [400, 800],
            'overlap_ratios': [0.1, 0.2],
            'generate_plots': False
        })
        print("🚀 快速测试模式")
    
    # 显示配置
    print("📋 实验配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 运行实验
    runner = ChunkExperimentRunner(config)
    runner.run_experiment(args.output)

if __name__ == '__main__':
    main()