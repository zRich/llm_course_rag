#!/usr/bin/env python3
"""Chunk分块参数优化器"""

import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ExperimentResult:
    """实验结果数据类"""
    chunk_size: int
    overlap_ratio: float
    avg_chunk_length: float
    total_chunks: int
    retrieval_accuracy: float
    retrieval_recall: float
    response_time: float
    storage_overhead: float
    
class ChunkOptimizer:
    """Chunk参数优化器"""
    
    def __init__(self, rag_system, test_documents: List[str], evaluation_queries: List[Dict]):
        self.rag_system = rag_system
        self.test_documents = test_documents
        self.evaluation_queries = evaluation_queries
        self.results: List[ExperimentResult] = []
        
    def run_grid_search(self, 
                       chunk_sizes: List[int] = [300, 500, 800, 1000, 1200],
                       overlap_ratios: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3]) -> List[ExperimentResult]:
        """运行网格搜索实验"""
        print(f"🔍 开始网格搜索实验...")
        print(f"📊 参数组合数量: {len(chunk_sizes)} × {len(overlap_ratios)} = {len(chunk_sizes) * len(overlap_ratios)}")
        
        total_experiments = len(chunk_sizes) * len(overlap_ratios)
        current_experiment = 0
        
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                current_experiment += 1
                print(f"\n🧪 实验 {current_experiment}/{total_experiments}: chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")
                
                result = self._run_single_experiment(chunk_size, overlap_ratio)
                self.results.append(result)
                
                # 显示实时结果
                print(f"   ✅ 准确率: {result.retrieval_accuracy:.3f}")
                print(f"   ✅ 召回率: {result.retrieval_recall:.3f}")
                print(f"   ⏱️  响应时间: {result.response_time:.2f}ms")
                
        return self.results
    
    def _run_single_experiment(self, chunk_size: int, overlap_ratio: float) -> ExperimentResult:
        """运行单个参数组合实验"""
        # 1. 重新配置分块参数
        self._reconfigure_chunking(chunk_size, overlap_ratio)
        
        # 2. 重新处理测试文档
        chunk_stats = self._reprocess_documents()
        
        # 3. 运行检索评估
        retrieval_metrics = self._evaluate_retrieval()
        
        # 4. 计算存储开销
        storage_overhead = self._calculate_storage_overhead(overlap_ratio)
        
        return ExperimentResult(
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
            avg_chunk_length=chunk_stats['avg_length'],
            total_chunks=chunk_stats['total_count'],
            retrieval_accuracy=retrieval_metrics['accuracy'],
            retrieval_recall=retrieval_metrics['recall'],
            response_time=retrieval_metrics['avg_response_time'],
            storage_overhead=storage_overhead
        )
    
    def _reconfigure_chunking(self, chunk_size: int, overlap_ratio: float):
        """重新配置分块参数"""
        # 这里需要根据实际的RAG系统接口调整
        if hasattr(self.rag_system, 'chunk_manager'):
            self.rag_system.chunk_manager.chunk_size = chunk_size
            self.rag_system.chunk_manager.overlap_ratio = overlap_ratio
        elif hasattr(self.rag_system, 'set_chunk_params'):
            self.rag_system.set_chunk_params(chunk_size, overlap_ratio)
    
    def _reprocess_documents(self) -> Dict[str, float]:
        """重新处理文档并返回统计信息"""
        total_chunks = 0
        total_length = 0
        
        for doc_path in self.test_documents:
            # 模拟文档处理
            if hasattr(self.rag_system, 'process_document'):
                chunks = self.rag_system.process_document(doc_path)
                total_chunks += len(chunks)
                total_length += sum(len(chunk.content) for chunk in chunks)
            else:
                # 模拟处理结果
                chunk_count = np.random.randint(10, 50)
                avg_length = np.random.randint(200, 1000)
                total_chunks += chunk_count
                total_length += chunk_count * avg_length
        
        return {
            'total_count': total_chunks,
            'avg_length': total_length / total_chunks if total_chunks > 0 else 0
        }
    
    def _evaluate_retrieval(self) -> Dict[str, float]:
        """评估检索性能"""
        correct_retrievals = 0
        total_relevant = 0
        total_retrieved = 0
        total_time = 0
        
        for query_data in self.evaluation_queries:
            query = query_data['query']
            expected_chunks = query_data.get('expected_chunks', [])
            
            # 执行检索
            start_time = time.time()
            if hasattr(self.rag_system, 'search'):
                results = self.rag_system.search(query, top_k=5)
            else:
                # 模拟检索结果
                results = [{'chunk_id': f'chunk_{i}'} for i in range(5)]
            end_time = time.time()
            
            total_time += (end_time - start_time) * 1000  # 转换为毫秒
            
            # 计算准确率和召回率
            retrieved_ids = [r.get('chunk_id') for r in results]
            
            # 计算交集
            relevant_retrieved = len(set(retrieved_ids) & set(expected_chunks))
            correct_retrievals += relevant_retrieved
            total_retrieved += len(retrieved_ids)
            total_relevant += len(expected_chunks)
        
        accuracy = correct_retrievals / total_retrieved if total_retrieved > 0 else 0
        recall = correct_retrievals / total_relevant if total_relevant > 0 else 0
        avg_response_time = total_time / len(self.evaluation_queries) if self.evaluation_queries else 0
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'avg_response_time': avg_response_time
        }
    
    def _calculate_storage_overhead(self, overlap_ratio: float) -> float:
        """计算存储开销"""
        # 重叠比例直接影响存储开销
        base_storage = 1.0
        overhead = overlap_ratio * 0.8  # 简化计算
        return base_storage + overhead
    
    def get_best_parameters(self) -> ExperimentResult:
        """获取最佳参数组合"""
        if not self.results:
            raise ValueError("没有实验结果，请先运行实验")
        
        # 综合评分：准确率权重0.4，召回率权重0.3，响应时间权重0.2，存储开销权重0.1
        best_result = None
        best_score = -1
        
        for result in self.results:
            # 归一化指标（响应时间和存储开销需要取倒数）
            normalized_accuracy = result.retrieval_accuracy
            normalized_recall = result.retrieval_recall
            normalized_time = 1 / (1 + result.response_time / 100)  # 归一化响应时间
            normalized_storage = 1 / result.storage_overhead  # 归一化存储开销
            
            # 计算综合评分
            score = (0.4 * normalized_accuracy + 
                    0.3 * normalized_recall + 
                    0.2 * normalized_time + 
                    0.1 * normalized_storage)
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def save_results(self, filepath: str):
        """保存实验结果"""
        results_data = []
        for result in self.results:
            results_data.append({
                'chunk_size': result.chunk_size,
                'overlap_ratio': result.overlap_ratio,
                'avg_chunk_length': result.avg_chunk_length,
                'total_chunks': result.total_chunks,
                'retrieval_accuracy': result.retrieval_accuracy,
                'retrieval_recall': result.retrieval_recall,
                'response_time': result.response_time,
                'storage_overhead': result.storage_overhead
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 实验结果已保存到: {filepath}")
    
    def load_results(self, filepath: str):
        """加载实验结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        self.results = []
        for data in results_data:
            result = ExperimentResult(**data)
            self.results.append(result)
        
        print(f"📁 已加载 {len(self.results)} 个实验结果")
    
    def run_parallel_experiments(self, 
                               chunk_sizes: List[int] = [300, 500, 800, 1000, 1200],
                               overlap_ratios: List[float] = [0.1, 0.15, 0.2, 0.25, 0.3],
                               max_workers: int = 4) -> List[ExperimentResult]:
        """并行运行实验以提高效率"""
        print(f"🚀 开始并行网格搜索实验 (最大工作线程: {max_workers})...")
        
        # 生成所有参数组合
        param_combinations = [(size, ratio) for size in chunk_sizes for ratio in overlap_ratios]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self._run_single_experiment, size, ratio) 
                      for size, ratio in param_combinations]
            
            # 收集结果
            for i, future in enumerate(futures):
                result = future.result()
                self.results.append(result)
                print(f"✅ 完成实验 {i+1}/{len(param_combinations)}: "
                      f"size={result.chunk_size}, ratio={result.overlap_ratio:.2f}, "
                      f"accuracy={result.retrieval_accuracy:.3f}")
        
        return self.results