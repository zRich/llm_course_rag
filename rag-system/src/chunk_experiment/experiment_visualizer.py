#!/usr/bin/env python3
"""实验结果可视化分析器"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from chunk_optimizer import ExperimentResult

class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, results: List[ExperimentResult]):
        self.results = results
        self.df = self._create_dataframe()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """将实验结果转换为DataFrame"""
        data = []
        for result in self.results:
            data.append({
                'chunk_size': result.chunk_size,
                'overlap_ratio': result.overlap_ratio,
                'avg_chunk_length': result.avg_chunk_length,
                'total_chunks': result.total_chunks,
                'retrieval_accuracy': result.retrieval_accuracy,
                'retrieval_recall': result.retrieval_recall,
                'response_time': result.response_time,
                'storage_overhead': result.storage_overhead,
                'f1_score': 2 * result.retrieval_accuracy * result.retrieval_recall / 
                           (result.retrieval_accuracy + result.retrieval_recall) 
                           if (result.retrieval_accuracy + result.retrieval_recall) > 0 else 0
            })
        return pd.DataFrame(data)
    
    def create_heatmap(self, metric: str = 'retrieval_accuracy', save_path: str = None) -> None:
        """创建参数热力图"""
        # 创建透视表
        pivot_table = self.df.pivot(index='overlap_ratio', 
                                   columns='chunk_size', 
                                   values=metric)
        
        plt.figure(figsize=(12, 8))
        
        # 创建热力图
        sns.heatmap(pivot_table, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': self._get_metric_label(metric)})
        
        plt.title(f'Chunk参数对{self._get_metric_label(metric)}的影响热力图', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Chunk大小', fontsize=12)
        plt.ylabel('重叠比例', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 热力图已保存到: {save_path}")
        
        plt.show()
    
    def create_performance_curves(self, save_path: str = None) -> None:
        """创建性能曲线图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率vs Chunk大小
        for overlap in self.df['overlap_ratio'].unique():
            subset = self.df[self.df['overlap_ratio'] == overlap]
            axes[0, 0].plot(subset['chunk_size'], subset['retrieval_accuracy'], 
                           marker='o', label=f'重叠比例={overlap:.2f}')
        axes[0, 0].set_title('准确率 vs Chunk大小')
        axes[0, 0].set_xlabel('Chunk大小')
        axes[0, 0].set_ylabel('检索准确率')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 召回率vs重叠比例
        for chunk_size in self.df['chunk_size'].unique():
            subset = self.df[self.df['chunk_size'] == chunk_size]
            axes[0, 1].plot(subset['overlap_ratio'], subset['retrieval_recall'], 
                           marker='s', label=f'Chunk大小={chunk_size}')
        axes[0, 1].set_title('召回率 vs 重叠比例')
        axes[0, 1].set_xlabel('重叠比例')
        axes[0, 1].set_ylabel('检索召回率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 响应时间vs Chunk大小
        for overlap in self.df['overlap_ratio'].unique():
            subset = self.df[self.df['overlap_ratio'] == overlap]
            axes[1, 0].plot(subset['chunk_size'], subset['response_time'], 
                           marker='^', label=f'重叠比例={overlap:.2f}')
        axes[1, 0].set_title('响应时间 vs Chunk大小')
        axes[1, 0].set_xlabel('Chunk大小')
        axes[1, 0].set_ylabel('平均响应时间 (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. F1分数vs参数组合
        scatter = axes[1, 1].scatter(self.df['chunk_size'], self.df['overlap_ratio'], 
                                    c=self.df['f1_score'], s=100, cmap='viridis')
        axes[1, 1].set_title('F1分数热力散点图')
        axes[1, 1].set_xlabel('Chunk大小')
        axes[1, 1].set_ylabel('重叠比例')
        plt.colorbar(scatter, ax=axes[1, 1], label='F1分数')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 性能曲线图已保存到: {save_path}")
        
        plt.show()
    
    def create_3d_surface_plot(self, metric: str = 'retrieval_accuracy', save_path: str = None) -> None:
        """创建3D表面图"""
        # 准备数据
        chunk_sizes = sorted(self.df['chunk_size'].unique())
        overlap_ratios = sorted(self.df['overlap_ratio'].unique())
        
        # 创建网格
        X, Y = np.meshgrid(chunk_sizes, overlap_ratios)
        Z = np.zeros_like(X)
        
        # 填充Z值
        for i, overlap in enumerate(overlap_ratios):
            for j, chunk_size in enumerate(chunk_sizes):
                value = self.df[(self.df['chunk_size'] == chunk_size) & 
                               (self.df['overlap_ratio'] == overlap)][metric]
                Z[i, j] = value.iloc[0] if not value.empty else 0
        
        # 创建3D图
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        
        fig.update_layout(
            title=f'Chunk参数对{self._get_metric_label(metric)}的3D影响图',
            scene=dict(
                xaxis_title='Chunk大小',
                yaxis_title='重叠比例',
                zaxis_title=self._get_metric_label(metric)
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"🎯 3D表面图已保存到: {save_path}")
        
        fig.show()
    
    def create_comparison_radar_chart(self, top_n: int = 5, save_path: str = None) -> None:
        """创建最佳参数组合的雷达图对比"""
        # 计算综合评分并选择前N个
        self.df['composite_score'] = (
            0.3 * self.df['retrieval_accuracy'] +
            0.3 * self.df['retrieval_recall'] +
            0.2 * (1 / (1 + self.df['response_time'] / 100)) +
            0.2 * (1 / self.df['storage_overhead'])
        )
        
        top_results = self.df.nlargest(top_n, 'composite_score')
        
        # 准备雷达图数据
        categories = ['准确率', '召回率', '响应速度', '存储效率']
        
        fig = go.Figure()
        
        for idx, row in top_results.iterrows():
            values = [
                row['retrieval_accuracy'],
                row['retrieval_recall'],
                1 / (1 + row['response_time'] / 100),  # 归一化响应速度
                1 / row['storage_overhead']  # 归一化存储效率
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'Size={int(row["chunk_size"])}, Overlap={row["overlap_ratio"]:.2f}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"前{top_n}个最佳参数组合性能对比",
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"🎯 雷达图已保存到: {save_path}")
        
        fig.show()
    
    def create_correlation_matrix(self, save_path: str = None) -> None:
        """创建指标相关性矩阵"""
        # 选择数值列
        numeric_cols = ['chunk_size', 'overlap_ratio', 'avg_chunk_length', 
                       'total_chunks', 'retrieval_accuracy', 'retrieval_recall', 
                       'response_time', 'storage_overhead', 'f1_score']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # 创建相关性热力图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': '相关系数'})
        
        plt.title('Chunk参数与性能指标相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔗 相关性矩阵已保存到: {save_path}")
        
        plt.show()
    
    def create_pareto_frontier(self, save_path: str = None) -> None:
        """创建帕累托前沿分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 准确率 vs 响应时间的帕累托前沿
        scatter1 = ax1.scatter(self.df['response_time'], self.df['retrieval_accuracy'], 
                              c=self.df['chunk_size'], s=100, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('响应时间 (ms)')
        ax1.set_ylabel('检索准确率')
        ax1.set_title('准确率 vs 响应时间 (颜色=Chunk大小)')
        plt.colorbar(scatter1, ax=ax1, label='Chunk大小')
        
        # 2. 召回率 vs 存储开销的帕累托前沿
        scatter2 = ax2.scatter(self.df['storage_overhead'], self.df['retrieval_recall'], 
                              c=self.df['overlap_ratio'], s=100, cmap='plasma', alpha=0.7)
        ax2.set_xlabel('存储开销')
        ax2.set_ylabel('检索召回率')
        ax2.set_title('召回率 vs 存储开销 (颜色=重叠比例)')
        plt.colorbar(scatter2, ax=ax2, label='重叠比例')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 帕累托前沿图已保存到: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_path: str = None) -> Dict[str, Any]:
        """生成实验总结报告"""
        # 计算统计信息
        best_accuracy = self.df.loc[self.df['retrieval_accuracy'].idxmax()]
        best_recall = self.df.loc[self.df['retrieval_recall'].idxmax()]
        best_f1 = self.df.loc[self.df['f1_score'].idxmax()]
        fastest_response = self.df.loc[self.df['response_time'].idxmin()]
        
        report = {
            'experiment_summary': {
                'total_experiments': len(self.df),
                'chunk_size_range': [int(self.df['chunk_size'].min()), int(self.df['chunk_size'].max())],
                'overlap_ratio_range': [float(self.df['overlap_ratio'].min()), float(self.df['overlap_ratio'].max())]
            },
            'best_configurations': {
                'highest_accuracy': {
                    'chunk_size': int(best_accuracy['chunk_size']),
                    'overlap_ratio': float(best_accuracy['overlap_ratio']),
                    'accuracy': float(best_accuracy['retrieval_accuracy'])
                },
                'highest_recall': {
                    'chunk_size': int(best_recall['chunk_size']),
                    'overlap_ratio': float(best_recall['overlap_ratio']),
                    'recall': float(best_recall['retrieval_recall'])
                },
                'highest_f1': {
                    'chunk_size': int(best_f1['chunk_size']),
                    'overlap_ratio': float(best_f1['overlap_ratio']),
                    'f1_score': float(best_f1['f1_score'])
                },
                'fastest_response': {
                    'chunk_size': int(fastest_response['chunk_size']),
                    'overlap_ratio': float(fastest_response['overlap_ratio']),
                    'response_time': float(fastest_response['response_time'])
                }
            },
            'performance_statistics': {
                'accuracy': {
                    'mean': float(self.df['retrieval_accuracy'].mean()),
                    'std': float(self.df['retrieval_accuracy'].std()),
                    'min': float(self.df['retrieval_accuracy'].min()),
                    'max': float(self.df['retrieval_accuracy'].max())
                },
                'recall': {
                    'mean': float(self.df['retrieval_recall'].mean()),
                    'std': float(self.df['retrieval_recall'].std()),
                    'min': float(self.df['retrieval_recall'].min()),
                    'max': float(self.df['retrieval_recall'].max())
                },
                'response_time': {
                    'mean': float(self.df['response_time'].mean()),
                    'std': float(self.df['response_time'].std()),
                    'min': float(self.df['response_time'].min()),
                    'max': float(self.df['response_time'].max())
                }
            }
        }
        
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📋 实验报告已保存到: {save_path}")
        
        return report
    
    def _get_metric_label(self, metric: str) -> str:
        """获取指标的中文标签"""
        labels = {
            'retrieval_accuracy': '检索准确率',
            'retrieval_recall': '检索召回率',
            'response_time': '响应时间(ms)',
            'storage_overhead': '存储开销',
            'f1_score': 'F1分数',
            'avg_chunk_length': '平均Chunk长度',
            'total_chunks': 'Chunk总数'
        }
        return labels.get(metric, metric)
    
    def create_interactive_dashboard(self, save_path: str = None) -> None:
        """创建交互式仪表板"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('准确率热力图', '召回率vs重叠比例', '响应时间分布', 'F1分数散点图'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # 1. 准确率热力图
        pivot_accuracy = self.df.pivot(index='overlap_ratio', columns='chunk_size', values='retrieval_accuracy')
        fig.add_trace(
            go.Heatmap(z=pivot_accuracy.values, 
                      x=pivot_accuracy.columns, 
                      y=pivot_accuracy.index,
                      colorscale='Viridis'),
            row=1, col=1
        )
        
        # 2. 召回率散点图
        fig.add_trace(
            go.Scatter(x=self.df['overlap_ratio'], 
                      y=self.df['retrieval_recall'],
                      mode='markers',
                      marker=dict(size=8, color=self.df['chunk_size'], colorscale='Plasma'),
                      text=[f'Size: {size}' for size in self.df['chunk_size']]),
            row=1, col=2
        )
        
        # 3. 响应时间直方图
        fig.add_trace(
            go.Histogram(x=self.df['response_time'], nbinsx=20),
            row=2, col=1
        )
        
        # 4. F1分数散点图
        fig.add_trace(
            go.Scatter(x=self.df['chunk_size'], 
                      y=self.df['f1_score'],
                      mode='markers',
                      marker=dict(size=10, color=self.df['overlap_ratio'], colorscale='RdYlBu'),
                      text=[f'Overlap: {ratio:.2f}' for ratio in self.df['overlap_ratio']]),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Chunk参数优化实验交互式仪表板",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"📊 交互式仪表板已保存到: {save_path}")
        
        fig.show()