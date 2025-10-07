#!/usr/bin/env python3
"""基于Streamlit的交互式Chunk参数调优工具"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
import sys

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from chunk_optimizer import ChunkOptimizer, ExperimentResult
from experiment_visualizer import ExperimentVisualizer
from mock_rag_system import MockRAGSystem, MockDocumentGenerator

class InteractiveChunkTuner:
    """交互式Chunk参数调优器"""
    
    def __init__(self):
        self.rag_system = None
        self.optimizer = None
        self.results = []
        
    def initialize_system(self, num_docs: int = 10, doc_length: int = 2000, num_queries: int = 20):
        """初始化RAG系统和测试数据"""
        # 创建RAG系统
        self.rag_system = MockRAGSystem()
        
        # 生成测试文档
        documents = MockDocumentGenerator.generate_test_documents(num_docs, doc_length)
        
        # 添加文档到系统
        for doc_id, content in documents.items():
            self.rag_system.add_document(doc_id, content)
        
        # 生成测试查询
        test_queries = MockDocumentGenerator.generate_test_queries(documents, num_queries)
        
        # 创建优化器
        self.optimizer = ChunkOptimizer(
            rag_system=self.rag_system,
            test_documents=list(documents.keys()),
            evaluation_queries=test_queries
        )
        
        return len(documents), len(test_queries)
    
    def run_single_experiment(self, chunk_size: int, overlap_ratio: float) -> ExperimentResult:
        """运行单个实验"""
        if not self.optimizer:
            raise ValueError("请先初始化系统")
        
        return self.optimizer._run_single_experiment(chunk_size, overlap_ratio)
    
    def run_grid_search(self, chunk_sizes: list, overlap_ratios: list) -> list:
        """运行网格搜索"""
        if not self.optimizer:
            raise ValueError("请先初始化系统")
        
        self.results = []
        total_experiments = len(chunk_sizes) * len(overlap_ratios)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        current_experiment = 0
        
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                current_experiment += 1
                
                # 更新进度
                progress = current_experiment / total_experiments
                progress_bar.progress(progress)
                status_text.text(f'实验进度: {current_experiment}/{total_experiments} - '
                               f'Chunk大小: {chunk_size}, 重叠比例: {overlap_ratio:.2f}')
                
                # 运行实验
                result = self.run_single_experiment(chunk_size, overlap_ratio)
                self.results.append(result)
        
        progress_bar.progress(1.0)
        status_text.text('实验完成！')
        
        return self.results

def main():
    """主函数"""
    st.set_page_config(
        page_title="Chunk参数优化工具",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Chunk参数优化交互式工具")
    st.markdown("---")
    
    # 初始化session state
    if 'tuner' not in st.session_state:
        st.session_state.tuner = InteractiveChunkTuner()
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = []
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 系统初始化参数
        st.subheader("数据集配置")
        num_docs = st.slider("文档数量", min_value=5, max_value=50, value=10, step=5)
        doc_length = st.slider("文档长度", min_value=1000, max_value=5000, value=2000, step=500)
        num_queries = st.slider("测试查询数量", min_value=10, max_value=100, value=20, step=10)
        
        # 初始化按钮
        if st.button("🚀 初始化系统", type="primary"):
            with st.spinner("正在初始化系统..."):
                try:
                    doc_count, query_count = st.session_state.tuner.initialize_system(
                        num_docs, doc_length, num_queries
                    )
                    st.session_state.system_initialized = True
                    st.success(f"✅ 系统初始化成功！\n- 文档数量: {doc_count}\n- 查询数量: {query_count}")
                except Exception as e:
                    st.error(f"❌ 初始化失败: {str(e)}")
        
        st.markdown("---")
        
        # 实验参数配置
        st.subheader("实验参数")
        
        # Chunk大小范围
        chunk_size_range = st.slider(
            "Chunk大小范围",
            min_value=200, max_value=2000, value=(300, 1200), step=100
        )
        chunk_size_step = st.selectbox("Chunk大小步长", [100, 200, 300], index=1)
        
        # 重叠比例范围
        overlap_range = st.slider(
            "重叠比例范围",
            min_value=0.0, max_value=0.5, value=(0.1, 0.3), step=0.05
        )
        overlap_step = st.selectbox("重叠比例步长", [0.05, 0.1, 0.15], index=0)
        
        # 生成参数列表
        chunk_sizes = list(range(chunk_size_range[0], chunk_size_range[1] + 1, chunk_size_step))
        overlap_ratios = [round(x, 2) for x in np.arange(overlap_range[0], overlap_range[1] + overlap_step, overlap_step)]
        
        st.info(f"📊 将运行 {len(chunk_sizes)} × {len(overlap_ratios)} = {len(chunk_sizes) * len(overlap_ratios)} 个实验")
    
    # 主界面
    if not st.session_state.system_initialized:
        st.warning("⚠️ 请先在侧边栏初始化系统")
        st.info("""
        ### 使用说明
        1. 在左侧侧边栏配置数据集参数
        2. 点击"初始化系统"按钮
        3. 配置实验参数范围
        4. 运行网格搜索实验
        5. 查看实验结果和可视化分析
        """)
        return
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 实验运行", "📊 结果分析", "🎯 参数优化", "📋 实验报告"])
    
    with tab1:
        st.header("🧪 实验运行")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("实验参数预览")
            param_df = pd.DataFrame({
                'Chunk大小': chunk_sizes,
                '数量': [len(chunk_sizes)] * len(chunk_sizes)
            })
            st.dataframe(param_df.head())
            
            overlap_df = pd.DataFrame({
                '重叠比例': overlap_ratios,
                '数量': [len(overlap_ratios)] * len(overlap_ratios)
            })
            st.dataframe(overlap_df.head())
        
        with col2:
            st.subheader("快速测试")
            
            # 单个实验测试
            test_chunk_size = st.selectbox("测试Chunk大小", chunk_sizes)
            test_overlap_ratio = st.selectbox("测试重叠比例", overlap_ratios)
            
            if st.button("🔬 运行单个测试"):
                with st.spinner("正在运行测试..."):
                    try:
                        result = st.session_state.tuner.run_single_experiment(
                            test_chunk_size, test_overlap_ratio
                        )
                        
                        st.success("✅ 测试完成！")
                        
                        # 显示结果
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("准确率", f"{result.retrieval_accuracy:.3f}")
                            st.metric("响应时间", f"{result.response_time:.2f}ms")
                        with col_b:
                            st.metric("召回率", f"{result.retrieval_recall:.3f}")
                            st.metric("存储开销", f"{result.storage_overhead:.2f}")
                    
                    except Exception as e:
                        st.error(f"❌ 测试失败: {str(e)}")
        
        st.markdown("---")
        
        # 网格搜索实验
        st.subheader("🔍 网格搜索实验")
        
        if st.button("🚀 开始网格搜索", type="primary"):
            try:
                results = st.session_state.tuner.run_grid_search(chunk_sizes, overlap_ratios)
                st.session_state.experiment_results = results
                st.success(f"✅ 网格搜索完成！共运行了 {len(results)} 个实验")
                
                # 显示最佳结果预览
                if results:
                    best_result = max(results, key=lambda x: x.retrieval_accuracy)
                    st.info(f"🏆 最佳准确率: {best_result.retrieval_accuracy:.3f} "
                           f"(Chunk大小: {best_result.chunk_size}, 重叠比例: {best_result.overlap_ratio:.2f})")
            
            except Exception as e:
                st.error(f"❌ 网格搜索失败: {str(e)}")
    
    with tab2:
        st.header("📊 结果分析")
        
        if not st.session_state.experiment_results:
            st.warning("⚠️ 请先运行实验获取结果")
            return
        
        # 创建可视化器
        visualizer = ExperimentVisualizer(st.session_state.experiment_results)
        
        # 结果概览
        st.subheader("📈 实验结果概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        results_df = visualizer.df
        
        with col1:
            st.metric(
                "最高准确率",
                f"{results_df['retrieval_accuracy'].max():.3f}",
                f"+{(results_df['retrieval_accuracy'].max() - results_df['retrieval_accuracy'].min()):.3f}"
            )
        
        with col2:
            st.metric(
                "最高召回率",
                f"{results_df['retrieval_recall'].max():.3f}",
                f"+{(results_df['retrieval_recall'].max() - results_df['retrieval_recall'].min()):.3f}"
            )
        
        with col3:
            st.metric(
                "最快响应时间",
                f"{results_df['response_time'].min():.2f}ms",
                f"-{(results_df['response_time'].max() - results_df['response_time'].min()):.2f}ms"
            )
        
        with col4:
            st.metric(
                "最高F1分数",
                f"{results_df['f1_score'].max():.3f}",
                f"+{(results_df['f1_score'].max() - results_df['f1_score'].min()):.3f}"
            )
        
        st.markdown("---")
        
        # 可视化选项
        viz_type = st.selectbox(
            "选择可视化类型",
            ["热力图", "性能曲线", "3D表面图", "相关性矩阵", "帕累托前沿"]
        )
        
        if viz_type == "热力图":
            metric = st.selectbox(
                "选择指标",
                ["retrieval_accuracy", "retrieval_recall", "response_time", "f1_score"]
            )
            
            # 创建热力图
            pivot_table = results_df.pivot(
                index='overlap_ratio',
                columns='chunk_size',
                values=metric
            )
            
            fig = px.imshow(
                pivot_table,
                labels=dict(x="Chunk大小", y="重叠比例", color=metric),
                title=f"{metric} 热力图",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "性能曲线":
            # 创建性能曲线图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('准确率 vs Chunk大小', '召回率 vs 重叠比例', '响应时间分布', 'F1分数散点图')
            )
            
            # 准确率 vs Chunk大小
            for overlap in results_df['overlap_ratio'].unique():
                subset = results_df[results_df['overlap_ratio'] == overlap]
                fig.add_trace(
                    go.Scatter(
                        x=subset['chunk_size'],
                        y=subset['retrieval_accuracy'],
                        mode='lines+markers',
                        name=f'重叠={overlap:.2f}',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # 召回率 vs 重叠比例
            for chunk_size in results_df['chunk_size'].unique():
                subset = results_df[results_df['chunk_size'] == chunk_size]
                fig.add_trace(
                    go.Scatter(
                        x=subset['overlap_ratio'],
                        y=subset['retrieval_recall'],
                        mode='lines+markers',
                        name=f'大小={chunk_size}',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 响应时间分布
            fig.add_trace(
                go.Histogram(
                    x=results_df['response_time'],
                    nbinsx=20,
                    name='响应时间分布',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # F1分数散点图
            fig.add_trace(
                go.Scatter(
                    x=results_df['chunk_size'],
                    y=results_df['f1_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=results_df['overlap_ratio'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='F1分数',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="性能分析图表")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "3D表面图":
            metric = st.selectbox(
                "选择3D指标",
                ["retrieval_accuracy", "retrieval_recall", "f1_score"]
            )
            
            # 准备3D数据
            chunk_sizes_unique = sorted(results_df['chunk_size'].unique())
            overlap_ratios_unique = sorted(results_df['overlap_ratio'].unique())
            
            X, Y = np.meshgrid(chunk_sizes_unique, overlap_ratios_unique)
            Z = np.zeros_like(X)
            
            for i, overlap in enumerate(overlap_ratios_unique):
                for j, chunk_size in enumerate(chunk_sizes_unique):
                    value = results_df[
                        (results_df['chunk_size'] == chunk_size) &
                        (results_df['overlap_ratio'] == overlap)
                    ][metric]
                    Z[i, j] = value.iloc[0] if not value.empty else 0
            
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
            fig.update_layout(
                title=f'{metric} 3D表面图',
                scene=dict(
                    xaxis_title='Chunk大小',
                    yaxis_title='重叠比例',
                    zaxis_title=metric
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "相关性矩阵":
            # 计算相关性矩阵
            numeric_cols = ['chunk_size', 'overlap_ratio', 'retrieval_accuracy',
                           'retrieval_recall', 'response_time', 'f1_score']
            corr_matrix = results_df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="相关系数"),
                title="参数与性能指标相关性矩阵",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "帕累托前沿":
            # 帕累托前沿分析
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('准确率 vs 响应时间', '召回率 vs 存储开销')
            )
            
            # 准确率 vs 响应时间
            fig.add_trace(
                go.Scatter(
                    x=results_df['response_time'],
                    y=results_df['retrieval_accuracy'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=results_df['chunk_size'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f'Size: {size}, Overlap: {overlap:.2f}' 
                          for size, overlap in zip(results_df['chunk_size'], results_df['overlap_ratio'])],
                    name='准确率 vs 响应时间'
                ),
                row=1, col=1
            )
            
            # 召回率 vs 存储开销
            fig.add_trace(
                go.Scatter(
                    x=results_df['storage_overhead'],
                    y=results_df['retrieval_recall'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=results_df['overlap_ratio'],
                        colorscale='Plasma',
                        showscale=False
                    ),
                    text=[f'Size: {size}, Overlap: {overlap:.2f}' 
                          for size, overlap in zip(results_df['chunk_size'], results_df['overlap_ratio'])],
                    name='召回率 vs 存储开销',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=500, title_text="帕累托前沿分析")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🎯 参数优化建议")
        
        if not st.session_state.experiment_results:
            st.warning("⚠️ 请先运行实验获取结果")
            return
        
        results_df = pd.DataFrame([
            {
                'chunk_size': r.chunk_size,
                'overlap_ratio': r.overlap_ratio,
                'retrieval_accuracy': r.retrieval_accuracy,
                'retrieval_recall': r.retrieval_recall,
                'response_time': r.response_time,
                'storage_overhead': r.storage_overhead,
                'f1_score': 2 * r.retrieval_accuracy * r.retrieval_recall / 
                           (r.retrieval_accuracy + r.retrieval_recall) 
                           if (r.retrieval_accuracy + r.retrieval_recall) > 0 else 0
            }
            for r in st.session_state.experiment_results
        ])
        
        # 计算综合评分
        results_df['composite_score'] = (
            0.3 * results_df['retrieval_accuracy'] +
            0.3 * results_df['retrieval_recall'] +
            0.2 * (1 / (1 + results_df['response_time'] / 100)) +
            0.2 * (1 / results_df['storage_overhead'])
        )
        
        # 最佳配置推荐
        st.subheader("🏆 最佳配置推荐")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 按单一指标排序")
            
            # 最高准确率
            best_accuracy = results_df.loc[results_df['retrieval_accuracy'].idxmax()]
            st.info(f"""
            **最高准确率配置:**
            - Chunk大小: {int(best_accuracy['chunk_size'])}
            - 重叠比例: {best_accuracy['overlap_ratio']:.2f}
            - 准确率: {best_accuracy['retrieval_accuracy']:.3f}
            """)
            
            # 最高召回率
            best_recall = results_df.loc[results_df['retrieval_recall'].idxmax()]
            st.info(f"""
            **最高召回率配置:**
            - Chunk大小: {int(best_recall['chunk_size'])}
            - 重叠比例: {best_recall['overlap_ratio']:.2f}
            - 召回率: {best_recall['retrieval_recall']:.3f}
            """)
            
            # 最快响应
            fastest = results_df.loc[results_df['response_time'].idxmin()]
            st.info(f"""
            **最快响应配置:**
            - Chunk大小: {int(fastest['chunk_size'])}
            - 重叠比例: {fastest['overlap_ratio']:.2f}
            - 响应时间: {fastest['response_time']:.2f}ms
            """)
        
        with col2:
            st.markdown("### 🎯 综合最优配置")
            
            # 综合最佳
            best_overall = results_df.loc[results_df['composite_score'].idxmax()]
            
            st.success(f"""
            **综合最优配置:**
            - Chunk大小: {int(best_overall['chunk_size'])}
            - 重叠比例: {best_overall['overlap_ratio']:.2f}
            - 综合评分: {best_overall['composite_score']:.3f}
            
            **性能指标:**
            - 准确率: {best_overall['retrieval_accuracy']:.3f}
            - 召回率: {best_overall['retrieval_recall']:.3f}
            - F1分数: {best_overall['f1_score']:.3f}
            - 响应时间: {best_overall['response_time']:.2f}ms
            """)
            
            # 前5名配置
            st.markdown("### 📋 前5名配置")
            top5 = results_df.nlargest(5, 'composite_score')[[
                'chunk_size', 'overlap_ratio', 'composite_score',
                'retrieval_accuracy', 'retrieval_recall'
            ]]
            
            st.dataframe(
                top5.round(3),
                column_config={
                    'chunk_size': 'Chunk大小',
                    'overlap_ratio': '重叠比例',
                    'composite_score': '综合评分',
                    'retrieval_accuracy': '准确率',
                    'retrieval_recall': '召回率'
                }
            )
        
        st.markdown("---")
        
        # 参数影响分析
        st.subheader("📈 参数影响分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Chunk大小影响
            chunk_impact = results_df.groupby('chunk_size').agg({
                'retrieval_accuracy': 'mean',
                'retrieval_recall': 'mean',
                'response_time': 'mean'
            }).round(3)
            
            st.markdown("**Chunk大小对性能的影响:**")
            st.dataframe(chunk_impact)
        
        with col2:
            # 重叠比例影响
            overlap_impact = results_df.groupby('overlap_ratio').agg({
                'retrieval_accuracy': 'mean',
                'retrieval_recall': 'mean',
                'storage_overhead': 'mean'
            }).round(3)
            
            st.markdown("**重叠比例对性能的影响:**")
            st.dataframe(overlap_impact)
    
    with tab4:
        st.header("📋 实验报告")
        
        if not st.session_state.experiment_results:
            st.warning("⚠️ 请先运行实验获取结果")
            return
        
        # 生成实验报告
        visualizer = ExperimentVisualizer(st.session_state.experiment_results)
        report = visualizer.generate_summary_report()
        
        # 实验概要
        st.subheader("📊 实验概要")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("实验总数", report['experiment_summary']['total_experiments'])
        
        with col2:
            chunk_range = report['experiment_summary']['chunk_size_range']
            st.metric("Chunk大小范围", f"{chunk_range[0]} - {chunk_range[1]}")
        
        with col3:
            overlap_range = report['experiment_summary']['overlap_ratio_range']
            st.metric("重叠比例范围", f"{overlap_range[0]:.2f} - {overlap_range[1]:.2f}")
        
        # 性能统计
        st.subheader("📈 性能统计")
        
        stats_df = pd.DataFrame({
            '指标': ['准确率', '召回率', '响应时间(ms)'],
            '平均值': [
                f"{report['performance_statistics']['accuracy']['mean']:.3f}",
                f"{report['performance_statistics']['recall']['mean']:.3f}",
                f"{report['performance_statistics']['response_time']['mean']:.2f}"
            ],
            '标准差': [
                f"{report['performance_statistics']['accuracy']['std']:.3f}",
                f"{report['performance_statistics']['recall']['std']:.3f}",
                f"{report['performance_statistics']['response_time']['std']:.2f}"
            ],
            '最小值': [
                f"{report['performance_statistics']['accuracy']['min']:.3f}",
                f"{report['performance_statistics']['recall']['min']:.3f}",
                f"{report['performance_statistics']['response_time']['min']:.2f}"
            ],
            '最大值': [
                f"{report['performance_statistics']['accuracy']['max']:.3f}",
                f"{report['performance_statistics']['recall']['max']:.3f}",
                f"{report['performance_statistics']['response_time']['max']:.2f}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # 最佳配置汇总
        st.subheader("🏆 最佳配置汇总")
        
        best_configs = report['best_configurations']
        
        config_df = pd.DataFrame({
            '优化目标': ['最高准确率', '最高召回率', '最高F1分数', '最快响应'],
            'Chunk大小': [
                best_configs['highest_accuracy']['chunk_size'],
                best_configs['highest_recall']['chunk_size'],
                best_configs['highest_f1']['chunk_size'],
                best_configs['fastest_response']['chunk_size']
            ],
            '重叠比例': [
                f"{best_configs['highest_accuracy']['overlap_ratio']:.2f}",
                f"{best_configs['highest_recall']['overlap_ratio']:.2f}",
                f"{best_configs['highest_f1']['overlap_ratio']:.2f}",
                f"{best_configs['fastest_response']['overlap_ratio']:.2f}"
            ],
            '性能值': [
                f"{best_configs['highest_accuracy']['accuracy']:.3f}",
                f"{best_configs['highest_recall']['recall']:.3f}",
                f"{best_configs['highest_f1']['f1_score']:.3f}",
                f"{best_configs['fastest_response']['response_time']:.2f}ms"
            ]
        })
        
        st.dataframe(config_df, use_container_width=True)
        
        # 导出功能
        st.subheader("💾 导出结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 导出实验数据
            results_df = pd.DataFrame([
                {
                    'chunk_size': r.chunk_size,
                    'overlap_ratio': r.overlap_ratio,
                    'retrieval_accuracy': r.retrieval_accuracy,
                    'retrieval_recall': r.retrieval_recall,
                    'response_time': r.response_time,
                    'storage_overhead': r.storage_overhead
                }
                for r in st.session_state.experiment_results
            ])
            
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 下载实验数据 (CSV)",
                data=csv_data,
                file_name=f"chunk_experiment_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        with col2:
            # 导出实验报告
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="📋 下载实验报告 (JSON)",
                data=report_json,
                file_name=f"chunk_experiment_report_{int(time.time())}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()