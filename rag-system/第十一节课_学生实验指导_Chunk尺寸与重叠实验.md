# 第十一节课：学生实验指导 - Chunk尺寸与重叠实验

## 实验目标

通过本实验，学生将：
1. 掌握Chunk分块参数对RAG系统性能的影响
2. 学会设计和执行分块参数优化实验
3. 理解如何评估分块质量和检索效果
4. 实现自动化的参数调优工具

## 实验环境准备

### 依赖安装
```bash
pip install streamlit plotly pandas numpy scikit-learn
```

### 项目结构
```
src/
├── chunking/
│   ├── chunk_optimizer.py      # 参数优化器
│   ├── experiment_runner.py    # 实验执行器
│   └── quality_evaluator.py    # 质量评估器
├── experiments/
│   ├── chunk_optimization/
│   │   ├── interactive_tuner.py    # 交互式调优工具
│   │   ├── visualizer.py          # 结果可视化
│   │   └── test_data/             # 测试数据
│   └── results/                   # 实验结果
```

## 核心实验组件

### 1. ChunkOptimizer 类实现

创建 `src/chunking/chunk_optimizer.py`：

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from ..chunking import ChunkManager, ChunkingConfig
from ..embedding import EmbeddingService

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: Dict[str, any]
    best_score: float
    all_results: List[Dict]
    optimization_history: List[Dict]

class ChunkOptimizer:
    """Chunk参数优化器"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.chunk_manager = ChunkManager()
        self.evaluation_cache = {}
    
    def grid_search(self, 
                   documents: List[str],
                   test_queries: List[str],
                   param_grid: Dict[str, List],
                   scoring_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        网格搜索最优分块参数
        
        Args:
            documents: 测试文档列表
            test_queries: 测试查询列表
            param_grid: 参数网格，如 {'chunk_size': [300, 500, 800], 'overlap_ratio': [0.1, 0.2, 0.3]}
            scoring_weights: 评分权重，如 {'precision': 0.4, 'recall': 0.3, 'coherence': 0.3}
        
        Returns:
            OptimizationResult: 优化结果
        """
        if scoring_weights is None:
            scoring_weights = {
                'precision': 0.3,
                'recall': 0.3, 
                'coherence': 0.2,
                'efficiency': 0.2
            }
        
        results = []
        best_score = -1
        best_params = None
        
        # 生成所有参数组合
        param_combinations = self._generate_param_combinations(param_grid)
        
        print(f"开始网格搜索，共 {len(param_combinations)} 种参数组合...")
        
        for i, params in enumerate(param_combinations):
            print(f"\n测试参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            # 评估当前参数组合
            metrics = self._evaluate_params(
                documents, test_queries, params
            )
            
            # 计算综合得分
            composite_score = sum(
                metrics.get(metric, 0) * weight 
                for metric, weight in scoring_weights.items()
            )
            
            result = {
                'params': params.copy(),
                'metrics': metrics,
                'composite_score': composite_score
            }
            results.append(result)
            
            # 更新最佳结果
            if composite_score > best_score:
                best_score = composite_score
                best_params = params.copy()
            
            print(f"综合得分: {composite_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_history=results
        )
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """生成所有参数组合"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(self, 
                        documents: List[str], 
                        test_queries: List[str], 
                        params: Dict) -> Dict[str, float]:
        """评估特定参数组合的性能"""
        
        # 创建分块配置
        config = ChunkingConfig(
            chunk_size=params.get('chunk_size', 500),
            chunk_overlap=int(params.get('chunk_size', 500) * params.get('overlap_ratio', 0.2)),
            min_chunk_size=params.get('min_chunk_size', 100)
        )
        
        # 对所有文档进行分块
        all_chunks = []
        chunk_doc_mapping = []  # 记录每个chunk属于哪个文档
        
        for doc_idx, document in enumerate(documents):
            chunks = self.chunk_manager.chunk_text(
                document, 
                chunker_type=params.get('chunker_type', 'sentence'),
                config=config
            )
            
            chunk_texts = [chunk.content for chunk in chunks]
            all_chunks.extend(chunk_texts)
            chunk_doc_mapping.extend([doc_idx] * len(chunk_texts))
        
        if not all_chunks:
            return {'precision': 0, 'recall': 0, 'coherence': 0, 'efficiency': 0}
        
        # 计算各项指标
        metrics = {}
        
        # 1. 检索精度和召回率
        precision, recall = self._calculate_retrieval_metrics(
            all_chunks, chunk_doc_mapping, test_queries, documents
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # 2. 语义连贯性
        coherence = self._calculate_semantic_coherence(all_chunks)
        metrics['coherence'] = coherence
        
        # 3. 计算效率
        efficiency = self._calculate_efficiency(all_chunks, params)
        metrics['efficiency'] = efficiency
        
        # 4. F1分数
        if precision + recall > 0:
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1_score'] = 0
        
        return metrics
    
    def _calculate_retrieval_metrics(self, 
                                   chunks: List[str], 
                                   chunk_doc_mapping: List[int],
                                   queries: List[str], 
                                   documents: List[str]) -> Tuple[float, float]:
        """计算检索精度和召回率"""
        if not chunks or not queries:
            return 0.0, 0.0
        
        try:
            # 获取chunk和query的嵌入
            chunk_embeddings = self.embedding_service.encode_batch(chunks)
            query_embeddings = self.embedding_service.encode_batch(queries)
            
            total_precision = 0
            total_recall = 0
            valid_queries = 0
            
            for query_idx, query_emb in enumerate(query_embeddings):
                # 计算相似度
                similarities = cosine_similarity([query_emb], chunk_embeddings)[0]
                
                # 获取top-k最相似的chunks
                k = min(5, len(chunks))
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                # 计算精度：top-k中有多少来自相关文档
                # 这里简化处理，假设查询与所有文档都相关
                relevant_chunks = len(set(chunk_doc_mapping[i] for i in top_k_indices))
                precision = relevant_chunks / len(set(chunk_doc_mapping))
                
                # 计算召回率：相关文档的chunks有多少被检索到
                total_relevant = len(set(chunk_doc_mapping))
                recall = relevant_chunks / total_relevant if total_relevant > 0 else 0
                
                total_precision += precision
                total_recall += recall
                valid_queries += 1
            
            avg_precision = total_precision / valid_queries if valid_queries > 0 else 0
            avg_recall = total_recall / valid_queries if valid_queries > 0 else 0
            
            return avg_precision, avg_recall
            
        except Exception as e:
            print(f"计算检索指标时出错: {e}")
            return 0.0, 0.0
    
    def _calculate_semantic_coherence(self, chunks: List[str]) -> float:
        """计算语义连贯性"""
        if len(chunks) < 2:
            return 1.0
        
        try:
            # 计算相邻chunks的语义相似度
            embeddings = self.embedding_service.encode_batch(chunks)
            
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append(similarity)
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            print(f"计算语义连贯性时出错: {e}")
            return 0.0
    
    def _calculate_efficiency(self, chunks: List[str], params: Dict) -> float:
        """计算效率指标（基于chunk数量和大小）"""
        if not chunks:
            return 0.0
        
        # 效率指标：考虑chunk数量和平均长度
        avg_chunk_length = np.mean([len(chunk) for chunk in chunks])
        chunk_count = len(chunks)
        
        # 理想的chunk数量和长度（可调整）
        ideal_chunk_length = params.get('chunk_size', 500)
        ideal_chunk_count_ratio = 1.0  # 假设理想比例
        
        # 长度效率：越接近理想长度越好
        length_efficiency = 1 - abs(avg_chunk_length - ideal_chunk_length) / ideal_chunk_length
        length_efficiency = max(0, length_efficiency)
        
        # 数量效率：适中的chunk数量
        count_efficiency = 1 / (1 + chunk_count * 0.001)  # 数量越多效率越低
        
        return (length_efficiency + count_efficiency) / 2
    
    def bayesian_optimization(self, 
                            documents: List[str],
                            test_queries: List[str],
                            param_bounds: Dict[str, Tuple[float, float]],
                            n_iterations: int = 20) -> OptimizationResult:
        """
        贝叶斯优化（简化版本）
        
        Args:
            documents: 测试文档
            test_queries: 测试查询
            param_bounds: 参数边界，如 {'chunk_size': (200, 1500), 'overlap_ratio': (0.1, 0.5)}
            n_iterations: 迭代次数
        
        Returns:
            OptimizationResult: 优化结果
        """
        # 这里实现简化的随机搜索作为贝叶斯优化的替代
        results = []
        best_score = -1
        best_params = None
        
        print(f"开始贝叶斯优化，迭代 {n_iterations} 次...")
        
        for i in range(n_iterations):
            # 随机采样参数
            params = {}
            for param_name, (min_val, max_val) in param_bounds.items():
                if param_name == 'chunk_size':
                    params[param_name] = int(np.random.uniform(min_val, max_val))
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            print(f"\n迭代 {i+1}/{n_iterations}: {params}")
            
            # 评估参数
            metrics = self._evaluate_params(documents, test_queries, params)
            composite_score = np.mean(list(metrics.values()))
            
            result = {
                'params': params.copy(),
                'metrics': metrics,
                'composite_score': composite_score
            }
            results.append(result)
            
            if composite_score > best_score:
                best_score = composite_score
                best_params = params.copy()
            
            print(f"综合得分: {composite_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            optimization_history=results
        )
```

### 2. MockRAGSystem 类实现

创建 `src/experiments/chunk_optimization/mock_rag_system.py`：

```python
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from ...chunking import ChunkManager, ChunkingConfig
from ...embedding import EmbeddingService

class MockRAGSystem:
    """模拟RAG系统，用于测试不同分块策略"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.chunk_manager = ChunkManager()
        self.knowledge_base = []
        self.chunk_embeddings = []
    
    def build_knowledge_base(self, 
                           documents: List[str], 
                           chunking_config: ChunkingConfig,
                           chunker_type: str = 'sentence') -> Dict[str, any]:
        """
        构建知识库
        
        Args:
            documents: 文档列表
            chunking_config: 分块配置
            chunker_type: 分块器类型
        
        Returns:
            Dict: 构建统计信息
        """
        print(f"构建知识库，使用 {chunker_type} 分块器...")
        
        self.knowledge_base = []
        chunk_texts = []
        
        for doc_idx, document in enumerate(documents):
            # 对文档进行分块
            chunks = self.chunk_manager.chunk_text(
                document, 
                chunker_type=chunker_type,
                config=chunking_config
            )
            
            for chunk in chunks:
                chunk_info = {
                    'content': chunk.content,
                    'doc_id': doc_idx,
                    'chunk_id': chunk.metadata.chunk_id,
                    'chunk_index': chunk.metadata.chunk_index,
                    'metadata': chunk.metadata
                }
                self.knowledge_base.append(chunk_info)
                chunk_texts.append(chunk.content)
        
        # 计算embeddings
        if chunk_texts:
            print(f"计算 {len(chunk_texts)} 个chunks的embeddings...")
            self.chunk_embeddings = self.embedding_service.encode_batch(chunk_texts)
        else:
            self.chunk_embeddings = []
        
        stats = {
            'total_documents': len(documents),
            'total_chunks': len(self.knowledge_base),
            'avg_chunks_per_doc': len(self.knowledge_base) / len(documents) if documents else 0,
            'avg_chunk_length': np.mean([len(chunk['content']) for chunk in self.knowledge_base]) if self.knowledge_base else 0
        }
        
        print(f"知识库构建完成: {stats}")
        return stats
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关chunks
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果
        
        Returns:
            List[Dict]: 检索结果
        """
        if not self.knowledge_base or not self.chunk_embeddings:
            return []
        
        # 计算查询embedding
        query_embedding = self.embedding_service.encode([query])[0]
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        # 获取top-k结果
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'chunk': self.knowledge_base[idx],
                'similarity': float(similarities[idx]),
                'rank': len(results) + 1
            }
            results.append(result)
        
        return results
    
    def evaluate_retrieval(self, 
                         test_queries: List[str], 
                         ground_truth: List[List[int]] = None) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            test_queries: 测试查询列表
            ground_truth: 每个查询的相关文档ID列表（可选）
        
        Returns:
            Dict: 评估指标
        """
        if not test_queries:
            return {}
        
        metrics = {
            'avg_similarity': 0,
            'coverage': 0,
            'diversity': 0
        }
        
        total_similarity = 0
        covered_docs = set()
        all_similarities = []
        
        for query in test_queries:
            results = self.retrieve(query, top_k=5)
            
            if results:
                # 平均相似度
                query_similarities = [r['similarity'] for r in results]
                total_similarity += np.mean(query_similarities)
                all_similarities.extend(query_similarities)
                
                # 文档覆盖度
                for result in results:
                    covered_docs.add(result['chunk']['doc_id'])
        
        # 计算指标
        if test_queries:
            metrics['avg_similarity'] = total_similarity / len(test_queries)
        
        total_docs = len(set(chunk['doc_id'] for chunk in self.knowledge_base))
        if total_docs > 0:
            metrics['coverage'] = len(covered_docs) / total_docs
        
        if all_similarities:
            metrics['diversity'] = np.std(all_similarities)  # 相似度的标准差作为多样性指标
        
        return metrics
    
    def analyze_chunk_distribution(self) -> Dict[str, any]:
        """
        分析chunk分布情况
        
        Returns:
            Dict: 分布统计信息
        """
        if not self.knowledge_base:
            return {}
        
        chunk_lengths = [len(chunk['content']) for chunk in self.knowledge_base]
        doc_chunk_counts = {}
        
        for chunk in self.knowledge_base:
            doc_id = chunk['doc_id']
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
        
        analysis = {
            'chunk_count': len(self.knowledge_base),
            'length_stats': {
                'mean': np.mean(chunk_lengths),
                'std': np.std(chunk_lengths),
                'min': np.min(chunk_lengths),
                'max': np.max(chunk_lengths),
                'median': np.median(chunk_lengths)
            },
            'chunks_per_doc': {
                'mean': np.mean(list(doc_chunk_counts.values())),
                'std': np.std(list(doc_chunk_counts.values())),
                'min': np.min(list(doc_chunk_counts.values())),
                'max': np.max(list(doc_chunk_counts.values()))
            }
        }
        
        return analysis
```

### 3. 交互式调优工具

创建 `src/experiments/chunk_optimization/interactive_tuner.py`：

```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

from ...chunking.chunk_optimizer import ChunkOptimizer
from .mock_rag_system import MockRAGSystem
from ...embedding import EmbeddingService
from ...chunking import ChunkingConfig

class InteractiveChunkTuner:
    """交互式Chunk参数调优工具"""
    
    def __init__(self):
        self.embedding_service = None
        self.optimizer = None
        self.mock_rag = None
        self.test_documents = []
        self.test_queries = []
    
    def initialize_services(self):
        """初始化服务"""
        if 'embedding_service' not in st.session_state:
            st.session_state.embedding_service = EmbeddingService()
        
        self.embedding_service = st.session_state.embedding_service
        self.optimizer = ChunkOptimizer(self.embedding_service)
        self.mock_rag = MockRAGSystem(self.embedding_service)
    
    def run_app(self):
        """运行Streamlit应用"""
        st.set_page_config(
            page_title="Chunk参数调优工具",
            page_icon="🔧",
            layout="wide"
        )
        
        st.title("🔧 Chunk参数调优工具")
        st.markdown("---")
        
        # 初始化服务
        self.initialize_services()
        
        # 侧边栏配置
        self._render_sidebar()
        
        # 主界面
        tab1, tab2, tab3, tab4 = st.tabs(["📊 参数实验", "🔍 实时测试", "📈 结果分析", "💾 历史记录"])
        
        with tab1:
            self._render_parameter_experiment()
        
        with tab2:
            self._render_real_time_test()
        
        with tab3:
            self._render_result_analysis()
        
        with tab4:
            self._render_history()
    
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.header("🛠️ 配置")
        
        # 测试数据配置
        st.sidebar.subheader("测试数据")
        
        # 文档输入
        doc_input_method = st.sidebar.radio(
            "文档输入方式",
            ["手动输入", "示例数据", "文件上传"]
        )
        
        if doc_input_method == "手动输入":
            doc_text = st.sidebar.text_area(
                "输入测试文档（每行一个文档）",
                height=150,
                placeholder="输入您的测试文档..."
            )
            if doc_text:
                self.test_documents = [doc.strip() for doc in doc_text.split('\n') if doc.strip()]
        
        elif doc_input_method == "示例数据":
            self.test_documents = self._get_sample_documents()
            st.sidebar.success(f"已加载 {len(self.test_documents)} 个示例文档")
        
        # 查询输入
        query_text = st.sidebar.text_area(
            "测试查询（每行一个查询）",
            height=100,
            placeholder="输入您的测试查询..."
        )
        if query_text:
            self.test_queries = [q.strip() for q in query_text.split('\n') if q.strip()]
        
        # 显示数据统计
        if self.test_documents or self.test_queries:
            st.sidebar.markdown("**数据统计:**")
            st.sidebar.write(f"📄 文档数量: {len(self.test_documents)}")
            st.sidebar.write(f"❓ 查询数量: {len(self.test_queries)}")
    
    def _render_parameter_experiment(self):
        """渲染参数实验界面"""
        st.header("📊 参数网格搜索实验")
        
        if not self.test_documents or not self.test_queries:
            st.warning("请先在侧边栏配置测试数据和查询")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("参数配置")
            
            # Chunk大小范围
            chunk_sizes = st.multiselect(
                "Chunk大小",
                [200, 300, 500, 800, 1000, 1200, 1500],
                default=[300, 500, 800]
            )
            
            # 重叠比例
            overlap_ratios = st.multiselect(
                "重叠比例",
                [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
                default=[0.1, 0.2, 0.3]
            )
            
            # 分块器类型
            chunker_type = st.selectbox(
                "分块器类型",
                ["sentence", "semantic", "structure"],
                index=0
            )
        
        with col2:
            st.subheader("评估权重")
            
            precision_weight = st.slider("精确度权重", 0.0, 1.0, 0.3, 0.1)
            recall_weight = st.slider("召回率权重", 0.0, 1.0, 0.3, 0.1)
            coherence_weight = st.slider("连贯性权重", 0.0, 1.0, 0.2, 0.1)
            efficiency_weight = st.slider("效率权重", 0.0, 1.0, 0.2, 0.1)
            
            # 归一化权重
            total_weight = precision_weight + recall_weight + coherence_weight + efficiency_weight
            if total_weight > 0:
                weights = {
                    'precision': precision_weight / total_weight,
                    'recall': recall_weight / total_weight,
                    'coherence': coherence_weight / total_weight,
                    'efficiency': efficiency_weight / total_weight
                }
            else:
                weights = {'precision': 0.25, 'recall': 0.25, 'coherence': 0.25, 'efficiency': 0.25}
        
        # 开始实验按钮
        if st.button("🚀 开始参数优化实验", type="primary"):
            if chunk_sizes and overlap_ratios:
                with st.spinner("正在进行参数优化实验..."):
                    param_grid = {
                        'chunk_size': chunk_sizes,
                        'overlap_ratio': overlap_ratios,
                        'chunker_type': [chunker_type]
                    }
                    
                    result = self.optimizer.grid_search(
                        self.test_documents,
                        self.test_queries,
                        param_grid,
                        weights
                    )
                    
                    # 保存结果到session state
                    st.session_state.experiment_result = result
                    
                    st.success("实验完成！")
                    
                    # 显示最佳参数
                    st.subheader("🏆 最佳参数")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("最佳Chunk大小", result.best_params['chunk_size'])
                    with col2:
                        st.metric("最佳重叠比例", f"{result.best_params['overlap_ratio']:.2f}")
                    with col3:
                        st.metric("最佳综合得分", f"{result.best_score:.4f}")
                    
                    # 显示详细结果
                    self._display_experiment_results(result)
            else:
                st.error("请至少选择一个chunk大小和重叠比例")
    
    def _render_real_time_test(self):
        """渲染实时测试界面"""
        st.header("🔍 实时参数测试")
        
        if not self.test_documents:
            st.warning("请先在侧边栏配置测试文档")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("参数设置")
            
            chunk_size = st.slider("Chunk大小", 100, 2000, 500, 50)
            overlap_ratio = st.slider("重叠比例", 0.0, 0.5, 0.2, 0.05)
            chunker_type = st.selectbox(
                "分块器类型",
                ["sentence", "semantic", "structure"]
            )
            
            test_query = st.text_input(
                "测试查询",
                placeholder="输入查询来测试检索效果..."
            )
        
        with col2:
            st.subheader("实时结果")
            
            if st.button("🔄 更新分块结果"):
                config = ChunkingConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=int(chunk_size * overlap_ratio)
                )
                
                # 构建知识库
                stats = self.mock_rag.build_knowledge_base(
                    self.test_documents, config, chunker_type
                )
                
                # 显示统计信息
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("总Chunk数", stats['total_chunks'])
                with col_b:
                    st.metric("平均每文档Chunk数", f"{stats['avg_chunks_per_doc']:.1f}")
                with col_c:
                    st.metric("平均Chunk长度", f"{stats['avg_chunk_length']:.0f}")
                
                # 显示chunk分布
                analysis = self.mock_rag.analyze_chunk_distribution()
                if analysis:
                    st.subheader("Chunk长度分布")
                    
                    chunk_lengths = [len(chunk['content']) for chunk in self.mock_rag.knowledge_base]
                    fig = px.histogram(
                        x=chunk_lengths,
                        nbins=20,
                        title="Chunk长度分布",
                        labels={'x': 'Chunk长度', 'y': '频次'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # 测试查询
            if test_query and hasattr(self.mock_rag, 'knowledge_base') and self.mock_rag.knowledge_base:
                st.subheader(f"查询结果: \"{test_query}\"")
                
                results = self.mock_rag.retrieve(test_query, top_k=3)
                
                for i, result in enumerate(results):
                    with st.expander(f"结果 {i+1} (相似度: {result['similarity']:.4f})"):
                        st.write(result['chunk']['content'])
                        st.caption(f"文档ID: {