#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson19 RAG系统评估实现模板
解决评估指标、基准测试和A/B测试缺失问题

功能特性：
1. 全面的RAG评估指标体系
2. 自动化基准测试框架
3. A/B测试和对比分析
4. 评估数据集管理
5. 评估报告生成和可视化
"""

import json
import time
import uuid
import random
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, mannwhitneyu
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import openai
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import yaml

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """评估指标枚举"""
    # 检索质量指标
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MAP = "mean_average_precision"
    NDCG = "normalized_dcg"
    MRR = "mean_reciprocal_rank"
    
    # 生成质量指标
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    BERT_SCORE = "bert_score"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    
    # 综合指标
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    
    # 性能指标
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"

class TestType(Enum):
    """测试类型枚举"""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    BENCHMARK_TEST = "benchmark_test"
    AB_TEST = "ab_test"
    STRESS_TEST = "stress_test"

class EvaluationStatus(Enum):
    """评估状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class EvaluationQuery:
    """评估查询"""
    id: str
    query: str
    expected_documents: List[str] = field(default_factory=list)
    expected_answer: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """检索结果"""
    query_id: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    retrieval_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResult:
    """生成结果"""
    query_id: str
    answer: str
    generation_time: float
    source_documents: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """评估结果"""
    query_id: str
    metric: EvaluationMetric
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_id: str
    test_name: str
    system_version: str
    timestamp: datetime
    metrics: Dict[str, float]
    query_results: List[EvaluationResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    status: EvaluationStatus = EvaluationStatus.COMPLETED

@dataclass
class ABTestResult:
    """A/B测试结果"""
    test_id: str
    test_name: str
    variant_a: str
    variant_b: str
    timestamp: datetime
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    statistical_significance: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    confidence_level: float = 0.95

class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self):
        self.metrics = {
            EvaluationMetric.PRECISION_AT_K: self._precision_at_k,
            EvaluationMetric.RECALL_AT_K: self._recall_at_k,
            EvaluationMetric.MAP: self._mean_average_precision,
            EvaluationMetric.NDCG: self._normalized_dcg,
            EvaluationMetric.MRR: self._mean_reciprocal_rank
        }
    
    def evaluate(self, query: EvaluationQuery, result: RetrievalResult, 
                k: int = 10) -> List[EvaluationResult]:
        """评估检索结果"""
        evaluation_results = []
        
        # 获取检索到的文档ID
        retrieved_docs = [doc.get('id', doc.get('document_id', '')) 
                         for doc in result.documents[:k]]
        expected_docs = query.expected_documents
        
        if not expected_docs:
            logger.warning(f"查询 {query.id} 没有期望的文档列表")
            return evaluation_results
        
        # 计算各种指标
        for metric, func in self.metrics.items():
            try:
                if metric in [EvaluationMetric.PRECISION_AT_K, EvaluationMetric.RECALL_AT_K]:
                    value = func(retrieved_docs, expected_docs, k)
                elif metric == EvaluationMetric.MAP:
                    value = func([retrieved_docs], [expected_docs])
                elif metric == EvaluationMetric.NDCG:
                    # 为NDCG计算相关性分数
                    relevance_scores = [1 if doc in expected_docs else 0 for doc in retrieved_docs]
                    value = self._ndcg_at_k(relevance_scores, k)
                elif metric == EvaluationMetric.MRR:
                    value = func(retrieved_docs, expected_docs)
                else:
                    continue
                
                evaluation_results.append(EvaluationResult(
                    query_id=query.id,
                    metric=metric,
                    value=value,
                    details={'k': k, 'retrieved_count': len(retrieved_docs)}
                ))
                
            except Exception as e:
                logger.error(f"计算指标 {metric} 失败: {e}")
        
        return evaluation_results
    
    def _precision_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """计算Precision@K"""
        if not retrieved or k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(expected))
        return relevant_retrieved / len(retrieved_k)
    
    def _recall_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """计算Recall@K"""
        if not expected:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(expected))
        return relevant_retrieved / len(expected)
    
    def _mean_average_precision(self, retrieved_lists: List[List[str]], 
                               expected_lists: List[List[str]]) -> float:
        """计算平均精度均值(MAP)"""
        if len(retrieved_lists) != len(expected_lists):
            raise ValueError("检索列表和期望列表长度不匹配")
        
        average_precisions = []
        
        for retrieved, expected in zip(retrieved_lists, expected_lists):
            if not expected:
                continue
            
            precisions = []
            relevant_count = 0
            
            for i, doc in enumerate(retrieved):
                if doc in expected:
                    relevant_count += 1
                    precision = relevant_count / (i + 1)
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(sum(precisions) / len(expected))
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def _normalized_dcg(self, relevance_scores: List[float], k: int) -> float:
        """计算归一化折扣累积增益(NDCG@K)"""
        def dcg(scores: List[float]) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        if not relevance_scores or k == 0:
            return 0.0
        
        # 计算DCG@K
        dcg_k = dcg(relevance_scores[:k])
        
        # 计算理想DCG@K
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg_k = dcg(ideal_scores[:k])
        
        return dcg_k / idcg_k if idcg_k > 0 else 0.0
    
    def _ndcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """计算NDCG@K"""
        return self._normalized_dcg(relevance_scores, k)
    
    def _mean_reciprocal_rank(self, retrieved: List[str], expected: List[str]) -> float:
        """计算平均倒数排名(MRR)"""
        for i, doc in enumerate(retrieved):
            if doc in expected:
                return 1.0 / (i + 1)
        return 0.0

class GenerationEvaluator:
    """生成评估器"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def evaluate(self, query: EvaluationQuery, result: GenerationResult) -> List[EvaluationResult]:
        """评估生成结果"""
        evaluation_results = []
        
        if not query.expected_answer:
            logger.warning(f"查询 {query.id} 没有期望答案")
            return evaluation_results
        
        generated_answer = result.answer
        expected_answer = query.expected_answer
        
        # BLEU分数
        try:
            bleu_score = self._calculate_bleu(generated_answer, expected_answer)
            evaluation_results.append(EvaluationResult(
                query_id=query.id,
                metric=EvaluationMetric.BLEU,
                value=bleu_score
            ))
        except Exception as e:
            logger.error(f"计算BLEU分数失败: {e}")
        
        # ROUGE分数
        try:
            rouge_scores = self._calculate_rouge(generated_answer, expected_answer)
            evaluation_results.append(EvaluationResult(
                query_id=query.id,
                metric=EvaluationMetric.ROUGE_L,
                value=rouge_scores['rougeL'].fmeasure,
                details={
                    'rouge1': rouge_scores['rouge1'].fmeasure,
                    'rouge2': rouge_scores['rouge2'].fmeasure,
                    'rougeL': rouge_scores['rougeL'].fmeasure
                }
            ))
        except Exception as e:
            logger.error(f"计算ROUGE分数失败: {e}")
        
        # 语义相似度
        try:
            semantic_sim = self._calculate_semantic_similarity(generated_answer, expected_answer)
            evaluation_results.append(EvaluationResult(
                query_id=query.id,
                metric=EvaluationMetric.SEMANTIC_SIMILARITY,
                value=semantic_sim
            ))
        except Exception as e:
            logger.error(f"计算语义相似度失败: {e}")
        
        # BERTScore
        try:
            bert_scores = self._calculate_bert_score(generated_answer, expected_answer)
            evaluation_results.append(EvaluationResult(
                query_id=query.id,
                metric=EvaluationMetric.BERT_SCORE,
                value=bert_scores['f1'],
                details={
                    'precision': bert_scores['precision'],
                    'recall': bert_scores['recall'],
                    'f1': bert_scores['f1']
                }
            ))
        except Exception as e:
            logger.error(f"计算BERTScore失败: {e}")
        
        return evaluation_results
    
    def _calculate_bleu(self, generated: str, reference: str) -> float:
        """计算BLEU分数"""
        # 分词
        generated_tokens = nltk.word_tokenize(generated.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())
        
        # 计算BLEU分数
        smoothing = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    
    def _calculate_rouge(self, generated: str, reference: str) -> Dict:
        """计算ROUGE分数"""
        return self.rouge_scorer.score(reference, generated)
    
    def _calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """计算语义相似度"""
        # 获取句子嵌入
        embeddings = self.sentence_model.encode([generated, reference])
        
        # 计算余弦相似度
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def _calculate_bert_score(self, generated: str, reference: str) -> Dict[str, float]:
        """计算BERTScore"""
        P, R, F1 = bert_score([generated], [reference], lang='en', verbose=False)
        
        return {
            'precision': float(P[0]),
            'recall': float(R[0]),
            'f1': float(F1[0])
        }

class LLMEvaluator:
    """基于LLM的评估器"""
    
    def __init__(self, api_key: str = None):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        
    def evaluate_faithfulness(self, query: str, answer: str, context: str) -> float:
        """评估答案的忠实性"""
        if not self.client:
            logger.warning("未配置OpenAI API，跳过忠实性评估")
            return 0.0
        
        prompt = f"""
        请评估以下答案相对于给定上下文的忠实性。忠实性是指答案中的信息是否完全基于上下文，没有添加上下文中不存在的信息。
        
        上下文：{context}
        
        问题：{query}
        
        答案：{answer}
        
        请给出0-1之间的忠实性分数，其中：
        - 1.0：答案完全忠实于上下文
        - 0.8：答案大部分忠实，有少量推理
        - 0.6：答案基本忠实，有一些额外信息
        - 0.4：答案部分忠实，有明显的额外信息
        - 0.2：答案很少基于上下文
        - 0.0：答案与上下文无关或完全错误
        
        只返回数字分数，不要解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            logger.error(f"LLM忠实性评估失败: {e}")
            return 0.0
    
    def evaluate_relevance(self, query: str, answer: str) -> float:
        """评估答案的相关性"""
        if not self.client:
            logger.warning("未配置OpenAI API，跳过相关性评估")
            return 0.0
        
        prompt = f"""
        请评估以下答案对问题的相关性。相关性是指答案是否直接回答了问题。
        
        问题：{query}
        
        答案：{answer}
        
        请给出0-1之间的相关性分数，其中：
        - 1.0：答案完全相关，直接回答问题
        - 0.8：答案高度相关，基本回答问题
        - 0.6：答案相关，部分回答问题
        - 0.4：答案部分相关，偏离主题
        - 0.2：答案很少相关
        - 0.0：答案完全不相关
        
        只返回数字分数，不要解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            logger.error(f"LLM相关性评估失败: {e}")
            return 0.0
    
    def evaluate_coherence(self, answer: str) -> float:
        """评估答案的连贯性"""
        if not self.client:
            logger.warning("未配置OpenAI API，跳过连贯性评估")
            return 0.0
        
        prompt = f"""
        请评估以下文本的连贯性。连贯性是指文本的逻辑结构和语言流畅性。
        
        文本：{answer}
        
        请给出0-1之间的连贯性分数，其中：
        - 1.0：文本非常连贯，逻辑清晰，语言流畅
        - 0.8：文本连贯，逻辑较清晰
        - 0.6：文本基本连贯，有少量逻辑跳跃
        - 0.4：文本部分连贯，逻辑不够清晰
        - 0.2：文本很少连贯，逻辑混乱
        - 0.0：文本完全不连贯
        
        只返回数字分数，不要解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
            
        except Exception as e:
            logger.error(f"LLM连贯性评估失败: {e}")
            return 0.0

class DatasetManager:
    """评估数据集管理器"""
    
    def __init__(self, dataset_dir: str = "evaluation_datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        self.datasets: Dict[str, List[EvaluationQuery]] = {}
    
    def load_dataset(self, name: str, file_path: str = None) -> List[EvaluationQuery]:
        """加载评估数据集"""
        if file_path is None:
            file_path = self.dataset_dir / f"{name}.json"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            queries = []
            for item in data:
                query = EvaluationQuery(
                    id=item.get('id', str(uuid.uuid4())),
                    query=item['query'],
                    expected_documents=item.get('expected_documents', []),
                    expected_answer=item.get('expected_answer'),
                    category=item.get('category'),
                    difficulty=item.get('difficulty'),
                    metadata=item.get('metadata', {})
                )
                queries.append(query)
            
            self.datasets[name] = queries
            logger.info(f"加载数据集 {name}，包含 {len(queries)} 个查询")
            return queries
            
        except Exception as e:
            logger.error(f"加载数据集 {name} 失败: {e}")
            return []
    
    def save_dataset(self, name: str, queries: List[EvaluationQuery], file_path: str = None):
        """保存评估数据集"""
        if file_path is None:
            file_path = self.dataset_dir / f"{name}.json"
        
        try:
            data = [asdict(query) for query in queries]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self.datasets[name] = queries
            logger.info(f"保存数据集 {name}，包含 {len(queries)} 个查询")
            
        except Exception as e:
            logger.error(f"保存数据集 {name} 失败: {e}")
    
    def create_sample_dataset(self, name: str = "sample") -> List[EvaluationQuery]:
        """创建示例数据集"""
        sample_queries = [
            EvaluationQuery(
                id="q1",
                query="什么是机器学习？",
                expected_documents=["doc1", "doc2"],
                expected_answer="机器学习是一种人工智能技术，通过算法让计算机从数据中学习模式。",
                category="基础概念",
                difficulty="简单"
            ),
            EvaluationQuery(
                id="q2",
                query="深度学习和传统机器学习有什么区别？",
                expected_documents=["doc3", "doc4", "doc5"],
                expected_answer="深度学习使用多层神经网络，能够自动学习特征，而传统机器学习通常需要手工设计特征。",
                category="技术对比",
                difficulty="中等"
            ),
            EvaluationQuery(
                id="q3",
                query="如何评估RAG系统的性能？",
                expected_documents=["doc6", "doc7"],
                expected_answer="RAG系统性能评估包括检索质量（精确率、召回率）和生成质量（BLEU、ROUGE等指标）。",
                category="系统评估",
                difficulty="困难"
            )
        ]
        
        self.save_dataset(name, sample_queries)
        return sample_queries
    
    def get_dataset(self, name: str) -> List[EvaluationQuery]:
        """获取数据集"""
        return self.datasets.get(name, [])
    
    def list_datasets(self) -> List[str]:
        """列出所有数据集"""
        return list(self.datasets.keys())

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.llm_evaluator = LLMEvaluator()
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, test_name: str, dataset_name: str, 
                     rag_system: Any, system_version: str = "1.0",
                     parallel: bool = True, max_workers: int = 4) -> BenchmarkResult:
        """运行基准测试"""
        test_id = str(uuid.uuid4())
        logger.info(f"开始基准测试: {test_name} (ID: {test_id})")
        
        # 获取测试数据集
        queries = self.dataset_manager.get_dataset(dataset_name)
        if not queries:
            raise ValueError(f"数据集 {dataset_name} 不存在或为空")
        
        start_time = time.time()
        all_results = []
        
        if parallel and len(queries) > 1:
            # 并行执行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_query = {
                    executor.submit(self._evaluate_single_query, query, rag_system): query
                    for query in queries
                }
                
                for future in tqdm(as_completed(future_to_query), total=len(queries), desc="评估进度"):
                    query = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        logger.error(f"评估查询 {query.id} 失败: {e}")
        else:
            # 串行执行
            for query in tqdm(queries, desc="评估进度"):
                try:
                    results = self._evaluate_single_query(query, rag_system)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"评估查询 {query.id} 失败: {e}")
        
        end_time = time.time()
        
        # 计算汇总指标
        metrics = self._calculate_summary_metrics(all_results)
        
        # 创建基准测试结果
        benchmark_result = BenchmarkResult(
            test_id=test_id,
            test_name=test_name,
            system_version=system_version,
            timestamp=datetime.now(),
            metrics=metrics,
            query_results=all_results,
            summary={
                'total_queries': len(queries),
                'successful_queries': len(set(r.query_id for r in all_results)),
                'total_time': end_time - start_time,
                'avg_time_per_query': (end_time - start_time) / len(queries)
            }
        )
        
        self.results.append(benchmark_result)
        logger.info(f"基准测试完成: {test_name}，总用时 {end_time - start_time:.2f}秒")
        
        return benchmark_result
    
    def _evaluate_single_query(self, query: EvaluationQuery, rag_system: Any) -> List[EvaluationResult]:
        """评估单个查询"""
        results = []
        
        try:
            # 执行RAG查询（这里需要根据实际的RAG系统接口调整）
            start_time = time.time()
            
            # 模拟RAG系统调用
            if hasattr(rag_system, 'search_and_generate'):
                response = rag_system.search_and_generate(query.query)
                retrieval_result = RetrievalResult(
                    query_id=query.id,
                    documents=response.get('documents', []),
                    scores=response.get('scores', []),
                    retrieval_time=response.get('retrieval_time', 0)
                )
                generation_result = GenerationResult(
                    query_id=query.id,
                    answer=response.get('answer', ''),
                    generation_time=response.get('generation_time', 0),
                    source_documents=response.get('source_documents', [])
                )
            else:
                # 模拟结果
                retrieval_result = RetrievalResult(
                    query_id=query.id,
                    documents=[{'id': f'doc{i}', 'content': f'内容{i}'} for i in range(5)],
                    scores=[0.9, 0.8, 0.7, 0.6, 0.5],
                    retrieval_time=0.1
                )
                generation_result = GenerationResult(
                    query_id=query.id,
                    answer="这是一个模拟的答案。",
                    generation_time=0.2
                )
            
            end_time = time.time()
            
            # 评估检索结果
            retrieval_results = self.retrieval_evaluator.evaluate(query, retrieval_result)
            results.extend(retrieval_results)
            
            # 评估生成结果
            generation_results = self.generation_evaluator.evaluate(query, generation_result)
            results.extend(generation_results)
            
            # 添加性能指标
            results.append(EvaluationResult(
                query_id=query.id,
                metric=EvaluationMetric.RESPONSE_TIME,
                value=end_time - start_time
            ))
            
        except Exception as e:
            logger.error(f"评估查询 {query.id} 时发生错误: {e}")
        
        return results
    
    def _calculate_summary_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """计算汇总指标"""
        metrics_by_type = defaultdict(list)
        
        for result in results:
            metrics_by_type[result.metric].append(result.value)
        
        summary = {}
        for metric, values in metrics_by_type.items():
            if values:
                summary[f"{metric.value}_mean"] = statistics.mean(values)
                summary[f"{metric.value}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                summary[f"{metric.value}_min"] = min(values)
                summary[f"{metric.value}_max"] = max(values)
        
        return summary
    
    def get_benchmark_history(self, test_name: str = None) -> List[BenchmarkResult]:
        """获取基准测试历史"""
        if test_name:
            return [r for r in self.results if r.test_name == test_name]
        return self.results

class ABTester:
    """A/B测试器"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.benchmark_runner = BenchmarkRunner(dataset_manager)
        self.results: List[ABTestResult] = []
    
    def run_ab_test(self, test_name: str, dataset_name: str,
                   system_a: Any, system_b: Any,
                   variant_a_name: str = "A", variant_b_name: str = "B",
                   confidence_level: float = 0.95) -> ABTestResult:
        """运行A/B测试"""
        test_id = str(uuid.uuid4())
        logger.info(f"开始A/B测试: {test_name} (ID: {test_id})")
        
        # 运行两个系统的基准测试
        result_a = self.benchmark_runner.run_benchmark(
            f"{test_name}_variant_A", dataset_name, system_a, variant_a_name
        )
        
        result_b = self.benchmark_runner.run_benchmark(
            f"{test_name}_variant_B", dataset_name, system_b, variant_b_name
        )
        
        # 进行统计显著性检验
        significance_results = self._statistical_significance_test(
            result_a.query_results, result_b.query_results, confidence_level
        )
        
        # 确定获胜者
        winner = self._determine_winner(result_a.metrics, result_b.metrics, significance_results)
        
        # 创建A/B测试结果
        ab_result = ABTestResult(
            test_id=test_id,
            test_name=test_name,
            variant_a=variant_a_name,
            variant_b=variant_b_name,
            timestamp=datetime.now(),
            metrics_a=result_a.metrics,
            metrics_b=result_b.metrics,
            statistical_significance=significance_results,
            winner=winner,
            confidence_level=confidence_level
        )
        
        self.results.append(ab_result)
        logger.info(f"A/B测试完成: {test_name}，获胜者: {winner or '无显著差异'}")
        
        return ab_result
    
    def _statistical_significance_test(self, results_a: List[EvaluationResult],
                                     results_b: List[EvaluationResult],
                                     confidence_level: float) -> Dict[str, Dict[str, float]]:
        """统计显著性检验"""
        # 按指标分组
        metrics_a = defaultdict(list)
        metrics_b = defaultdict(list)
        
        for result in results_a:
            metrics_a[result.metric].append(result.value)
        
        for result in results_b:
            metrics_b[result.metric].append(result.value)
        
        significance_results = {}
        
        for metric in set(metrics_a.keys()) & set(metrics_b.keys()):
            values_a = metrics_a[metric]
            values_b = metrics_b[metric]
            
            if len(values_a) < 2 or len(values_b) < 2:
                continue
            
            # 进行t检验
            try:
                t_stat, t_p_value = ttest_ind(values_a, values_b)
                
                # 进行Mann-Whitney U检验（非参数检验）
                u_stat, u_p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                
                significance_results[metric.value] = {
                    't_statistic': float(t_stat),
                    't_p_value': float(t_p_value),
                    't_significant': t_p_value < (1 - confidence_level),
                    'u_statistic': float(u_stat),
                    'u_p_value': float(u_p_value),
                    'u_significant': u_p_value < (1 - confidence_level),
                    'effect_size': self._calculate_effect_size(values_a, values_b)
                }
                
            except Exception as e:
                logger.error(f"统计检验失败 {metric}: {e}")
        
        return significance_results
    
    def _calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> float:
        """计算效应大小（Cohen's d）"""
        try:
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            std_a = statistics.stdev(values_a) if len(values_a) > 1 else 0
            std_b = statistics.stdev(values_b) if len(values_b) > 1 else 0
            
            pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                               (len(values_a) + len(values_b) - 2))
            
            if pooled_std == 0:
                return 0.0
            
            return (mean_a - mean_b) / pooled_std
            
        except Exception:
            return 0.0
    
    def _determine_winner(self, metrics_a: Dict[str, float], metrics_b: Dict[str, float],
                         significance: Dict[str, Dict[str, float]]) -> Optional[str]:
        """确定获胜者"""
        # 定义关键指标及其权重
        key_metrics = {
            'precision_at_k_mean': 0.3,
            'recall_at_k_mean': 0.3,
            'bleu_mean': 0.2,
            'semantic_similarity_mean': 0.2
        }
        
        score_a = 0
        score_b = 0
        total_weight = 0
        
        for metric, weight in key_metrics.items():
            if metric in metrics_a and metric in metrics_b:
                # 检查是否有统计显著性
                base_metric = metric.replace('_mean', '')
                is_significant = significance.get(base_metric, {}).get('t_significant', False)
                
                if is_significant:
                    if metrics_a[metric] > metrics_b[metric]:
                        score_a += weight
                    else:
                        score_b += weight
                    total_weight += weight
        
        if total_weight == 0:
            return None  # 无显著差异
        
        if score_a > score_b:
            return "A"
        elif score_b > score_a:
            return "B"
        else:
            return None  # 平局

class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_benchmark_report(self, result: BenchmarkResult, 
                                include_plots: bool = True) -> str:
        """生成基准测试报告"""
        report_file = self.output_dir / f"benchmark_{result.test_id}.html"
        
        # 生成图表
        plots = []
        if include_plots:
            plots = self._generate_benchmark_plots(result)
        
        # 生成HTML报告
        html_content = self._generate_benchmark_html(result, plots)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"基准测试报告已生成: {report_file}")
        return str(report_file)
    
    def generate_ab_test_report(self, result: ABTestResult,
                              include_plots: bool = True) -> str:
        """生成A/B测试报告"""
        report_file = self.output_dir / f"ab_test_{result.test_id}.html"
        
        # 生成图表
        plots = []
        if include_plots:
            plots = self._generate_ab_test_plots(result)
        
        # 生成HTML报告
        html_content = self._generate_ab_test_html(result, plots)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"A/B测试报告已生成: {report_file}")
        return str(report_file)
    
    def _generate_benchmark_plots(self, result: BenchmarkResult) -> List[str]:
        """生成基准测试图表"""
        plots = []
        
        # 指标分布图
        metrics_data = defaultdict(list)
        for eval_result in result.query_results:
            metrics_data[eval_result.metric.value].append(eval_result.value)
        
        if metrics_data:
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            n_metrics = len(metrics_data)
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            
            for i, (metric, values) in enumerate(metrics_data.items()):
                plt.subplot(rows, cols, i + 1)
                plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'{metric} 分布')
                plt.xlabel('值')
                plt.ylabel('频次')
            
            plt.tight_layout()
            plot_file = self.output_dir / f"benchmark_metrics_{result.test_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.append(str(plot_file))
        
        return plots
    
    def _generate_ab_test_plots(self, result: ABTestResult) -> List[str]:
        """生成A/B测试图表"""
        plots = []
        
        # 指标对比图
        metrics_comparison = {}
        for metric in result.metrics_a.keys():
            if metric in result.metrics_b:
                metrics_comparison[metric] = {
                    'A': result.metrics_a[metric],
                    'B': result.metrics_b[metric]
                }
        
        if metrics_comparison:
            plt.figure(figsize=(12, 6))
            
            metrics = list(metrics_comparison.keys())
            values_a = [metrics_comparison[m]['A'] for m in metrics]
            values_b = [metrics_comparison[m]['B'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, values_a, width, label=f'变体 {result.variant_a}', alpha=0.8)
            plt.bar(x + width/2, values_b, width, label=f'变体 {result.variant_b}', alpha=0.8)
            
            plt.xlabel('指标')
            plt.ylabel('值')
            plt.title('A/B测试指标对比')
            plt.xticks(x, metrics, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            plot_file = self.output_dir / f"ab_test_comparison_{result.test_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.append(str(plot_file))
        
        return plots
    
    def _generate_benchmark_html(self, result: BenchmarkResult, plots: List[str]) -> str:
        """生成基准测试HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>基准测试报告 - {result.test_name}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>基准测试报告</h1>
                <p><strong>测试名称:</strong> {result.test_name}</p>
                <p><strong>系统版本:</strong> {result.system_version}</p>
                <p><strong>测试时间:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>测试ID:</strong> {result.test_id}</p>
            </div>
            
            <h2>测试摘要</h2>
            <div class="metrics">
                <div class="metric"><strong>总查询数:</strong> {result.summary['total_queries']}</div>
                <div class="metric"><strong>成功查询数:</strong> {result.summary['successful_queries']}</div>
                <div class="metric"><strong>总用时:</strong> {result.summary['total_time']:.2f} 秒</div>
                <div class="metric"><strong>平均每查询用时:</strong> {result.summary['avg_time_per_query']:.2f} 秒</div>
            </div>
            
            <h2>性能指标</h2>
            <table>
                <tr><th>指标</th><th>平均值</th><th>标准差</th><th>最小值</th><th>最大值</th></tr>
        """
        
        for metric, value in result.metrics.items():
            if metric.endswith('_mean'):
                base_metric = metric.replace('_mean', '')
                std_metric = f"{base_metric}_std"
                min_metric = f"{base_metric}_min"
                max_metric = f"{base_metric}_max"
                
                html += f"""
                <tr>
                    <td>{base_metric}</td>
                    <td>{value:.4f}</td>
                    <td>{result.metrics.get(std_metric, 0):.4f}</td>
                    <td>{result.metrics.get(min_metric, 0):.4f}</td>
                    <td>{result.metrics.get(max_metric, 0):.4f}</td>
                </tr>
                """
        
        html += "</table>"
        
        # 添加图表
        for plot in plots:
            plot_name = Path(plot).name
            html += f'<div class="plot"><h3>图表</h3><img src="{plot_name}" alt="图表" style="max-width: 100%;"></div>'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_ab_test_html(self, result: ABTestResult, plots: List[str]) -> str:
        """生成A/B测试HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B测试报告 - {result.test_name}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .winner {{ background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .metrics {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ background-color: #fff3cd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>A/B测试报告</h1>
                <p><strong>测试名称:</strong> {result.test_name}</p>
                <p><strong>变体A:</strong> {result.variant_a}</p>
                <p><strong>变体B:</strong> {result.variant_b}</p>
                <p><strong>测试时间:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>置信水平:</strong> {result.confidence_level * 100}%</p>
            </div>
            
            <div class="winner">
                <h2>测试结果</h2>
                <p><strong>获胜者:</strong> {result.winner or '无显著差异'}</p>
            </div>
            
            <h2>指标对比</h2>
            <table>
                <tr><th>指标</th><th>变体A</th><th>变体B</th><th>差异</th><th>统计显著性</th></tr>
        """
        
        for metric in result.metrics_a.keys():
            if metric in result.metrics_b:
                value_a = result.metrics_a[metric]
                value_b = result.metrics_b[metric]
                diff = ((value_b - value_a) / value_a * 100) if value_a != 0 else 0
                
                base_metric = metric.replace('_mean', '')
                significance = result.statistical_significance.get(base_metric, {})
                is_significant = significance.get('t_significant', False)
                p_value = significance.get('t_p_value', 1.0)
                
                row_class = 'significant' if is_significant else ''
                
                html += f"""
                <tr class="{row_class}">
                    <td>{metric}</td>
                    <td>{value_a:.4f}</td>
                    <td>{value_b:.4f}</td>
                    <td>{diff:+.2f}%</td>
                    <td>{'是' if is_significant else '否'} (p={p_value:.4f})</td>
                </tr>
                """
        
        html += "</table>"
        
        # 添加统计显著性详情
        html += "<h2>统计显著性详情</h2><table><tr><th>指标</th><th>t统计量</th><th>p值</th><th>效应大小</th></tr>"
        
        for metric, stats in result.statistical_significance.items():
            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{stats.get('t_statistic', 0):.4f}</td>
                <td>{stats.get('t_p_value', 1):.4f}</td>
                <td>{stats.get('effect_size', 0):.4f}</td>
            </tr>
            """
        
        html += "</table>"
        
        # 添加图表
        for plot in plots:
            plot_name = Path(plot).name
            html += f'<div class="plot"><h3>图表</h3><img src="{plot_name}" alt="图表" style="max-width: 100%;"></div>'
        
        html += """
        </body>
        </html>
        """
        
        return html

class RAGEvaluationSystem:
    """RAG评估系统主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化组件
        self.dataset_manager = DatasetManager(
            self.config.get('dataset_dir', 'evaluation_datasets')
        )
        self.benchmark_runner = BenchmarkRunner(self.dataset_manager)
        self.ab_tester = ABTester(self.dataset_manager)
        self.reporter = EvaluationReporter(
            self.config.get('output_dir', 'evaluation_reports')
        )
        
        # 评估历史
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def create_evaluation_dataset(self, name: str, queries: List[Dict[str, Any]]) -> List[EvaluationQuery]:
        """创建评估数据集"""
        eval_queries = []
        for query_data in queries:
            query = EvaluationQuery(
                id=query_data.get('id', str(uuid.uuid4())),
                query=query_data['query'],
                expected_documents=query_data.get('expected_documents', []),
                expected_answer=query_data.get('expected_answer'),
                category=query_data.get('category'),
                difficulty=query_data.get('difficulty'),
                metadata=query_data.get('metadata', {})
            )
            eval_queries.append(query)
        
        self.dataset_manager.save_dataset(name, eval_queries)
        return eval_queries
    
    def run_evaluation(self, test_name: str, dataset_name: str, rag_system: Any,
                      test_type: TestType = TestType.BENCHMARK_TEST,
                      **kwargs) -> Union[BenchmarkResult, ABTestResult]:
        """运行评估"""
        if test_type == TestType.BENCHMARK_TEST:
            result = self.benchmark_runner.run_benchmark(
                test_name, dataset_name, rag_system, **kwargs
            )
            
            # 生成报告
            report_path = self.reporter.generate_benchmark_report(result)
            
            # 记录历史
            self.evaluation_history.append({
                'type': 'benchmark',
                'result': result,
                'report_path': report_path,
                'timestamp': datetime.now()
            })
            
            return result
            
        elif test_type == TestType.AB_TEST:
            system_b = kwargs.get('system_b')
            if not system_b:
                raise ValueError("A/B测试需要提供system_b参数")
            
            result = self.ab_tester.run_ab_test(
                test_name, dataset_name, rag_system, system_b, **kwargs
            )
            
            # 生成报告
            report_path = self.reporter.generate_ab_test_report(result)
            
            # 记录历史
            self.evaluation_history.append({
                'type': 'ab_test',
                'result': result,
                'report_path': report_path,
                'timestamp': datetime.now()
            })
            
            return result
        
        else:
            raise ValueError(f"不支持的测试类型: {test_type}")
    
    def get_evaluation_history(self, test_type: str = None) -> List[Dict[str, Any]]:
        """获取评估历史"""
        if test_type:
            return [h for h in self.evaluation_history if h['type'] == test_type]
        return self.evaluation_history
    
    def compare_systems(self, system_configs: List[Dict[str, Any]], 
                       dataset_name: str, test_name: str = "系统对比") -> Dict[str, Any]:
        """对比多个系统"""
        results = {}
        
        for config in system_configs:
            system_name = config['name']
            system = config['system']
            
            logger.info(f"评估系统: {system_name}")
            result = self.benchmark_runner.run_benchmark(
                f"{test_name}_{system_name}", dataset_name, system, system_name
            )
            results[system_name] = result
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(results, test_name)
        
        return {
            'results': results,
            'comparison': comparison_report,
            'timestamp': datetime.now()
        }
    
    def _generate_comparison_report(self, results: Dict[str, BenchmarkResult], 
                                  test_name: str) -> Dict[str, Any]:
        """生成系统对比报告"""
        comparison = {
            'test_name': test_name,
            'systems': list(results.keys()),
            'metrics_comparison': {},
            'rankings': {}
        }
        
        # 收集所有指标
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result.metrics.keys())
        
        # 对比指标
        for metric in all_metrics:
            metric_values = {}
            for system_name, result in results.items():
                if metric in result.metrics:
                    metric_values[system_name] = result.metrics[metric]
            
            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values
                
                # 排名（降序）
                sorted_systems = sorted(metric_values.items(), 
                                       key=lambda x: x[1], reverse=True)
                comparison['rankings'][metric] = [s[0] for s in sorted_systems]
        
        return comparison
    
    def export_results(self, output_file: str, format: str = 'json'):
        """导出评估结果"""
        if format.lower() == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_history, f, ensure_ascii=False, 
                         indent=2, default=str)
        elif format.lower() == 'csv':
            # 导出为CSV格式
            rows = []
            for history in self.evaluation_history:
                if history['type'] == 'benchmark':
                    result = history['result']
                    for metric, value in result.metrics.items():
                        rows.append({
                            'timestamp': history['timestamp'],
                            'test_type': 'benchmark',
                            'test_name': result.test_name,
                            'system_version': result.system_version,
                            'metric': metric,
                            'value': value
                        })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"评估结果已导出到: {output_file}")

# 模拟RAG系统类（用于演示）
class MockRAGSystem:
    """模拟RAG系统"""
    
    def __init__(self, name: str = "MockRAG", performance_factor: float = 1.0):
        self.name = name
        self.performance_factor = performance_factor
    
    def search_and_generate(self, query: str) -> Dict[str, Any]:
        """模拟搜索和生成"""
        # 模拟检索延迟
        retrieval_time = random.uniform(0.05, 0.2) / self.performance_factor
        time.sleep(retrieval_time)
        
        # 模拟生成延迟
        generation_time = random.uniform(0.1, 0.5) / self.performance_factor
        time.sleep(generation_time)
        
        # 模拟检索结果
        num_docs = random.randint(3, 8)
        documents = []
        scores = []
        
        for i in range(num_docs):
            doc_id = f"doc_{random.randint(1, 100)}"
            content = f"这是文档{i+1}的内容，与查询'{query}'相关。"
            score = random.uniform(0.3, 0.95) * self.performance_factor
            
            documents.append({
                'id': doc_id,
                'content': content,
                'metadata': {'source': f'source_{i}'}
            })
            scores.append(score)
        
        # 按分数排序
        sorted_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        documents, scores = zip(*sorted_pairs)
        
        # 模拟生成答案
        answer_quality = random.uniform(0.6, 0.9) * self.performance_factor
        if answer_quality > 0.8:
            answer = f"根据检索到的文档，{query}的答案是：这是一个高质量的回答，包含了相关的详细信息。"
        elif answer_quality > 0.6:
            answer = f"关于{query}，根据相关文档可以得出：这是一个中等质量的回答。"
        else:
            answer = f"对于{query}这个问题，答案可能是：这是一个较低质量的回答。"
        
        return {
            'documents': list(documents),
            'scores': list(scores),
            'answer': answer,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'source_documents': [doc['id'] for doc in documents[:3]]
        }

def main():
    """主函数 - 演示RAG评估系统的使用"""
    print("=== RAG系统评估演示 ===")
    
    # 1. 初始化评估系统
    evaluation_system = RAGEvaluationSystem({
        'dataset_dir': 'evaluation_datasets',
        'output_dir': 'evaluation_reports'
    })
    
    # 2. 创建示例数据集
    print("\n1. 创建评估数据集...")
    sample_queries = [
        {
            'id': 'q1',
            'query': '什么是机器学习？',
            'expected_documents': ['doc1', 'doc2', 'doc15'],
            'expected_answer': '机器学习是一种人工智能技术，通过算法让计算机从数据中学习模式和规律。',
            'category': '基础概念',
            'difficulty': '简单'
        },
        {
            'id': 'q2', 
            'query': '深度学习和传统机器学习的区别？',
            'expected_documents': ['doc3', 'doc4', 'doc25'],
            'expected_answer': '深度学习使用多层神经网络自动学习特征，而传统机器学习需要手工设计特征。',
            'category': '技术对比',
            'difficulty': '中等'
        },
        {
            'id': 'q3',
            'query': 'RAG系统的核心组件有哪些？',
            'expected_documents': ['doc5', 'doc6', 'doc35'],
            'expected_answer': 'RAG系统包含检索器、生成器和知识库三个核心组件。',
            'category': '系统架构',
            'difficulty': '中等'
        }
    ]
    
    dataset = evaluation_system.create_evaluation_dataset('demo_dataset', sample_queries)
    print(f"创建了包含 {len(dataset)} 个查询的数据集")
    
    # 3. 创建模拟RAG系统
    print("\n2. 创建模拟RAG系统...")
    rag_system_v1 = MockRAGSystem("RAG_v1.0", performance_factor=0.8)
    rag_system_v2 = MockRAGSystem("RAG_v2.0", performance_factor=1.2)
    
    # 4. 运行基准测试
    print("\n3. 运行基准测试...")
    benchmark_result = evaluation_system.run_evaluation(
        test_name="RAG系统基准测试",
        dataset_name="demo_dataset",
        rag_system=rag_system_v1,
        test_type=TestType.BENCHMARK_TEST,
        system_version="v1.0"
    )
    
    print(f"基准测试完成，测试ID: {benchmark_result.test_id}")
    print(f"主要指标:")
    for metric, value in list(benchmark_result.metrics.items())[:5]:
        print(f"  {metric}: {value:.4f}")
    
    # 5. 运行A/B测试
    print("\n4. 运行A/B测试...")
    ab_result = evaluation_system.run_evaluation(
        test_name="RAG系统A/B测试",
        dataset_name="demo_dataset",
        rag_system=rag_system_v1,
        test_type=TestType.AB_TEST,
        system_b=rag_system_v2,
        variant_a_name="v1.0",
        variant_b_name="v2.0"
    )
    
    print(f"A/B测试完成，获胜者: {ab_result.winner or '无显著差异'}")
    
    # 6. 系统对比
    print("\n5. 多系统对比...")
    system_configs = [
        {'name': 'RAG_v1.0', 'system': rag_system_v1},
        {'name': 'RAG_v2.0', 'system': rag_system_v2},
        {'name': 'RAG_v1.5', 'system': MockRAGSystem("RAG_v1.5", 1.0)}
    ]
    
    comparison_result = evaluation_system.compare_systems(
        system_configs, 'demo_dataset', '多系统性能对比'
    )
    
    print("系统对比完成:")
    for metric, rankings in comparison_result['comparison']['rankings'].items():
        if 'mean' in metric:
            print(f"  {metric}: {' > '.join(rankings[:3])}")
    
    # 7. 导出结果
    print("\n6. 导出评估结果...")
    evaluation_system.export_results('evaluation_results.json', 'json')
    evaluation_system.export_results('evaluation_results.csv', 'csv')
    
    # 8. 显示评估历史
    print("\n7. 评估历史:")
    history = evaluation_system.get_evaluation_history()
    for i, record in enumerate(history):
        print(f"  {i+1}. {record['type']} - {record['timestamp'].strftime('%H:%M:%S')}")
    
    print("\n=== RAG评估系统演示完成 ===")
    print("\n生成的文件:")
    print("- evaluation_datasets/demo_dataset.json (评估数据集)")
    print("- evaluation_reports/*.html (评估报告)")
    print("- evaluation_results.json (结果导出)")
    print("- evaluation_results.csv (结果导出)")

if __name__ == "__main__":
    main()