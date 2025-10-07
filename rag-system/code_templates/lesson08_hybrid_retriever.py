#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson08 混合检索器实现模板
解决BM25集成和融合策略优化缺失问题

功能特性：
1. BM25关键词检索
2. 语义向量检索
3. 多种融合策略（RRF、加权融合等）
4. 检索结果重排序
5. 性能优化和缓存
"""

import logging
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果结构"""
    doc_id: str
    content: str
    score: float
    source: str  # 'bm25', 'semantic', 'hybrid'
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextProcessor:
    """文本预处理器"""
    
    def __init__(self, use_jieba: bool = True):
        self.use_jieba = use_jieba
        if use_jieba:
            jieba.initialize()
    
    def tokenize(self, text: str) -> List[str]:
        """文本分词"""
        if not text:
            return []
        
        if self.use_jieba:
            # 中文分词
            tokens = list(jieba.cut(text.lower()))
        else:
            # 英文分词
            tokens = text.lower().split()
        
        # 过滤停用词和短词
        filtered_tokens = [token for token in tokens 
                          if len(token) > 1 and token.isalnum()]
        
        return filtered_tokens
    
    def preprocess_corpus(self, documents: List[str]) -> List[List[str]]:
        """预处理文档语料库"""
        return [self.tokenize(doc) for doc in documents]

class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, 
                 documents: List[str],
                 doc_ids: List[str] = None,
                 algorithm: str = 'okapi',
                 k1: float = 1.5,
                 b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表
            doc_ids: 文档ID列表
            algorithm: BM25算法类型 ('okapi', 'l', 'plus')
            k1: BM25参数k1
            b: BM25参数b
        """
        self.documents = documents
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        self.processor = TextProcessor()
        
        # 预处理文档
        logger.info(f"预处理{len(documents)}个文档...")
        self.tokenized_corpus = self.processor.preprocess_corpus(documents)
        
        # 初始化BM25模型
        if algorithm == 'okapi':
            self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
        elif algorithm == 'l':
            self.bm25 = BM25L(self.tokenized_corpus, k1=k1, b=b)
        elif algorithm == 'plus':
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=k1, b=b)
        else:
            raise ValueError(f"不支持的BM25算法: {algorithm}")
        
        logger.info(f"BM25索引构建完成，算法: {algorithm}")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """BM25搜索"""
        if not query:
            return []
        
        # 查询预处理
        query_tokens = self.processor.tokenize(query)
        if not query_tokens:
            return []
        
        # 计算BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有相关性的结果
                result = SearchResult(
                    doc_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    source='bm25'
                )
                results.append(result)
        
        return results
    
    def get_term_frequencies(self, query: str) -> Dict[str, float]:
        """获取查询词在语料库中的词频"""
        query_tokens = self.processor.tokenize(query)
        term_freqs = {}
        
        for token in query_tokens:
            # 计算词在整个语料库中的频率
            doc_freq = sum(1 for doc in self.tokenized_corpus if token in doc)
            term_freqs[token] = doc_freq / len(self.tokenized_corpus)
        
        return term_freqs

class SemanticRetriever:
    """语义检索器"""
    
    def __init__(self, 
                 documents: List[str],
                 doc_ids: List[str] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = './cache'):
        """
        初始化语义检索器
        
        Args:
            documents: 文档列表
            doc_ids: 文档ID列表
            model_name: 预训练模型名称
            cache_dir: 缓存目录
        """
        self.documents = documents
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 加载预训练模型
        logger.info(f"加载语义模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 计算或加载文档向量
        self.doc_embeddings = self._get_or_compute_embeddings()
        
        logger.info(f"语义检索器初始化完成，文档数: {len(documents)}")
    
    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        # 基于文档内容生成缓存键
        content_hash = hashlib.md5(
            ''.join(self.documents).encode('utf-8')
        ).hexdigest()[:8]
        return self.cache_dir / f"embeddings_{content_hash}.pkl"
    
    def _get_or_compute_embeddings(self) -> np.ndarray:
        """获取或计算文档向量"""
        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            logger.info("从缓存加载文档向量...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info("计算文档向量...")
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # 保存到缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """语义搜索"""
        if not query:
            return []
        
        # 计算查询向量
        query_embedding = self.model.encode([query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 设置最小相似度阈值
                result = SearchResult(
                    doc_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(similarities[idx]),
                    source='semantic'
                )
                results.append(result)
        
        return results

class HybridFusion:
    """混合融合策略"""
    
    @staticmethod
    def reciprocal_rank_fusion(results_list: List[List[SearchResult]], 
                              k: int = 60) -> List[SearchResult]:
        """倒数排名融合(RRF)"""
        if not results_list:
            return []
        
        # 收集所有文档
        all_docs = {}
        doc_scores = defaultdict(float)
        
        for results in results_list:
            for rank, result in enumerate(results):
                doc_id = result.doc_id
                all_docs[doc_id] = result
                
                # RRF公式: 1/(k + rank)
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] += rrf_score
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for doc_id, score in sorted_docs:
            result = all_docs[doc_id]
            result.score = score
            result.source = 'hybrid'
            fused_results.append(result)
        
        return fused_results
    
    @staticmethod
    def weighted_fusion(semantic_results: List[SearchResult],
                       bm25_results: List[SearchResult],
                       semantic_weight: float = 0.7,
                       bm25_weight: float = 0.3) -> List[SearchResult]:
        """加权融合"""
        # 标准化分数
        semantic_scores = HybridFusion._normalize_scores(
            [r.score for r in semantic_results]
        )
        bm25_scores = HybridFusion._normalize_scores(
            [r.score for r in bm25_results]
        )
        
        # 更新标准化分数
        for i, result in enumerate(semantic_results):
            result.score = semantic_scores[i]
        for i, result in enumerate(bm25_results):
            result.score = bm25_scores[i]
        
        # 合并结果
        all_docs = {}
        doc_scores = defaultdict(float)
        
        # 语义检索结果
        for result in semantic_results:
            doc_id = result.doc_id
            all_docs[doc_id] = result
            doc_scores[doc_id] += result.score * semantic_weight
        
        # BM25检索结果
        for result in bm25_results:
            doc_id = result.doc_id
            if doc_id not in all_docs:
                all_docs[doc_id] = result
            doc_scores[doc_id] += result.score * bm25_weight
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for doc_id, score in sorted_docs:
            result = all_docs[doc_id]
            result.score = score
            result.source = 'hybrid'
            fused_results.append(result)
        
        return fused_results
    
    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """分数标准化"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) 
                for score in scores]

class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, 
                 documents: List[str],
                 doc_ids: List[str] = None,
                 semantic_model: str = 'all-MiniLM-L6-v2',
                 bm25_algorithm: str = 'okapi',
                 cache_dir: str = './cache'):
        """
        初始化混合检索器
        
        Args:
            documents: 文档列表
            doc_ids: 文档ID列表
            semantic_model: 语义模型名称
            bm25_algorithm: BM25算法类型
            cache_dir: 缓存目录
        """
        self.documents = documents
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        
        # 初始化检索器
        logger.info("初始化BM25检索器...")
        self.bm25_retriever = BM25Retriever(
            documents, doc_ids, algorithm=bm25_algorithm
        )
        
        logger.info("初始化语义检索器...")
        self.semantic_retriever = SemanticRetriever(
            documents, doc_ids, model_name=semantic_model, cache_dir=cache_dir
        )
        
        self.fusion = HybridFusion()
        
        logger.info("混合检索器初始化完成")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               fusion_method: str = 'rrf',
               semantic_weight: float = 0.7,
               bm25_weight: float = 0.3) -> List[SearchResult]:
        """混合搜索"""
        if not query:
            return []
        
        # 并行检索
        logger.debug(f"执行混合检索: {query}")
        
        semantic_results = self.semantic_retriever.search(query, top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k * 2)
        
        logger.debug(f"语义检索结果: {len(semantic_results)}")
        logger.debug(f"BM25检索结果: {len(bm25_results)}")
        
        # 融合结果
        if fusion_method == 'rrf':
            fused_results = self.fusion.reciprocal_rank_fusion(
                [semantic_results, bm25_results]
            )
        elif fusion_method == 'weighted':
            fused_results = self.fusion.weighted_fusion(
                semantic_results, bm25_results, 
                semantic_weight, bm25_weight
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        return fused_results[:top_k]
    
    def search_semantic_only(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """仅语义搜索"""
        return self.semantic_retriever.search(query, top_k)
    
    def search_bm25_only(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """仅BM25搜索"""
        return self.bm25_retriever.search(query, top_k)
    
    def analyze_query(self, query: str) -> Dict:
        """查询分析"""
        analysis = {
            'query': query,
            'query_length': len(query),
            'token_count': len(self.bm25_retriever.processor.tokenize(query)),
            'term_frequencies': self.bm25_retriever.get_term_frequencies(query)
        }
        return analysis

def main():
    """示例用法"""
    # 示例文档
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
        "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能的一个分支，使计算机能够识别和理解图像和视频内容。"
    ]
    
    # 初始化混合检索器
    retriever = HybridRetriever(documents)
    
    # 执行搜索
    query = "什么是机器学习"
    results = retriever.search(query, top_k=3)
    
    print(f"查询: {query}")
    print(f"找到 {len(results)} 个结果:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result.score:.4f}] [来源: {result.source}]")
        print(f"   内容: {result.content}\n")
    
    # 查询分析
    analysis = retriever.analyze_query(query)
    print(f"查询分析: {analysis}")

if __name__ == "__main__":
    main()