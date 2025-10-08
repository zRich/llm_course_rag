#!/usr/bin/env python3
"""
Lesson 06 - 向量检索演示
演示向量相似度计算和TopK检索的基本实现
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Document:
    """文档数据结构"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = None


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Document
    score: float
    rank: int


class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算点积相似度"""
        return np.dot(vec1, vec2)
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算欧几里得距离（转换为相似度）"""
        distance = np.linalg.norm(vec1 - vec2)
        # 转换为相似度：距离越小，相似度越高
        return 1.0 / (1.0 + distance)


class MockEmbeddingModel:
    """模拟向量化模型（用于演示）"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        np.random.seed(42)  # 确保结果可重现
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """模拟文本向量化"""
        vectors = []
        for text in texts:
            # 基于文本内容生成伪随机向量
            hash_value = hash(text) % 1000000
            np.random.seed(hash_value)
            vector = np.random.normal(0, 1, self.dimension)
            # 归一化向量
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        
        return np.array(vectors)


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.documents = []
        self.vectors = None
        self.similarity_calculator = SimilarityCalculator()
    
    def add_documents(self, documents: List[Document]):
        """添加文档到检索器"""
        self.documents.extend(documents)
        
        # 向量化所有文档内容
        texts = [doc.content for doc in documents]
        new_vectors = self.embedding_model.encode(texts)
        
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        
        print(f"已添加 {len(documents)} 个文档，总计 {len(self.documents)} 个文档")
    
    def search(self, query: str, top_k: int = 5, 
               similarity_method: str = "cosine") -> List[SearchResult]:
        """执行向量检索"""
        if not self.documents:
            return []
        
        # 向量化查询
        query_vector = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            if similarity_method == "cosine":
                score = self.similarity_calculator.cosine_similarity(
                    query_vector, doc_vector
                )
            elif similarity_method == "dot_product":
                score = self.similarity_calculator.dot_product_similarity(
                    query_vector, doc_vector
                )
            elif similarity_method == "euclidean":
                score = self.similarity_calculator.euclidean_distance(
                    query_vector, doc_vector
                )
            else:
                raise ValueError(f"不支持的相似度方法: {similarity_method}")
            
            similarities.append((i, score))
        
        # 按相似度排序并取TopK
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # 构建搜索结果
        results = []
        for rank, (doc_idx, score) in enumerate(top_results, 1):
            result = SearchResult(
                document=self.documents[doc_idx],
                score=score,
                rank=rank
            )
            results.append(result)
        
        return results


class QdrantRetriever:
    """基于Qdrant的向量检索器"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "rag_documents"
        self.embedding_model = MockEmbeddingModel()
        self._setup_collection()
    
    def _setup_collection(self):
        """设置Qdrant集合"""
        try:
            # 删除已存在的集合（用于演示）
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # 创建新集合
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_model.dimension,
                distance=Distance.COSINE
            )
        )
        print(f"已创建Qdrant集合: {self.collection_name}")
    
    def add_documents(self, documents: List[Document]):
        """添加文档到Qdrant"""
        # 向量化文档
        texts = [doc.content for doc in documents]
        vectors = self.embedding_model.encode(texts)
        
        # 构建Points
        points = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            point = PointStruct(
                id=hash(doc.id) % 1000000,  # 简单的ID生成
                vector=vector.tolist(),
                payload={
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": doc.metadata or {}
                }
            )
            points.append(point)
        
        # 批量插入
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"已向Qdrant添加 {len(documents)} 个文档")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """使用Qdrant执行向量检索"""
        # 向量化查询
        query_vector = self.embedding_model.encode([query])[0]
        
        # 执行搜索
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        # 构建结果
        results = []
        for rank, hit in enumerate(search_results, 1):
            doc = Document(
                id=hit.payload["id"],
                title=hit.payload["title"],
                content=hit.payload["content"],
                metadata=hit.payload["metadata"]
            )
            
            result = SearchResult(
                document=doc,
                score=hit.score,
                rank=rank
            )
            results.append(result)
        
        return results


def demo_similarity_methods():
    """演示不同相似度计算方法"""
    print("=== 相似度计算方法演示 ===")
    
    # 创建示例文档
    documents = [
        Document("1", "Python编程", "Python是一种高级编程语言，易于学习和使用"),
        Document("2", "机器学习", "机器学习是人工智能的一个分支，使用算法从数据中学习"),
        Document("3", "深度学习", "深度学习使用神经网络来模拟人脑的学习过程"),
        Document("4", "数据科学", "数据科学结合统计学、编程和领域知识来分析数据"),
        Document("5", "自然语言处理", "NLP是计算机科学和人工智能的交叉领域")
    ]
    
    # 创建检索器
    retriever = VectorRetriever()
    retriever.add_documents(documents)
    
    query = "人工智能和机器学习"
    print(f"\n查询: {query}")
    
    # 测试不同相似度方法
    methods = ["cosine", "dot_product", "euclidean"]
    
    for method in methods:
        print(f"\n--- {method.upper()} 相似度 ---")
        results = retriever.search(query, top_k=3, similarity_method=method)
        
        for result in results:
            print(f"排名 {result.rank}: {result.document.title}")
            print(f"  相似度: {result.score:.4f}")
            print(f"  内容: {result.document.content[:50]}...")


def demo_qdrant_retrieval():
    """演示Qdrant向量检索"""
    print("\n=== Qdrant向量检索演示 ===")
    
    try:
        # 创建Qdrant检索器
        qdrant_retriever = QdrantRetriever()
        
        # 添加示例文档
        documents = [
            Document("doc1", "RAG系统介绍", 
                    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构"),
            Document("doc2", "向量数据库", 
                    "向量数据库专门用于存储和检索高维向量数据，支持相似度搜索"),
            Document("doc3", "文本向量化", 
                    "文本向量化将文本转换为数值向量，保持语义信息"),
            Document("doc4", "语言模型", 
                    "大语言模型如GPT能够理解和生成人类语言"),
            Document("doc5", "检索增强", 
                    "检索增强通过外部知识库提高生成质量和准确性")
        ]
        
        qdrant_retriever.add_documents(documents)
        
        # 执行搜索
        queries = [
            "什么是RAG系统？",
            "向量数据库的作用",
            "如何提高生成质量？"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            results = qdrant_retriever.search(query, top_k=3)
            
            for result in results:
                print(f"排名 {result.rank}: {result.document.title}")
                print(f"  相似度: {result.score:.4f}")
                print(f"  内容: {result.document.content}")
    
    except Exception as e:
        print(f"Qdrant演示失败: {e}")
        print("请确保Qdrant服务正在运行 (docker run -p 6333:6333 qdrant/qdrant)")


def performance_benchmark():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    # 生成大量测试文档
    documents = []
    for i in range(1000):
        doc = Document(
            id=f"doc_{i}",
            title=f"文档 {i}",
            content=f"这是第 {i} 个测试文档，包含一些示例内容用于向量化和检索测试。"
        )
        documents.append(doc)
    
    # 创建检索器并添加文档
    retriever = VectorRetriever()
    
    start_time = time.time()
    retriever.add_documents(documents)
    index_time = time.time() - start_time
    
    print(f"索引构建时间: {index_time:.2f}秒")
    print(f"文档数量: {len(documents)}")
    print(f"平均每文档索引时间: {index_time/len(documents)*1000:.2f}毫秒")
    
    # 测试检索性能
    test_queries = [
        "测试文档内容",
        "示例向量化",
        "检索性能测试"
    ]
    
    total_search_time = 0
    for query in test_queries:
        start_time = time.time()
        results = retriever.search(query, top_k=10)
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(f"查询 '{query}' 耗时: {search_time*1000:.2f}毫秒")
    
    avg_search_time = total_search_time / len(test_queries)
    print(f"平均检索时间: {avg_search_time*1000:.2f}毫秒")


if __name__ == "__main__":
    print("Lesson 06 - 向量检索演示")
    print("=" * 50)
    
    # 运行演示
    demo_similarity_methods()
    demo_qdrant_retrieval()
    performance_benchmark()
    
    print("\n演示完成！")
    print("\n关键要点:")
    print("1. 余弦相似度适用于大多数文本检索场景")
    print("2. Qdrant提供了高效的向量存储和检索能力")
    print("3. 检索性能随文档数量增长，需要考虑索引优化")
    print("4. 向量质量直接影响检索效果")