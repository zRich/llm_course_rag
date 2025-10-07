#!/usr/bin/env python3
"""模拟RAG系统用于Chunk参数测试"""

import time
import random
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class MockChunk:
    """模拟文档块"""
    chunk_id: str
    content: str
    source_doc: str
    start_pos: int
    end_pos: int
    embedding: Optional[List[float]] = None

@dataclass
class MockSearchResult:
    """模拟搜索结果"""
    chunk_id: str
    content: str
    score: float
    source_doc: str
    
    def get(self, key: str, default=None):
        """获取属性值，兼容字典访问方式"""
        return getattr(self, key, default)

class MockChunkManager:
    """模拟分块管理器"""
    
    def __init__(self, chunk_size: int = 500, overlap_ratio: float = 0.2):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.chunks: Dict[str, MockChunk] = {}
        
    def set_params(self, chunk_size: int, overlap_ratio: float):
        """设置分块参数"""
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        
    def chunk_text(self, text: str, source_doc: str) -> List[MockChunk]:
        """将文本分块"""
        chunks = []
        overlap_size = int(self.chunk_size * self.overlap_ratio)
        
        # 简单的文本分块逻辑
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # 尝试在句号处断开
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + self.chunk_size * 0.7:  # 至少保留70%的内容
                    end = last_period + 1
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk_id = f"{source_doc}_chunk_{chunk_index}"
                chunk = MockChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    source_doc=source_doc,
                    start_pos=start,
                    end_pos=end,
                    embedding=self._generate_mock_embedding(chunk_content)
                )
                chunks.append(chunk)
                self.chunks[chunk_id] = chunk
                chunk_index += 1
            
            # 计算下一个起始位置（考虑重叠）
            start = max(start + 1, end - overlap_size)
            
            # 防止无限循环
            if start >= end:
                break
                
        return chunks
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """生成模拟的文本嵌入向量"""
        # 使用文本哈希生成确定性的嵌入向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # 转换为384维向量（模拟bge-small-zh的维度）
        embedding = []
        for i in range(0, len(hash_bytes), 2):
            if i + 1 < len(hash_bytes):
                val = (hash_bytes[i] + hash_bytes[i+1] * 256) / 65535.0 * 2 - 1
                embedding.append(val)
        
        # 扩展到384维
        while len(embedding) < 384:
            embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
        
        return embedding[:384]

class MockVectorStore:
    """模拟向量存储"""
    
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict] = {}
        
    def add_chunks(self, chunks: List[MockChunk]):
        """添加文档块到向量存储"""
        for chunk in chunks:
            self.vectors[chunk.chunk_id] = chunk.embedding
            self.metadata[chunk.chunk_id] = {
                'content': chunk.content,
                'source_doc': chunk.source_doc,
                'start_pos': chunk.start_pos,
                'end_pos': chunk.end_pos
            }
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[MockSearchResult]:
        """搜索相似的文档块"""
        results = []
        
        for chunk_id, vector in self.vectors.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, vector)
            
            # 添加一些随机性来模拟真实场景
            similarity += random.uniform(-0.1, 0.1)
            similarity = max(0, min(1, similarity))
            
            metadata = self.metadata[chunk_id]
            result = MockSearchResult(
                chunk_id=chunk_id,
                content=metadata['content'],
                score=similarity,
                source_doc=metadata['source_doc']
            )
            results.append(result)
        
        # 按相似度排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)

class MockRAGSystem:
    """模拟RAG系统"""
    
    def __init__(self, chunk_size: int = 500, overlap_ratio: float = 0.2):
        self.chunk_manager = MockChunkManager(chunk_size, overlap_ratio)
        self.vector_store = MockVectorStore()
        self.documents: Dict[str, str] = {}
        
    def set_chunk_params(self, chunk_size: int, overlap_ratio: float):
        """设置分块参数"""
        self.chunk_manager.set_params(chunk_size, overlap_ratio)
        
    def add_document(self, doc_id: str, content: str):
        """添加文档"""
        self.documents[doc_id] = content
        
    def process_document(self, doc_id: str) -> List[MockChunk]:
        """处理文档并返回分块结果"""
        if doc_id not in self.documents:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        content = self.documents[doc_id]
        chunks = self.chunk_manager.chunk_text(content, doc_id)
        self.vector_store.add_chunks(chunks)
        
        return chunks
    
    def process_all_documents(self):
        """处理所有文档"""
        # 清空现有的向量存储
        self.vector_store = MockVectorStore()
        
        all_chunks = []
        for doc_id in self.documents:
            chunks = self.process_document(doc_id)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def search(self, query: str, top_k: int = 5) -> List[MockSearchResult]:
        """搜索相关文档块"""
        # 生成查询的嵌入向量
        query_embedding = self.chunk_manager._generate_mock_embedding(query)
        
        # 模拟搜索延迟
        time.sleep(random.uniform(0.01, 0.05))
        
        # 执行向量搜索
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """获取分块统计信息"""
        chunks = list(self.chunk_manager.chunks.values())
        
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0,
                'total_documents': len(self.documents)
            }
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_documents': len(self.documents),
            'chunk_size_setting': self.chunk_manager.chunk_size,
            'overlap_ratio_setting': self.chunk_manager.overlap_ratio
        }
    
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估检索性能"""
        total_queries = len(test_queries)
        correct_retrievals = 0
        total_relevant = 0
        total_retrieved = 0
        total_response_time = 0
        
        for query_data in test_queries:
            query = query_data['query']
            expected_docs = query_data.get('expected_docs', [])
            
            # 执行搜索
            start_time = time.time()
            results = self.search(query, top_k=5)
            end_time = time.time()
            
            total_response_time += (end_time - start_time) * 1000  # 转换为毫秒
            
            # 计算准确率和召回率
            retrieved_docs = [result.source_doc for result in results]
            
            # 计算相关文档的交集
            relevant_retrieved = len(set(retrieved_docs) & set(expected_docs))
            correct_retrievals += relevant_retrieved
            total_retrieved += len(retrieved_docs)
            total_relevant += len(expected_docs)
        
        # 计算指标
        accuracy = correct_retrievals / total_retrieved if total_retrieved > 0 else 0
        recall = correct_retrievals / total_relevant if total_relevant > 0 else 0
        f1_score = 2 * accuracy * recall / (accuracy + recall) if (accuracy + recall) > 0 else 0
        avg_response_time = total_response_time / total_queries if total_queries > 0 else 0
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1_score,
            'avg_response_time': avg_response_time
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息（兼容性方法）"""
        return self.get_chunk_statistics()

class MockDocumentGenerator:
    """模拟文档生成器"""
    
    @staticmethod
    def generate_test_documents(num_docs: int = 10, doc_length: int = 2000) -> Dict[str, str]:
        """生成测试文档"""
        documents = {}
        
        # 预定义的主题和内容模板
        topics = [
            "人工智能技术发展",
            "机器学习算法原理",
            "深度学习应用场景",
            "自然语言处理技术",
            "计算机视觉应用",
            "数据科学方法论",
            "云计算架构设计",
            "区块链技术原理",
            "物联网系统开发",
            "网络安全防护策略"
        ]
        
        content_templates = [
            "在现代科技发展中，{topic}扮演着越来越重要的角色。通过深入研究和实践应用，我们可以发现其在各个领域的广泛应用价值。",
            "{topic}的核心原理基于先进的算法和数学模型。这些技术的发展为解决复杂问题提供了新的思路和方法。",
            "随着{topic}技术的不断成熟，其在工业界和学术界都获得了广泛关注。相关研究成果为未来发展奠定了坚实基础。",
            "实践证明，{topic}在提高效率、降低成本、优化流程等方面具有显著优势。这为相关行业的数字化转型提供了强有力的支撑。"
        ]
        
        for i in range(num_docs):
            topic = topics[i % len(topics)]
            doc_id = f"doc_{i+1:03d}"
            
            # 生成文档内容
            content_parts = []
            current_length = 0
            
            while current_length < doc_length:
                template = random.choice(content_templates)
                paragraph = template.format(topic=topic)
                
                # 添加一些随机的详细内容
                details = [
                    f"具体来说，{topic}涉及多个技术层面的创新。",
                    f"从技术实现角度看，{topic}需要考虑性能、可扩展性和可维护性等因素。",
                    f"在实际应用中，{topic}面临着数据质量、算法优化、系统集成等挑战。",
                    f"未来，{topic}的发展趋势将更加注重智能化、自动化和个性化。",
                    f"通过不断的技术迭代和优化，{topic}将为用户提供更好的体验和价值。"
                ]
                
                paragraph += " " + " ".join(random.sample(details, random.randint(2, 4)))
                content_parts.append(paragraph)
                current_length += len(paragraph)
            
            documents[doc_id] = " ".join(content_parts)
        
        return documents
    
    @staticmethod
    def generate_test_queries(documents: Dict[str, str], num_queries: int = 20) -> List[Dict[str, Any]]:
        """生成测试查询"""
        queries = []
        doc_ids = list(documents.keys())
        
        query_templates = [
            "如何理解{keyword}的核心概念？",
            "{keyword}在实际应用中有哪些优势？",
            "关于{keyword}的技术发展趋势是什么？",
            "{keyword}面临的主要挑战有哪些？",
            "如何优化{keyword}的性能表现？"
        ]
        
        keywords = [
            "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
            "数据科学", "云计算", "区块链", "物联网", "网络安全",
            "算法优化", "系统架构", "技术创新", "数字化转型", "智能化应用"
        ]
        
        for i in range(num_queries):
            keyword = random.choice(keywords)
            query_template = random.choice(query_templates)
            query = query_template.format(keyword=keyword)
            
            # 随机选择相关文档（模拟真实场景中的相关性）
            expected_docs = random.sample(doc_ids, random.randint(1, 3))
            
            queries.append({
                'query': query,
                'expected_docs': expected_docs,
                'keyword': keyword
            })
        
        return queries

# 使用示例
if __name__ == "__main__":
    # 创建模拟RAG系统
    rag_system = MockRAGSystem(chunk_size=500, overlap_ratio=0.2)
    
    # 生成测试文档
    documents = MockDocumentGenerator.generate_test_documents(num_docs=5, doc_length=1500)
    
    # 添加文档到系统
    for doc_id, content in documents.items():
        rag_system.add_document(doc_id, content)
    
    # 处理所有文档
    chunks = rag_system.process_all_documents()
    print(f"生成了 {len(chunks)} 个文档块")
    
    # 获取统计信息
    stats = rag_system.get_chunk_statistics()
    print(f"平均块长度: {stats['avg_chunk_length']:.1f}")
    
    # 生成测试查询
    test_queries = MockDocumentGenerator.generate_test_queries(documents, num_queries=10)
    
    # 评估检索性能
    metrics = rag_system.evaluate_retrieval(test_queries)
    print(f"检索准确率: {metrics['accuracy']:.3f}")
    print(f"检索召回率: {metrics['recall']:.3f}")
    print(f"平均响应时间: {metrics['avg_response_time']:.2f}ms")