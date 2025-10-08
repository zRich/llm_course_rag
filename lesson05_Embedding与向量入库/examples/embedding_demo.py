#!/usr/bin/env python3
"""
Lesson 05: Embedding与向量入库 - 向量化演示示例

本文件演示bge-m3模型的基本使用方法，包括：
1. 模型加载和初始化
2. 单文本向量化
3. 批量文本向量化
4. 向量相似度计算
5. 性能测试和优化

作者: RAG课程团队
日期: 2024-01-01
用途: Lesson 05课堂演示
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class BGEEmbeddingDemo:
    """bge-m3模型向量化演示类"""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        """
        初始化向量化演示器
        
        Args:
            model_name: 模型名称，默认使用bge-m3
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """加载bge-m3模型"""
        print(f"正在加载模型: {self.model_name}")
        print("首次加载可能需要下载模型文件，请耐心等待...")
        
        start_time = time.time()
        try:
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            print(f"✅ 模型加载成功！耗时: {load_time:.2f}秒")
            print(f"📊 模型信息:")
            print(f"   - 模型名称: {self.model_name}")
            print(f"   - 最大序列长度: {self.model.max_seq_length}")
            print(f"   - 向量维度: 768")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        单文本向量化演示
        
        Args:
            text: 输入文本
            
        Returns:
            向量化结果
        """
        print(f"\n🔄 单文本向量化演示")
        print(f"输入文本: '{text}'")
        
        start_time = time.time()
        embedding = self.model.encode(text)
        encode_time = time.time() - start_time
        
        print(f"✅ 向量化完成！")
        print(f"   - 向量维度: {embedding.shape}")
        print(f"   - 向量类型: {type(embedding)}")
        print(f"   - 处理耗时: {encode_time*1000:.2f}毫秒")
        print(f"   - 向量前5维: {embedding[:5]}")
        print(f"   - 向量范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
        
        return embedding
    
    def encode_batch_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量文本向量化演示
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量化结果矩阵
        """
        print(f"\n🔄 批量文本向量化演示")
        print(f"文本数量: {len(texts)}")
        print(f"批处理大小: {batch_size}")
        
        # 显示部分输入文本
        print("输入文本示例:")
        for i, text in enumerate(texts[:3]):
            print(f"  {i+1}. '{text}'")
        if len(texts) > 3:
            print(f"  ... 还有{len(texts)-3}个文本")
        
        start_time = time.time()
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        encode_time = time.time() - start_time
        
        print(f"✅ 批量向量化完成！")
        print(f"   - 输出形状: {embeddings.shape}")
        print(f"   - 总耗时: {encode_time:.2f}秒")
        print(f"   - 平均每文本: {encode_time/len(texts)*1000:.2f}毫秒")
        print(f"   - 处理速度: {len(texts)/encode_time:.1f}文本/秒")
        
        return embeddings
    
    def calculate_similarity_matrix(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算文本相似度矩阵演示
        
        Args:
            texts: 文本列表
            
        Returns:
            (embeddings, similarity_matrix)
        """
        print(f"\n🔄 相似度计算演示")
        print("计算文本间的语义相似度...")
        
        # 批量向量化
        embeddings = self.model.encode(texts)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        print(f"✅ 相似度计算完成！")
        print(f"相似度矩阵形状: {similarity_matrix.shape}")
        
        # 显示相似度结果
        print("\n📊 相似度矩阵:")
        print("文本列表:")
        for i, text in enumerate(texts):
            print(f"  {i}: '{text}'")
        
        print("\n相似度详情:")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                similarity = similarity_matrix[i][j]
                print(f"  文本{i} ↔ 文本{j}: {similarity:.4f}")
        
        return embeddings, similarity_matrix
    
    def semantic_search_demo(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        语义搜索演示
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k结果
            
        Returns:
            搜索结果列表 [(index, document, score), ...]
        """
        print(f"\n🔍 语义搜索演示")
        print(f"查询文本: '{query}'")
        print(f"文档数量: {len(documents)}")
        print(f"返回top-{top_k}结果")
        
        # 向量化查询和文档
        query_embedding = self.model.encode([query])
        doc_embeddings = self.model.encode(documents)
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        print(f"\n🎯 搜索结果:")
        for rank, idx in enumerate(top_indices, 1):
            score = similarities[idx]
            document = documents[idx]
            results.append((idx, document, score))
            print(f"  排名{rank}: (相似度: {score:.4f})")
            print(f"    文档{idx}: '{document}'")
        
        return results
    
    def performance_benchmark(self, text_counts: List[int] = [10, 50, 100, 500]) -> Dict[int, Dict[str, float]]:
        """
        性能基准测试
        
        Args:
            text_counts: 测试的文本数量列表
            
        Returns:
            性能测试结果
        """
        print(f"\n⚡ 性能基准测试")
        print("测试不同文本数量下的向量化性能...")
        
        results = {}
        
        for count in text_counts:
            print(f"\n测试 {count} 个文本:")
            
            # 生成测试文本
            test_texts = [f"这是第{i}个测试文本，内容关于人工智能和机器学习技术的发展应用。" for i in range(count)]
            
            # 测试不同批处理大小
            batch_sizes = [16, 32, 64] if count >= 64 else [16, 32]
            
            best_time = float('inf')
            best_batch_size = 16
            
            for batch_size in batch_sizes:
                start_time = time.time()
                embeddings = self.model.encode(test_texts, batch_size=batch_size)
                encode_time = time.time() - start_time
                
                if encode_time < best_time:
                    best_time = encode_time
                    best_batch_size = batch_size
                
                print(f"  批大小{batch_size}: {encode_time:.2f}秒 ({count/encode_time:.1f}文本/秒)")
            
            results[count] = {
                'best_time': best_time,
                'best_batch_size': best_batch_size,
                'texts_per_second': count / best_time,
                'ms_per_text': best_time / count * 1000
            }
            
            print(f"  ✅ 最佳配置: 批大小{best_batch_size}, {best_time:.2f}秒")
        
        # 性能总结
        print(f"\n📈 性能总结:")
        for count, result in results.items():
            print(f"  {count}文本: {result['texts_per_second']:.1f}文本/秒 "
                  f"({result['ms_per_text']:.2f}ms/文本)")
        
        return results


def main():
    """主演示函数"""
    print("=" * 60)
    print("🚀 Lesson 05: bge-m3向量化演示")
    print("=" * 60)
    
    # 初始化演示器
    demo = BGEEmbeddingDemo()
    
    # 1. 单文本向量化演示
    single_text = "人工智能技术正在改变世界"
    embedding = demo.encode_single_text(single_text)
    
    # 2. 批量文本向量化演示
    batch_texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是实现人工智能的重要方法",
        "深度学习是机器学习的一个子领域",
        "自然语言处理是AI的重要应用方向",
        "计算机视觉让机器能够理解图像",
        "今天天气很好，适合出门散步"
    ]
    batch_embeddings = demo.encode_batch_texts(batch_texts)
    
    # 3. 相似度计算演示
    similarity_texts = [
        "人工智能技术发展迅速",
        "AI技术进步很快",
        "今天天气很好",
        "机器学习是AI的重要分支"
    ]
    embeddings, similarity_matrix = demo.calculate_similarity_matrix(similarity_texts)
    
    # 4. 语义搜索演示
    query = "什么是机器学习？"
    documents = [
        "机器学习是一种人工智能技术，让计算机从数据中学习",
        "深度学习使用神经网络进行复杂的模式识别",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够识别和理解图像内容",
        "今天的天气预报显示会有小雨",
        "数据科学结合统计学和计算机科学来分析数据"
    ]
    search_results = demo.semantic_search_demo(query, documents)
    
    # 5. 性能基准测试
    performance_results = demo.performance_benchmark([10, 50, 100])
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    
    # 关键要点总结
    print("\n🎯 关键要点总结:")
    print("1. bge-m3模型支持中英文混合文本向量化")
    print("2. 向量维度固定为768维，适合大多数应用场景")
    print("3. 批处理可以显著提高处理效率")
    print("4. 余弦相似度是衡量文本语义相似性的有效方法")
    print("5. 语义搜索比关键词匹配更能理解用户意图")
    print("6. 合适的批处理大小可以优化性能")


if __name__ == "__main__":
    main()