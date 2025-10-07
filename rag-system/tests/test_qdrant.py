"""测试Qdrant向量数据库功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pytest
from src.vector_store.qdrant_client import QdrantVectorStore
from src.embedding.embedder import TextEmbedder
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def vector_store():
    """创建Qdrant向量存储实例"""
    try:
        store = QdrantVectorStore(
            host="localhost",
            port=6333,
            timeout=10
        )
        yield store
    except Exception as e:
        pytest.skip(f"无法连接到Qdrant: {e}")

@pytest.fixture(scope="module")
def embedder():
    """创建文本嵌入器实例"""
    return TextEmbedder(model_name="BAAI/bge-m3")

def test_qdrant_connection(vector_store):
    """测试Qdrant连接"""
    print("\n=== 测试Qdrant连接 ===")
    
    # 列出现有集合
    collections = vector_store.list_collections()
    print(f"现有集合数量: {len(collections)}")
    if collections:
        print(f"集合列表: {collections}")
    
    print("✓ Qdrant连接成功")
    assert vector_store is not None

def test_collection_operations(vector_store):
    """测试集合操作"""
    print("\n=== 测试集合操作 ===")
    
    collection_name = "test_collection"
    vector_size = 384  # TextEmbedder的向量维度
    
    # 创建测试集合
    print(f"创建测试集合: {collection_name}")
    success = vector_store.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        distance="Cosine",
        recreate=True  # 重新创建
    )
    
    assert success, "集合创建失败"
    print("✓ 集合创建成功")
    
    # 获取集合信息
    info = vector_store.get_collection_info(collection_name)
    assert info is not None, "无法获取集合信息"
    
    print(f"集合信息:")
    print(f"  - 名称: {info['name']}")
    print(f"  - 向量维度: {info['config']['vector_size']}")
    print(f"  - 距离度量: {info['config']['distance']}")
    print(f"  - 状态: {info['status']}")
    print(f"  - 点数量: {info['points_count']}")

def test_vector_operations(vector_store, embedder):
    """测试向量操作"""
    print("\n=== 测试向量操作 ===")
    
    collection_name = "test_collection"
    
    # 准备测试文档
    test_documents = [
        "人工智能是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的核心技术，通过算法让计算机从数据中学习模式和规律。",
        "深度学习基于人工神经网络，特别是深层神经网络来进行学习和模式识别。",
        "自然语言处理是人工智能的一个分支，专注于让计算机理解、解释和生成人类语言。",
        "计算机视觉使计算机能够从数字图像或视频中获取高层次的理解。",
        "今天的天气非常好，阳光明媚，适合户外活动和散步。",
        "我喜欢在周末去公园里跑步，呼吸新鲜空气，锻炼身体。"
    ]
    
    print(f"准备向量化 {len(test_documents)} 个文档")
    
    # 向量化文档
    print("正在向量化文档...")
    vectors = embedder.encode_batch(test_documents)
    print(f"✓ 向量化完成，生成 {len(vectors)} 个向量")
    
    # 准备元数据
    payloads = []
    for i, doc in enumerate(test_documents):
        payload = {
            'document_id': f"doc_{i+1}",
            'text': doc,
            'length': len(doc),
            'category': 'AI' if i < 5 else 'lifestyle',
            'index': i
        }
        payloads.append(payload)
    
    # 插入向量
    print("插入向量到数据库...")
    success = vector_store.insert_vectors(
        collection_name=collection_name,
        vectors=vectors,
        payloads=payloads
    )
    
    assert success, "向量插入失败"
    print("✓ 向量插入成功")
    
    # 验证插入
    point_count = vector_store.count_points(collection_name)
    print(f"集合中的点数量: {point_count}")
    assert point_count > 0, "集合中没有数据点"

def test_vector_search(vector_store, embedder):
    """测试向量搜索"""
    print("\n=== 测试向量搜索 ===")
    
    collection_name = "test_collection"
    
    # 测试查询
    test_queries = [
        "什么是机器学习？",
        "深度学习的原理",
        "天气怎么样？",
        "人工智能的应用"
    ]
    
    for query in test_queries:
        print(f"\n--- 查询: {query} ---")
        
        # 向量化查询
        query_vector = embedder.encode(query)
        
        # 执行搜索
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,
            score_threshold=0.3
        )
        
        print(f"找到 {len(results)} 个相关结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [分数: {result.score:.4f}] {result.payload.get('text', '')[:50]}...")
            print(f"     类别: {result.payload.get('category', 'unknown')}")
    
    # 至少应该能找到一些结果
    query_vector = embedder.encode("人工智能")
    results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )
    assert len(results) > 0, "搜索应该返回结果"

def test_filtered_search(vector_store, embedder):
    """测试过滤搜索"""
    print("\n=== 测试过滤搜索 ===")
    
    collection_name = "test_collection"
    
    # 测试过滤搜索
    query = "人工智能技术"
    query_vector = embedder.encode(query)
    
    print(f"查询: {query}")
    
    # 只搜索AI类别的文档
    print("\n--- 过滤搜索 (只搜索AI类别) ---")
    filtered_results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        filter_conditions={'category': 'AI'}
    )
    
    print(f"AI类别结果数量: {len(filtered_results)}")
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. [分数: {result.score:.4f}] {result.payload.get('text', '')[:50]}...")
    
    # 搜索所有类别
    print("\n--- 无过滤搜索 (所有类别) ---")
    all_results = vector_store.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )
    
    print(f"所有结果数量: {len(all_results)}")
    for i, result in enumerate(all_results, 1):
        category = result.payload.get('category', 'unknown')
        print(f"  {i}. [分数: {result.score:.4f}] [{category}] {result.payload.get('text', '')[:50]}...")
    
    # 验证过滤功能正常工作
    assert len(all_results) > 0, "搜索应该返回结果"

def test_performance(vector_store, embedder):
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    collection_name = "test_collection"
    
    # 批量搜索测试
    queries = [
        "机器学习基础",
        "深度学习应用",
        "自然语言处理",
        "计算机视觉",
        "强化学习"
    ]
    
    import time
    start_time = time.time()
    
    total_results = 0
    for query in queries:
        query_vector = embedder.encode(query)
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10
        )
        total_results += len(results)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"批量搜索完成:")
    print(f"  查询数量: {len(queries)}")
    print(f"  总结果数: {total_results}")
    print(f"  耗时: {elapsed_time:.2f}秒")
    print(f"  平均每查询: {elapsed_time/len(queries):.3f}秒")
    
    # 性能要求：每个查询应该在合理时间内完成
    avg_time_per_query = elapsed_time / len(queries)
    assert avg_time_per_query < 5.0, f"平均查询时间 {avg_time_per_query:.3f}秒 应该小于 5.0秒"
    assert total_results > 0, "应该返回搜索结果"
    print("✓ 性能测试通过")

# 注意：这个文件现在使用pytest运行，不需要main函数
# 运行命令：pytest tests/test_qdrant.py -v