#!/usr/bin/env python3
"""
Lesson 05: Embedding与向量入库 - Qdrant数据库操作演示

本文件演示Qdrant向量数据库的基本操作，包括：
1. 数据库连接和集合管理
2. 向量数据的增删改查
3. 向量相似度搜索
4. 批量数据操作
5. 过滤查询和元数据管理

作者: RAG课程团队
日期: 2024-01-01
用途: Lesson 05课堂演示
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
from sentence_transformers import SentenceTransformer
import numpy as np


class QdrantDemo:
    """Qdrant向量数据库操作演示类"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        初始化Qdrant演示器
        
        Args:
            host: Qdrant服务器地址
            port: Qdrant服务器端口
        """
        self.host = host
        self.port = port
        self.client = None
        self.model = None
        self.collection_name = "lesson05_demo"
        
        self.connect_to_qdrant()
        self.load_embedding_model()
    
    def connect_to_qdrant(self) -> None:
        """连接到Qdrant数据库"""
        print(f"🔗 连接Qdrant数据库: {self.host}:{self.port}")
        
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # 测试连接
            collections = self.client.get_collections()
            print(f"✅ 数据库连接成功！")
            print(f"   - 服务器地址: {self.host}:{self.port}")
            print(f"   - 现有集合数量: {len(collections.collections)}")
            
            if collections.collections:
                print("   - 现有集合:")
                for collection in collections.collections:
                    print(f"     • {collection.name}")
                    
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            print("请确保Qdrant服务正在运行")
            raise
    
    def load_embedding_model(self) -> None:
        """加载向量化模型"""
        print(f"\n🧠 加载向量化模型...")
        try:
            self.model = SentenceTransformer('BAAI/bge-m3')
            print(f"✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def create_collection_demo(self) -> None:
        """创建集合演示"""
        print(f"\n📁 创建集合演示")
        print(f"集合名称: {self.collection_name}")
        
        try:
            # 检查集合是否已存在
            existing_collections = self.client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            
            if self.collection_name in collection_names:
                print(f"⚠️  集合 '{self.collection_name}' 已存在，先删除...")
                self.client.delete_collection(self.collection_name)
            
            # 创建新集合
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,  # bge-m3模型的向量维度
                    distance=Distance.COSINE  # 使用余弦距离
                )
            )
            
            print(f"✅ 集合创建成功！")
            print(f"   - 集合名称: {self.collection_name}")
            print(f"   - 向量维度: 768")
            print(f"   - 距离度量: 余弦距离")
            
            # 获取集合信息
            collection_info = self.client.get_collection(self.collection_name)
            print(f"   - 集合状态: {collection_info.status}")
            print(f"   - 向量数量: {collection_info.points_count}")
            
        except Exception as e:
            print(f"❌ 集合创建失败: {e}")
            raise
    
    def insert_vectors_demo(self, texts: List[str]) -> List[str]:
        """插入向量演示"""
        print(f"\n📥 向量插入演示")
        print(f"插入文本数量: {len(texts)}")
        
        # 显示部分文本
        print("插入文本示例:")
        for i, text in enumerate(texts[:3]):
            print(f"  {i+1}. '{text}'")
        if len(texts) > 3:
            print(f"  ... 还有{len(texts)-3}个文本")
        
        try:
            # 向量化文本
            print("🔄 正在进行向量化...")
            embeddings = self.model.encode(texts)
            
            # 构造Points
            points = []
            point_ids = []
            
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": text,
                        "index": i,
                        "category": "demo_data",
                        "timestamp": time.time(),
                        "length": len(text),
                        "language": "zh" if any('\u4e00' <= char <= '\u9fff' for char in text) else "en"
                    }
                )
                points.append(point)
            
            # 批量插入
            print("🔄 正在插入向量数据库...")
            start_time = time.time()
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            insert_time = time.time() - start_time
            
            print(f"✅ 向量插入成功！")
            print(f"   - 插入数量: {len(points)}")
            print(f"   - 操作状态: {result.status}")
            print(f"   - 插入耗时: {insert_time:.2f}秒")
            print(f"   - 平均速度: {len(points)/insert_time:.1f}向量/秒")
            
            # 验证插入结果
            collection_info = self.client.get_collection(self.collection_name)
            print(f"   - 集合中总向量数: {collection_info.points_count}")
            
            return point_ids
            
        except Exception as e:
            print(f"❌ 向量插入失败: {e}")
            raise
    
    def search_vectors_demo(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """向量搜索演示"""
        print(f"\n🔍 向量搜索演示")
        print(f"查询文本: '{query}'")
        print(f"返回结果数: {limit}")
        
        try:
            # 向量化查询文本
            print("🔄 正在向量化查询文本...")
            query_embedding = self.model.encode([query])[0]
            
            # 执行向量搜索
            print("🔄 正在执行向量搜索...")
            start_time = time.time()
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True,
                with_vectors=False  # 不返回向量数据以节省带宽
            )
            search_time = time.time() - start_time
            
            print(f"✅ 搜索完成！耗时: {search_time*1000:.2f}毫秒")
            print(f"\n🎯 搜索结果:")
            
            results = []
            for i, result in enumerate(search_results, 1):
                result_data = {
                    'rank': i,
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'category': result.payload.get('category', ''),
                    'language': result.payload.get('language', ''),
                    'length': result.payload.get('length', 0)
                }
                results.append(result_data)
                
                print(f"  排名 {i}:")
                print(f"    相似度: {result.score:.4f}")
                print(f"    文本: '{result.payload['text']}'")
                print(f"    语言: {result.payload.get('language', 'unknown')}")
                print(f"    长度: {result.payload.get('length', 0)}字符")
                print()
            
            return results
            
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            raise
    
    def filter_search_demo(self, query: str, category_filter: str = None, language_filter: str = None) -> List[Dict[str, Any]]:
        """过滤搜索演示"""
        print(f"\n🎯 过滤搜索演示")
        print(f"查询文本: '{query}'")
        if category_filter:
            print(f"类别过滤: {category_filter}")
        if language_filter:
            print(f"语言过滤: {language_filter}")
        
        try:
            # 向量化查询文本
            query_embedding = self.model.encode([query])[0]
            
            # 构建过滤条件
            filter_conditions = []
            if category_filter:
                filter_conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category_filter))
                )
            if language_filter:
                filter_conditions.append(
                    FieldCondition(key="language", match=MatchValue(value=language_filter))
                )
            
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # 执行过滤搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=5,
                with_payload=True
            )
            
            print(f"✅ 过滤搜索完成！找到 {len(search_results)} 个结果")
            
            results = []
            for i, result in enumerate(search_results, 1):
                result_data = {
                    'rank': i,
                    'score': result.score,
                    'text': result.payload['text'],
                    'category': result.payload.get('category', ''),
                    'language': result.payload.get('language', '')
                }
                results.append(result_data)
                
                print(f"  排名 {i}: (相似度: {result.score:.4f})")
                print(f"    文本: '{result.payload['text']}'")
                print(f"    类别: {result.payload.get('category', 'unknown')}")
                print(f"    语言: {result.payload.get('language', 'unknown')}")
                print()
            
            return results
            
        except Exception as e:
            print(f"❌ 过滤搜索失败: {e}")
            raise
    
    def update_vectors_demo(self, point_ids: List[str]) -> None:
        """更新向量演示"""
        print(f"\n🔄 向量更新演示")
        
        if not point_ids:
            print("⚠️  没有可更新的向量点")
            return
        
        try:
            # 选择第一个点进行更新演示
            update_id = point_ids[0]
            print(f"更新向量ID: {update_id}")
            
            # 获取原始数据
            original_point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[update_id],
                with_payload=True
            )[0]
            
            print(f"原始文本: '{original_point.payload['text']}'")
            
            # 更新文本和向量
            new_text = original_point.payload['text'] + " [已更新]"
            new_embedding = self.model.encode([new_text])[0]
            
            # 执行更新
            updated_point = PointStruct(
                id=update_id,
                vector=new_embedding.tolist(),
                payload={
                    **original_point.payload,
                    "text": new_text,
                    "updated": True,
                    "update_timestamp": time.time()
                }
            )
            
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[updated_point]
            )
            
            print(f"✅ 向量更新成功！")
            print(f"   - 更新状态: {result.status}")
            print(f"   - 新文本: '{new_text}'")
            
            # 验证更新结果
            updated_point_check = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[update_id],
                with_payload=True
            )[0]
            
            print(f"   - 验证文本: '{updated_point_check.payload['text']}'")
            print(f"   - 更新标记: {updated_point_check.payload.get('updated', False)}")
            
        except Exception as e:
            print(f"❌ 向量更新失败: {e}")
            raise
    
    def delete_vectors_demo(self, point_ids: List[str], delete_count: int = 1) -> None:
        """删除向量演示"""
        print(f"\n🗑️  向量删除演示")
        
        if len(point_ids) < delete_count:
            print(f"⚠️  可删除向量数量不足，需要{delete_count}个，实际{len(point_ids)}个")
            return
        
        try:
            # 选择要删除的向量
            delete_ids = point_ids[-delete_count:]
            print(f"删除向量数量: {delete_count}")
            print(f"删除向量IDs: {delete_ids}")
            
            # 获取删除前的集合信息
            collection_info_before = self.client.get_collection(self.collection_name)
            points_before = collection_info_before.points_count
            
            # 执行删除
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=delete_ids
            )
            
            print(f"✅ 向量删除成功！")
            print(f"   - 删除状态: {result.status}")
            
            # 验证删除结果
            collection_info_after = self.client.get_collection(self.collection_name)
            points_after = collection_info_after.points_count
            
            print(f"   - 删除前向量数: {points_before}")
            print(f"   - 删除后向量数: {points_after}")
            print(f"   - 实际删除数: {points_before - points_after}")
            
        except Exception as e:
            print(f"❌ 向量删除失败: {e}")
            raise
    
    def collection_stats_demo(self) -> Dict[str, Any]:
        """集合统计信息演示"""
        print(f"\n📊 集合统计信息")
        
        try:
            # 获取集合信息
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                'name': self.collection_name,
                'status': collection_info.status,
                'points_count': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance
            }
            
            print(f"集合名称: {stats['name']}")
            print(f"集合状态: {stats['status']}")
            print(f"向量数量: {stats['points_count']}")
            print(f"向量维度: {stats['vector_size']}")
            print(f"距离度量: {stats['distance']}")
            
            # 如果有向量，获取一些样本统计
            if stats['points_count'] > 0:
                # 随机获取一些点进行分析
                sample_points = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(10, stats['points_count']),
                    with_payload=True
                )[0]
                
                if sample_points:
                    categories = [point.payload.get('category', 'unknown') for point in sample_points]
                    languages = [point.payload.get('language', 'unknown') for point in sample_points]
                    text_lengths = [point.payload.get('length', 0) for point in sample_points]
                    
                    print(f"\n样本分析 (基于{len(sample_points)}个样本):")
                    print(f"  类别分布: {dict(zip(*np.unique(categories, return_counts=True)))}")
                    print(f"  语言分布: {dict(zip(*np.unique(languages, return_counts=True)))}")
                    print(f"  文本长度: 平均{np.mean(text_lengths):.1f}字符, 范围[{min(text_lengths)}-{max(text_lengths)}]")
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            raise
    
    def cleanup_demo(self) -> None:
        """清理演示数据"""
        print(f"\n🧹 清理演示数据")
        
        try:
            # 删除演示集合
            self.client.delete_collection(self.collection_name)
            print(f"✅ 演示集合 '{self.collection_name}' 已删除")
            
        except Exception as e:
            print(f"⚠️  清理失败: {e}")


def main():
    """主演示函数"""
    print("=" * 60)
    print("🚀 Lesson 05: Qdrant向量数据库操作演示")
    print("=" * 60)
    
    # 初始化演示器
    demo = QdrantDemo()
    
    # 1. 创建集合
    demo.create_collection_demo()
    
    # 2. 准备演示数据
    demo_texts = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器",
        "机器学习是实现人工智能的重要方法，让计算机从数据中学习",
        "深度学习是机器学习的一个子领域，使用神经网络进行学习",
        "自然语言处理让计算机能够理解和生成人类语言",
        "计算机视觉使机器能够识别和理解图像内容",
        "强化学习通过奖励机制训练智能体做出最优决策",
        "今天天气很好，适合出门散步和运动",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text"
    ]
    
    # 3. 插入向量数据
    point_ids = demo.insert_vectors_demo(demo_texts)
    
    # 4. 向量搜索演示
    search_queries = [
        "什么是机器学习？",
        "人工智能的应用",
        "天气怎么样？"
    ]
    
    for query in search_queries:
        demo.search_vectors_demo(query, limit=3)
    
    # 5. 过滤搜索演示
    demo.filter_search_demo("artificial intelligence", language_filter="en")
    demo.filter_search_demo("人工智能", language_filter="zh")
    
    # 6. 更新向量演示
    demo.update_vectors_demo(point_ids)
    
    # 7. 集合统计信息
    stats = demo.collection_stats_demo()
    
    # 8. 删除向量演示
    demo.delete_vectors_demo(point_ids, delete_count=2)
    
    # 9. 最终统计
    final_stats = demo.collection_stats_demo()
    
    print("\n" + "=" * 60)
    print("✅ Qdrant演示完成！")
    print("=" * 60)
    
    # 关键要点总结
    print("\n🎯 关键要点总结:")
    print("1. Qdrant支持高效的向量存储和检索")
    print("2. 支持丰富的元数据和过滤查询")
    print("3. 提供完整的CRUD操作接口")
    print("4. 余弦距离适合文本语义相似度计算")
    print("5. 批量操作可以提高数据处理效率")
    print("6. 合理的索引配置可以优化查询性能")
    
    # 询问是否清理数据
    cleanup = input("\n是否清理演示数据？(y/N): ").lower().strip()
    if cleanup == 'y':
        demo.cleanup_demo()
    else:
        print(f"演示数据保留在集合 '{demo.collection_name}' 中")


if __name__ == "__main__":
    main()