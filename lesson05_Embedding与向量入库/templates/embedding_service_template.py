#!/usr/bin/env python3
"""
Lesson 05: Embedding与向量入库 - 向量化服务模板

本模板提供向量化服务的基础框架，学生需要完成以下任务：
1. 实现文本向量化功能
2. 实现Qdrant数据库操作
3. 构建完整的向量化服务
4. 添加性能监控和错误处理

作者: RAG课程团队
日期: 2024-01-01
用途: Lesson 05 Exercise模板
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# TODO: 导入必要的库
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """向量化请求数据结构"""
    text: str
    metadata: Dict[str, Any] = None
    source: str = "unknown"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingResponse:
    """向量化响应数据结构"""
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchRequest:
    """搜索请求数据结构"""
    query: str
    limit: int = 5
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbeddingModel(ABC):
    """向量化模型抽象基类"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """向量化文本列表"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass


class BGEEmbeddingModel(EmbeddingModel):
    """BGE-M3向量化模型实现"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        # TODO: Exercise 1 - 实现模型加载
        # 提示：使用SentenceTransformer加载bge-m3模型
        # self.model = SentenceTransformer(self.model_name)
        logger.info(f"TODO: 加载模型 {self.model_name}")
        pass
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """向量化文本列表"""
        # TODO: Exercise 1 - 实现文本向量化
        # 提示：使用self.model.encode()方法
        # 返回格式：List[List[float]]
        logger.info(f"TODO: 向量化 {len(texts)} 个文本")
        
        # 临时返回随机向量用于测试
        import random
        return [[random.random() for _ in range(768)] for _ in texts]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        # TODO: Exercise 1 - 返回正确的向量维度
        # BGE-M3模型的向量维度是768
        return 768


class VectorDatabase(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    def insert_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """插入向量"""
        pass
    
    @abstractmethod
    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      limit: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """搜索向量"""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        pass


class QdrantDatabase(VectorDatabase):
    """Qdrant向量数据库实现"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self):
        """连接数据库"""
        # TODO: Exercise 2 - 实现Qdrant连接
        # 提示：使用QdrantClient连接数据库
        # self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"TODO: 连接Qdrant数据库 {self.host}:{self.port}")
        pass
    
    def create_collection(self, name: str, dimension: int) -> bool:
        """创建集合"""
        # TODO: Exercise 2 - 实现集合创建
        # 提示：
        # 1. 检查集合是否已存在，如果存在则删除
        # 2. 使用client.create_collection()创建新集合
        # 3. 配置向量参数：size=dimension, distance=Distance.COSINE
        logger.info(f"TODO: 创建集合 {name}, 维度: {dimension}")
        return True
    
    def insert_vectors(self, collection_name: str, vectors: List[Dict[str, Any]]) -> bool:
        """插入向量"""
        # TODO: Exercise 2 - 实现向量插入
        # 提示：
        # 1. 将向量数据转换为PointStruct格式
        # 2. 使用client.upsert()批量插入
        # 3. 处理插入结果和异常
        logger.info(f"TODO: 插入 {len(vectors)} 个向量到集合 {collection_name}")
        return True
    
    def search_vectors(self, collection_name: str, query_vector: List[float], 
                      limit: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """搜索向量"""
        # TODO: Exercise 2 - 实现向量搜索
        # 提示：
        # 1. 构建搜索过滤条件（如果有）
        # 2. 使用client.search()执行搜索
        # 3. 处理搜索结果并返回标准格式
        logger.info(f"TODO: 在集合 {collection_name} 中搜索向量")
        
        # 临时返回空结果
        return []
    
    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        # TODO: Exercise 2 - 实现集合删除
        # 提示：使用client.delete_collection()
        logger.info(f"TODO: 删除集合 {name}")
        return True
    
    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """获取集合信息"""
        # TODO: Exercise 2 - 实现集合信息获取
        # 提示：使用client.get_collection()
        logger.info(f"TODO: 获取集合信息 {name}")
        return {"name": name, "points_count": 0, "status": "unknown"}


class EmbeddingService:
    """向量化服务主类"""
    
    def __init__(self, 
                 model: EmbeddingModel,
                 database: VectorDatabase,
                 collection_name: str = "embedding_service"):
        self.model = model
        self.database = database
        self.collection_name = collection_name
        self.stats = {
            "total_embeddings": 0,
            "total_searches": 0,
            "total_processing_time": 0.0
        }
        
        self._initialize_collection()
    
    def _initialize_collection(self):
        """初始化集合"""
        # TODO: Exercise 3 - 实现集合初始化
        # 提示：使用database.create_collection()创建集合
        dimension = self.model.get_dimension()
        logger.info(f"TODO: 初始化集合 {self.collection_name}, 维度: {dimension}")
        pass
    
    def embed_text(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """向量化单个文本"""
        # TODO: Exercise 3 - 实现单文本向量化
        # 提示：
        # 1. 记录开始时间
        # 2. 使用model.encode()向量化文本
        # 3. 计算处理时间
        # 4. 构建并返回EmbeddingResponse
        
        start_time = time.time()
        
        # 临时实现
        vector = self.model.encode([request.text])[0]
        processing_time = time.time() - start_time
        
        response = EmbeddingResponse(
            id=str(uuid.uuid4()),
            vector=vector,
            text=request.text,
            metadata=request.metadata,
            processing_time=processing_time
        )
        
        self.stats["total_embeddings"] += 1
        self.stats["total_processing_time"] += processing_time
        
        return response
    
    def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """批量向量化文本"""
        # TODO: Exercise 3 - 实现批量向量化
        # 提示：
        # 1. 提取所有文本
        # 2. 批量向量化
        # 3. 构建响应列表
        logger.info(f"TODO: 批量向量化 {len(requests)} 个文本")
        
        responses = []
        for request in requests:
            response = self.embed_text(request)
            responses.append(response)
        
        return responses
    
    def store_embeddings(self, embeddings: List[EmbeddingResponse]) -> bool:
        """存储向量到数据库"""
        # TODO: Exercise 3 - 实现向量存储
        # 提示：
        # 1. 将EmbeddingResponse转换为数据库格式
        # 2. 使用database.insert_vectors()存储
        # 3. 处理存储结果
        logger.info(f"TODO: 存储 {len(embeddings)} 个向量")
        
        # 转换格式
        vectors = []
        for embedding in embeddings:
            vector_data = {
                "id": embedding.id,
                "vector": embedding.vector,
                "payload": {
                    "text": embedding.text,
                    "metadata": embedding.metadata,
                    "processing_time": embedding.processing_time
                }
            }
            vectors.append(vector_data)
        
        return self.database.insert_vectors(self.collection_name, vectors)
    
    def search_similar(self, request: SearchRequest) -> List[SearchResult]:
        """搜索相似向量"""
        # TODO: Exercise 3 - 实现相似性搜索
        # 提示：
        # 1. 向量化查询文本
        # 2. 使用database.search_vectors()搜索
        # 3. 转换搜索结果格式
        
        start_time = time.time()
        
        # 向量化查询
        query_vector = self.model.encode([request.query])[0]
        
        # 搜索向量
        search_results = self.database.search_vectors(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=request.limit,
            filters=request.filters
        )
        
        # 转换结果格式
        results = []
        for result in search_results:
            search_result = SearchResult(
                id=result.get("id", ""),
                score=result.get("score", 0.0),
                text=result.get("text", ""),
                metadata=result.get("metadata", {})
            )
            results.append(search_result)
        
        processing_time = time.time() - start_time
        self.stats["total_searches"] += 1
        self.stats["total_processing_time"] += processing_time
        
        logger.info(f"搜索完成，找到 {len(results)} 个结果，耗时 {processing_time:.3f}秒")
        return results
    
    def process_documents(self, texts: List[str], sources: List[str] = None) -> Dict[str, Any]:
        """处理文档的完整流程"""
        # TODO: Exercise 3 - 实现完整的文档处理流程
        # 提示：
        # 1. 构建EmbeddingRequest列表
        # 2. 批量向量化
        # 3. 存储向量
        # 4. 返回处理结果统计
        
        logger.info(f"TODO: 处理 {len(texts)} 个文档")
        
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]
        
        # 构建请求
        requests = []
        for i, text in enumerate(texts):
            request = EmbeddingRequest(
                text=text,
                source=sources[i] if i < len(sources) else f"doc_{i}",
                metadata={"index": i, "timestamp": time.time()}
            )
            requests.append(request)
        
        # 批量处理
        embeddings = self.embed_batch(requests)
        
        # 存储向量
        storage_success = self.store_embeddings(embeddings)
        
        return {
            "success": storage_success,
            "processed_count": len(embeddings),
            "total_processing_time": sum(e.processing_time for e in embeddings),
            "average_processing_time": sum(e.processing_time for e in embeddings) / len(embeddings)
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        collection_info = self.database.get_collection_info(self.collection_name)
        
        return {
            "service_stats": self.stats,
            "collection_info": collection_info,
            "model_dimension": self.model.get_dimension()
        }
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理服务资源")
        self.database.delete_collection(self.collection_name)


# Exercise任务函数
def exercise_basic_embedding():
    """Exercise 1: 基础向量化实现"""
    print("\n" + "="*50)
    print("🎯 Exercise 1: 基础向量化实现")
    print("="*50)
    
    print("任务：实现BGEEmbeddingModel类的核心方法")
    print("1. 完成_load_model()方法")
    print("2. 完成encode()方法")
    print("3. 确认get_dimension()返回正确值")
    
    # 测试代码
    model = BGEEmbeddingModel()
    
    test_texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习让计算机从数据中学习",
        "深度学习使用神经网络进行学习"
    ]
    
    print(f"\n测试向量化 {len(test_texts)} 个文本...")
    vectors = model.encode(test_texts)
    
    print(f"✅ 向量化完成")
    print(f"   - 向量数量: {len(vectors)}")
    print(f"   - 向量维度: {len(vectors[0]) if vectors else 0}")
    print(f"   - 模型维度: {model.get_dimension()}")
    
    # TODO: 学生需要确保这里的输出是正确的


def exercise_qdrant_operations():
    """Exercise 2: Qdrant数据库操作"""
    print("\n" + "="*50)
    print("🎯 Exercise 2: Qdrant数据库操作")
    print("="*50)
    
    print("任务：实现QdrantDatabase类的核心方法")
    print("1. 完成_connect()方法")
    print("2. 完成create_collection()方法")
    print("3. 完成insert_vectors()方法")
    print("4. 完成search_vectors()方法")
    
    # 测试代码
    db = QdrantDatabase()
    
    # 测试集合创建
    collection_name = "exercise_test"
    print(f"\n测试创建集合: {collection_name}")
    success = db.create_collection(collection_name, 768)
    print(f"创建结果: {success}")
    
    # 测试向量插入
    test_vectors = [
        {
            "id": "test_1",
            "vector": [0.1] * 768,
            "payload": {"text": "测试文本1", "source": "test"}
        },
        {
            "id": "test_2", 
            "vector": [0.2] * 768,
            "payload": {"text": "测试文本2", "source": "test"}
        }
    ]
    
    print(f"\n测试插入 {len(test_vectors)} 个向量...")
    success = db.insert_vectors(collection_name, test_vectors)
    print(f"插入结果: {success}")
    
    # 测试向量搜索
    query_vector = [0.15] * 768
    print(f"\n测试向量搜索...")
    results = db.search_vectors(collection_name, query_vector, limit=2)
    print(f"搜索结果数量: {len(results)}")
    
    # 获取集合信息
    info = db.get_collection_info(collection_name)
    print(f"集合信息: {info}")
    
    # TODO: 学生需要确保所有操作都能正常工作


def exercise_complete_service():
    """Exercise 3: 完整向量化服务"""
    print("\n" + "="*50)
    print("🎯 Exercise 3: 完整向量化服务")
    print("="*50)
    
    print("任务：构建完整的向量化服务")
    print("1. 集成向量化模型和数据库")
    print("2. 实现文档处理流程")
    print("3. 实现相似性搜索")
    print("4. 添加性能监控")
    
    # 初始化服务
    model = BGEEmbeddingModel()
    database = QdrantDatabase()
    service = EmbeddingService(model, database, "exercise_service")
    
    # 测试文档处理
    test_documents = [
        "人工智能正在改变我们的世界",
        "机器学习是AI的重要分支",
        "深度学习在图像识别中表现出色",
        "自然语言处理让机器理解人类语言",
        "计算机视觉帮助机器看懂图像"
    ]
    
    print(f"\n处理 {len(test_documents)} 个文档...")
    result = service.process_documents(test_documents)
    print(f"处理结果: {result}")
    
    # 测试搜索
    search_queries = [
        "什么是人工智能？",
        "机器学习的应用",
        "深度学习技术"
    ]
    
    for query in search_queries:
        print(f"\n搜索查询: '{query}'")
        search_request = SearchRequest(query=query, limit=3)
        results = service.search_similar(search_request)
        
        print(f"搜索结果 ({len(results)}个):")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {result.score:.4f}")
            print(f"     文本: '{result.text}'")
    
    # 获取服务统计
    stats = service.get_service_stats()
    print(f"\n服务统计: {stats}")
    
    # TODO: 学生需要确保整个服务流程正常工作
    
    return service


def main():
    """主函数 - 运行所有Exercise"""
    print("🚀 Lesson 05: Embedding与向量入库 - Exercise模板")
    print("="*60)
    
    print("\n📋 Exercise任务列表:")
    print("1. 基础向量化实现 (BGE-M3模型)")
    print("2. Qdrant数据库操作")
    print("3. 完整向量化服务构建")
    
    print("\n⚠️  注意事项:")
    print("- 请确保已安装必要的依赖包")
    print("- 请确保Qdrant服务正在运行")
    print("- 完成TODO标记的代码实现")
    print("- 测试每个功能模块")
    
    try:
        # Exercise 1
        exercise_basic_embedding()
        
        # Exercise 2  
        exercise_qdrant_operations()
        
        # Exercise 3
        service = exercise_complete_service()
        
        print("\n" + "="*60)
        print("🎉 所有Exercise完成！")
        print("="*60)
        
        # 清理资源
        cleanup = input("\n是否清理测试数据？(y/N): ").lower().strip()
        if cleanup == 'y':
            service.cleanup()
            print("✅ 测试数据已清理")
        
    except Exception as e:
        print(f"❌ Exercise执行失败: {e}")
        print("请检查代码实现和环境配置")


if __name__ == "__main__":
    main()