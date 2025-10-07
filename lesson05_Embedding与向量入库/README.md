# Lesson 05: Embedding与向量入库

## 课程目标

学习使用BGE-M3模型进行文本向量化，并将向量存储到Qdrant向量数据库中，为后续的语义检索奠定基础。

## 学习内容

### 1. 文本向量化技术
- **BGE-M3模型**：多语言、多粒度的向量化模型
- **向量维度**：理解向量维度对检索效果的影响
- **批量处理**：高效处理大量文档的向量化
- **向量归一化**：提升检索精度的技术

### 2. Qdrant向量数据库
- **集合管理**：创建和配置向量集合
- **向量存储**：高效存储和索引向量数据
- **元数据过滤**：支持复合查询的元数据设计
- **性能优化**：索引配置和查询优化

### 3. 向量化服务设计
- **异步处理**：支持大规模文档的异步向量化
- **错误处理**：向量化失败的重试和恢复机制
- **进度跟踪**：实时监控向量化进度
- **质量评估**：向量质量的评估指标

## 技术栈

- **向量化模型**：BGE-M3 (BAAI/bge-m3)
- **向量数据库**：Qdrant
- **深度学习框架**：Transformers, PyTorch
- **Web框架**：FastAPI
- **数据库**：PostgreSQL (元数据), Qdrant (向量)
- **异步处理**：Celery + Redis

## 实践操作

### 1. BGE-M3模型集成
```python
# 模型加载和向量化
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
vectors = model.encode(texts, normalize_embeddings=True)
```

### 2. Qdrant集合配置
```python
# 创建向量集合
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)
```

### 3. 向量入库流程
```python
# 批量向量入库
points = [
    PointStruct(
        id=chunk.id,
        vector=vector.tolist(),
        payload={
            "document_id": chunk.document_id,
            "text": chunk.text,
            "metadata": chunk.metadata
        }
    )
    for chunk, vector in zip(chunks, vectors)
]

client.upsert(collection_name="documents", points=points)
```

## 项目结构

```
lesson05_Embedding与向量入库/
├── README.md                 # 课程说明
└── practice/                 # 实践项目
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              # FastAPI应用入口
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py        # 配置管理
    │   │   ├── database.py      # 数据库连接
    │   │   └── exceptions.py    # 异常定义
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── embedding.py     # 向量化模型
    │   │   └── vector.py        # 向量数据模型
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── embedding.py     # 向量化服务
    │   │   ├── qdrant.py        # Qdrant客户端
    │   │   └── vector_store.py  # 向量存储服务
    │   ├── api/
    │   │   ├── __init__.py
    │   │   └── v1/
    │   │       ├── __init__.py
    │   │       ├── embeddings.py # 向量化API
    │   │       └── vectors.py    # 向量查询API
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   ├── embedding.py     # 向量化请求响应
    │   │   └── vector.py        # 向量查询请求响应
    │   └── tasks/
    │       ├── __init__.py
    │       └── embedding.py     # 异步向量化任务
    ├── tests/
    │   ├── __init__.py
    │   ├── test_embedding.py    # 向量化测试
    │   ├── test_qdrant.py       # Qdrant测试
    │   └── test_vector_store.py # 向量存储测试
    ├── scripts/
    │   ├── start.sh             # 启动脚本
    │   ├── test.sh              # 测试脚本
    │   └── setup_qdrant.sh      # Qdrant初始化
    ├── requirements.txt         # Python依赖
    ├── Dockerfile              # Docker配置
    ├── docker-compose.yml      # 服务编排
    └── .env.example            # 环境变量示例
```

## 核心概念

### 1. 向量化策略

**文档级向量化**
- 整个文档生成一个向量
- 适用于文档级别的相似性检索
- 计算效率高，存储成本低

**分块级向量化**
- 每个文档分块生成独立向量
- 支持细粒度的语义检索
- 检索精度高，适合问答场景

**混合向量化**
- 同时保存文档级和分块级向量
- 支持多层次的检索策略
- 灵活性高，适应不同场景

### 2. 向量质量指标

**向量分布**
- 向量空间的分布均匀性
- 避免向量聚集和稀疏区域

**语义一致性**
- 相似内容的向量距离
- 不同内容的向量区分度

**检索效果**
- Top-K检索的准确率
- 语义匹配的召回率

### 3. 性能优化

**批量处理**
- 批量向量化提升效率
- 合理的批次大小设置

**内存管理**
- 大模型的内存优化
- 向量数据的内存控制

**并发处理**
- 多进程向量化
- 异步任务队列

## 学习成果

完成本课程后，你将掌握：

1. **BGE-M3模型使用**：加载模型、文本向量化、批量处理
2. **Qdrant操作**：集合管理、向量存储、元数据查询
3. **向量化服务**：异步处理、错误恢复、进度监控
4. **性能优化**：批量处理、内存管理、并发控制
5. **质量评估**：向量质量指标、检索效果评估

## 下一步

完成向量入库后，下一课将学习：
- **Lesson 06**：最小检索与生成 - 实现基础的RAG问答系统
- 向量检索算法
- 上下文拼接策略
- LLM生成优化

## 注意事项

1. **模型下载**：BGE-M3模型较大，首次使用需要下载
2. **GPU支持**：建议使用GPU加速向量化过程
3. **内存需求**：大批量处理需要充足的内存
4. **网络连接**：Qdrant服务需要稳定的网络连接
5. **数据备份**：重要向量数据需要定期备份