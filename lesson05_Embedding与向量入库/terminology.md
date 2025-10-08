# 第5课：Embedding与向量入库 - 术语与概念定义

## 核心术语

### Embedding相关术语

#### Embedding / 嵌入
- **中文定义**：将离散的符号（如词汇、句子）映射到连续向量空间的技术
- **英文定义**：A mapping of discrete symbols to vectors of real numbers in a continuous vector space
- **关键特点**：
  - 保持语义相似性
  - 支持数值计算
  - 降维表示
- **数学表示**：f: V → ℝⁿ，其中V是词汇表，ℝⁿ是n维实数空间

#### Text Embedding / 文本嵌入
- **中文定义**：将文本转换为固定长度数值向量的过程
- **英文定义**：The process of converting text into fixed-length numerical vectors
- **实现方式**：
  - 词袋模型 (Bag of Words)
  - TF-IDF
  - Word2Vec
  - Transformer模型
- **应用场景**：文本分类、相似度计算、信息检索

#### Sentence Embedding / 句子嵌入
- **中文定义**：将整个句子编码为单一向量表示
- **英文定义**：Encoding entire sentences into single vector representations
- **技术方法**：
  - 平均词向量
  - 句子级Transformer
  - 专门的句子编码器
- **优势**：捕获句子级语义信息

#### Contextual Embedding / 上下文嵌入
- **中文定义**：根据上下文动态生成的词向量表示
- **英文定义**：Word representations that vary based on context
- **代表模型**：BERT、GPT、RoBERTa
- **特点**：
  - 同一词在不同上下文中有不同向量
  - 更好的语义理解能力
  - 处理一词多义问题

### 向量空间术语

#### Vector Space / 向量空间
- **中文定义**：由向量组成的数学结构，支持向量加法和标量乘法
- **英文定义**：A mathematical structure formed by a collection of vectors
- **性质**：
  - 线性组合封闭性
  - 支持距离和角度计算
  - 可进行几何操作
- **在NLP中的意义**：语义空间的数学基础

#### Dimensionality / 维度
- **中文定义**：向量空间中向量的分量数量
- **英文定义**：The number of components in a vector
- **常见维度**：
  - Word2Vec: 100-300维
  - BERT: 768维
  - GPT: 1024-4096维
- **权衡考虑**：表达能力 vs 计算复杂度

#### Dense Vector / 密集向量
- **中文定义**：大部分元素非零的向量表示
- **英文定义**：Vector representation where most elements are non-zero
- **对比**：与稀疏向量(Sparse Vector)相对
- **优势**：
  - 更好的泛化能力
  - 更紧凑的表示
  - 更适合神经网络处理

#### Sparse Vector / 稀疏向量
- **中文定义**：大部分元素为零的向量表示
- **英文定义**：Vector representation where most elements are zero
- **典型例子**：TF-IDF向量、词袋模型
- **特点**：
  - 可解释性强
  - 存储效率高（压缩后）
  - 计算相对简单

## 相似度计算术语

### Cosine Similarity / 余弦相似度
- **中文定义**：通过计算两个向量夹角的余弦值来衡量相似度
- **英文定义**：A measure of similarity between two vectors based on the cosine of the angle between them
- **计算公式**：
  ```
  cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
  ```
- **取值范围**：[-1, 1]，1表示完全相似，-1表示完全相反
- **优势**：不受向量长度影响，适合文本相似度计算

### Euclidean Distance / 欧几里得距离
- **中文定义**：两点间的直线距离
- **英文定义**：The straight-line distance between two points in space
- **计算公式**：
  ```
  euclidean_distance(A, B) = √(Σ(Ai - Bi)²)
  ```
- **特点**：
  - 直观易理解
  - 受向量长度影响
  - 高维空间中可能失效

### Dot Product / 点积
- **中文定义**：两个向量对应元素乘积的和
- **英文定义**：The sum of the products of corresponding elements of two vectors
- **计算公式**：A · B = Σ(Ai × Bi)
- **几何意义**：反映向量的相似程度和长度信息
- **应用**：注意力机制、相似度计算

### Manhattan Distance / 曼哈顿距离
- **中文定义**：两点间各坐标差值绝对值的和
- **英文定义**：The sum of absolute differences between corresponding coordinates
- **计算公式**：manhattan_distance(A, B) = Σ|Ai - Bi|
- **特点**：对异常值不敏感，计算简单

## 向量数据库术语

### Vector Database / 向量数据库
- **中文定义**：专门用于存储和检索高维向量数据的数据库系统
- **英文定义**：Database systems specifically designed for storing and querying high-dimensional vector data
- **核心功能**：
  - 高效向量存储
  - 快速相似度搜索
  - 大规模数据处理
- **代表产品**：Qdrant、Pinecone、Weaviate、Milvus

### Vector Index / 向量索引
- **中文定义**：用于加速向量搜索的数据结构
- **英文定义**：Data structures designed to accelerate vector search operations
- **常见类型**：
  - HNSW (Hierarchical Navigable Small World)
  - IVF (Inverted File)
  - LSH (Locality Sensitive Hashing)
  - Annoy (Approximate Nearest Neighbors Oh Yeah)

### HNSW (Hierarchical Navigable Small World)
- **中文定义**：分层可导航小世界图索引算法
- **英文定义**：A graph-based indexing algorithm for approximate nearest neighbor search
- **特点**：
  - 高查询性能
  - 良好的召回率
  - 支持动态更新
- **适用场景**：大规模向量检索

### ANN (Approximate Nearest Neighbor)
- **中文定义**：近似最近邻搜索
- **英文定义**：Finding approximate nearest neighbors instead of exact ones
- **权衡**：查询速度 vs 精确度
- **算法类型**：
  - 基于树的方法
  - 基于哈希的方法
  - 基于图的方法
  - 基于量化的方法

## 模型相关术语

### Transformer
- **中文定义**：基于自注意力机制的神经网络架构
- **英文定义**：A neural network architecture based on self-attention mechanisms
- **核心组件**：
  - Multi-Head Attention
  - Position Encoding
  - Feed-Forward Networks
- **代表模型**：BERT、GPT、T5

### BERT (Bidirectional Encoder Representations from Transformers)
- **中文定义**：双向编码器表示的Transformer模型
- **英文定义**：Bidirectional transformer model for language understanding
- **特点**：
  - 双向上下文理解
  - 预训练+微调范式
  - 强大的语义表示能力
- **应用**：文本分类、问答系统、命名实体识别

### Sentence-BERT / SBERT
- **中文定义**：专门用于句子嵌入的BERT变体
- **英文定义**：A modification of BERT for generating sentence embeddings
- **优势**：
  - 高质量句子向量
  - 高效的相似度计算
  - 支持语义搜索
- **训练方式**：Siamese网络结构

### Pre-trained Model / 预训练模型
- **中文定义**：在大规模数据上预先训练好的模型
- **英文定义**：Models that have been trained on large-scale datasets beforehand
- **优势**：
  - 丰富的语言知识
  - 减少训练时间
  - 提高下游任务性能
- **使用方式**：直接使用或微调

## 性能评估术语

### Recall / 召回率
- **中文定义**：检索到的相关结果占所有相关结果的比例
- **英文定义**：The proportion of relevant results that are retrieved
- **计算公式**：Recall = TP / (TP + FN)
- **在向量搜索中**：找到的真正相似向量占所有相似向量的比例

### Precision / 精确率
- **中文定义**：检索结果中相关结果的比例
- **英文定义**：The proportion of retrieved results that are relevant
- **计算公式**：Precision = TP / (TP + FP)
- **权衡关系**：通常与召回率存在权衡

### Throughput / 吞吐量
- **中文定义**：单位时间内处理的向量数量
- **英文定义**：The number of vectors processed per unit time
- **测量单位**：vectors/second, queries/second
- **影响因素**：硬件性能、算法效率、并行度

### Latency / 延迟
- **中文定义**：单次查询的响应时间
- **英文定义**：The time taken to respond to a single query
- **测量方式**：从查询发起到结果返回的时间
- **优化目标**：在保证精度的前提下最小化延迟

## 存储与检索术语

### Collection / 集合
- **中文定义**：向量数据库中的数据容器
- **英文定义**：A container for vector data in a vector database
- **类比**：类似于关系数据库中的表
- **包含内容**：向量数据、元数据、索引配置

### Payload / 载荷
- **中文定义**：与向量关联的元数据信息
- **英文定义**：Metadata associated with vectors
- **内容示例**：
  - 原始文本
  - 文档ID
  - 时间戳
  - 分类标签
- **作用**：提供向量的上下文信息

### Sharding / 分片
- **中文定义**：将大型数据集分割成较小片段的技术
- **英文定义**：The technique of splitting large datasets into smaller fragments
- **目的**：
  - 提高查询性能
  - 支持水平扩展
  - 负载均衡
- **实现方式**：按向量ID、特征值或随机分片

### Replication / 复制
- **中文定义**：创建数据副本以提高可用性和性能
- **英文定义**：Creating copies of data to improve availability and performance
- **类型**：
  - 主从复制
  - 多主复制
  - 异步复制
  - 同步复制

## 优化技术术语

### Quantization / 量化
- **中文定义**：将高精度数值转换为低精度表示的技术
- **英文定义**：The technique of converting high-precision numbers to lower precision
- **类型**：
  - 标量量化
  - 向量量化
  - 产品量化 (Product Quantization)
- **权衡**：存储空间 vs 精度损失

### Product Quantization (PQ)
- **中文定义**：将向量分解为子向量并分别量化的技术
- **英文定义**：A technique that decomposes vectors into subvectors and quantizes them separately
- **优势**：
  - 大幅减少存储空间
  - 加速距离计算
  - 保持相对精度
- **应用**：大规模向量检索系统

### Dimensionality Reduction / 降维
- **中文定义**：减少向量维度的技术
- **英文定义**：Techniques for reducing the number of dimensions in vectors
- **方法**：
  - PCA (主成分分析)
  - t-SNE
  - UMAP
  - 随机投影
- **目的**：减少计算复杂度，可视化高维数据

### Batch Processing / 批处理
- **中文定义**：将多个操作组合在一起执行的处理方式
- **英文定义**：Processing multiple operations together as a group
- **优势**：
  - 提高吞吐量
  - 更好的资源利用
  - 减少系统开销
- **应用场景**：大规模向量生成、批量插入

## 配置参数术语

### 核心参数

#### vector_size / 向量维度
- **参数类型**：整数
- **作用**：定义向量的维度大小
- **常见值**：384, 768, 1024, 1536
- **选择依据**：模型输出维度

#### distance_metric / 距离度量
- **参数类型**：枚举
- **可选值**：Cosine, Euclidean, Dot Product
- **默认值**：通常为Cosine
- **影响**：相似度计算方法

#### ef_construction
- **参数类型**：整数
- **默认值**：200
- **作用**：HNSW索引构建时的搜索范围
- **权衡**：构建时间 vs 索引质量

#### m (HNSW参数)
- **参数类型**：整数
- **默认值**：16
- **作用**：每个节点的最大连接数
- **影响**：索引大小和查询性能

### 查询参数

#### top_k
- **参数类型**：整数
- **默认值**：10
- **作用**：返回最相似的k个结果
- **考虑因素**：业务需求、性能要求

#### ef (查询参数)
- **参数类型**：整数
- **作用**：查询时的搜索范围
- **权衡**：查询速度 vs 召回率
- **建议值**：通常设为top_k的2-10倍

#### score_threshold / 分数阈值
- **参数类型**：浮点数
- **取值范围**：0.0-1.0
- **作用**：过滤低相似度结果
- **设置原则**：根据业务需求调整

## 术语索引

### 按字母排序
- ANN (近似最近邻搜索)
- Batch Processing (批处理)
- BERT (双向编码器表示)
- Collection (集合)
- Contextual Embedding (上下文嵌入)
- Cosine Similarity (余弦相似度)
- Dense Vector (密集向量)
- Dimensionality (维度)
- Dimensionality Reduction (降维)
- Dot Product (点积)
- Embedding (嵌入)
- Euclidean Distance (欧几里得距离)
- HNSW (分层可导航小世界)
- Latency (延迟)
- Manhattan Distance (曼哈顿距离)
- Payload (载荷)
- Precision (精确率)
- Pre-trained Model (预训练模型)
- Product Quantization (产品量化)
- Quantization (量化)
- Recall (召回率)
- Replication (复制)
- Sentence Embedding (句子嵌入)
- Sentence-BERT (句子BERT)
- Sharding (分片)
- Sparse Vector (稀疏向量)
- Text Embedding (文本嵌入)
- Throughput (吞吐量)
- Transformer (变换器)
- Vector Database (向量数据库)
- Vector Index (向量索引)
- Vector Space (向量空间)

### 按主题分类

#### 基础概念
- Embedding, Text Embedding, Sentence Embedding, Contextual Embedding
- Vector Space, Dimensionality, Dense Vector, Sparse Vector

#### 相似度计算
- Cosine Similarity, Euclidean Distance, Dot Product, Manhattan Distance

#### 模型技术
- Transformer, BERT, Sentence-BERT, Pre-trained Model

#### 数据库技术
- Vector Database, Vector Index, Collection, Payload, Sharding, Replication

#### 性能优化
- ANN, HNSW, Quantization, Product Quantization, Dimensionality Reduction, Batch Processing

#### 评估指标
- Recall, Precision, Throughput, Latency

## 实际应用示例

### 术语在代码中的体现
```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Pre-trained Model 加载
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text Embedding 生成
text = "这是一个示例文本"
embedding = model.encode(text)  # Dense Vector

# Vector Database 初始化
client = QdrantClient(":memory:")

# Collection 创建，指定 vector_size 和 distance_metric
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=384,  # Dimensionality
        distance=Distance.COSINE  # Distance Metric
    )
)

# ANN 搜索
search_results = client.search(
    collection_name="documents",
    query_vector=embedding.tolist(),
    limit=5,  # top_k
    score_threshold=0.7  # Score Threshold
)
```

### 性能优化术语应用
```python
# Batch Processing 示例
def batch_encode_texts(texts, batch_size=32):
    """批处理文本编码，提高 Throughput"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

# Quantization 示例
def quantize_vectors(vectors, bits=8):
    """向量量化，减少存储空间"""
    # 简化的量化实现
    min_val, max_val = vectors.min(), vectors.max()
    scale = (max_val - min_val) / (2**bits - 1)
    quantized = ((vectors - min_val) / scale).astype(np.uint8)
    return quantized, scale, min_val
```

## 学习建议

### 重点掌握术语
1. **基础概念**：Embedding, Vector Space, Dimensionality
2. **相似度计算**：Cosine Similarity, Dot Product
3. **数据库技术**：Vector Database, Vector Index, Collection
4. **性能指标**：Recall, Precision, Throughput, Latency

### 实践应用
1. 在技术文档中准确使用专业术语
2. 在代码注释中标注相关概念
3. 在性能调优时运用相关术语进行分析
4. 在团队沟通中使用标准术语提高效率

### 深入学习方向
1. **数学基础**：线性代数、概率论
2. **算法原理**：Transformer、注意力机制
3. **系统设计**：分布式系统、索引算法
4. **性能优化**：并行计算、内存管理