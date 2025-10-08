# RAG 系统实验 2 - 详细操作手册

## 📋 实验概述

本实验在实验 1 的基础上，新增并集成“混合检索与重排序”能力：
- 关键词检索（TF‑IDF，字符 n‑gram 兼容中文）
- 检索融合（RRF 与线性加权两种策略）
- 元数据过滤 DSL（eq/in/range/exists）
- 基于 CrossEncoder 的语义重排（Top‑M 候选）

你将学会如何启动环境、配置应用、上传与向量化文档、并通过新端点 `POST /api/v1/retrieval/search` 使用混合检索与重排功能。

## 🎯 学习目标

- 理解 TF‑IDF 与字符 n‑gram 在中文检索中的作用
- 掌握 RRF 与线性加权两种融合策略的差异与应用
- 使用过滤 DSL 进行结果的二次筛选
- 使用 CrossEncoder 对候选结果进行语义重排并权衡时延

## 🛠️ 实验准备

### 1. 环境要求

必需软件：
- Python 3.12+
- Docker 与 Docker Compose
- uv（Python 包管理器）
- Git

系统要求：
- 内存：至少 4GB RAM
- 存储：至少 2GB 可用空间
- 网络：首次运行需下载模型（约 500MB）

### 2. 项目获取与结构

```bash
# 进入实验目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab02

# 查看项目结构
ls -la
```

关键目录与文件：
- `src/main.py`：应用入口，挂载所有路由（含新检索路由）
- `src/api/routes/retrieval.py`：混合检索路由与编排逻辑
- `src/api/schemas_hybrid.py`：混合检索请求/过滤/融合参数模型
- `src/services/keyword_service.py`：TF‑IDF 关键词检索
- `src/services/fusion_service.py`：RRF/线性加权融合
- `src/services/filter_dsl.py`：过滤 DSL
- `src/services/rerank_service.py`：CrossEncoder 重排
- `src/api/dependencies.py`：依赖注入（关键词与重排服务）

### 3. 依赖安装

#### 方法 1：自动安装（推荐）

```bash
# 在项目根目录运行一次自动脚本（与实验1一致）
python scripts/setup_dev_uv.py
```

脚本会自动完成：
- 检查 Python 版本
- 安装 uv
- 同步项目依赖
- 生成环境配置文件
- 验证安装结果

#### 方法 2：手动安装

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync --extra dev

# 安装嵌入与重排模型依赖
uv add sentence-transformers torch
```

### 4. 环境配置

在项目根目录或 `labs/full/lab02` 目录准备 `.env`，可复用实验 1 的配置并补充以下建议项（如果项目未读取这些变量，可作为文档指导项，或在代码中落地后使用）：

```env
# ===== 检索融合与重排（建议项） =====
HYBRID_SEARCH_ENABLED=true
FUSION_STRATEGY=rrf         # rrf | linear
FUSION_K=60                 # RRF 常数 k
FUSION_W_KEYWORD=0.5        # 线性加权关键词权重
FUSION_W_VECTOR=0.5         # 线性加权向量权重
RERANK_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_M=10

# ===== 数据库与服务（与实验1一致） =====
DATABASE_URL=postgresql://rag_user:rag_password@localhost:15432/rag_db
REDIS_URL=redis://localhost:16379/0
QDRANT_URL=http://localhost:6333
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
```

### 5. 模型准备（可选预下载）

```bash
# 预下载嵌入模型（与实验1相同）
python - << 'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Embedding 模型下载完成！')
PY

# 预下载 CrossEncoder 重排模型
python - << 'PY'
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('CrossEncoder 模型下载完成！')
PY
```

## 🚀 系统启动

### 1. 启动基础服务

```bash
# 进入 labs 根目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs

# 启动 PostgreSQL、Redis、Qdrant
docker-compose up -d postgres redis qdrant

# 验证服务状态
docker-compose ps
```

### 2. 数据库初始化（如已在实验1执行，可跳过）

```bash
# 进入实验1目录，运行迁移（仅首次）
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab01
uv run alembic upgrade head
```

### 3. 启动 RAG 应用（实验 2）

```bash
# 进入实验2目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab02

# 开发模式（热重载）
uv run python src/main.py

# 或使用自定义脚本（若存在）：
# uv run rag-server
```

### 4. 验证启动状态

- 健康检查: `http://localhost:8000/api/v1/system/health`
- API 文档: `http://localhost:8000/docs`
- ReDoc 文档: `http://localhost:8000/redoc`

## 📚 功能操作指南（实验 2 新增能力）

> 以下功能基于实验 1 的文档上传与向量化能力，请先完成基本文档入库与向量化。

### 1. 上传文档（与实验1一致）

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=测试文档" \
  -F "description=这是一个测试文档"
```

### 2. 向量化文档（与实验1一致）

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/vectorize" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc_123456"], "batch_size": 16}'
```

### 3. 混合检索（关键词 + 向量 + 融合 + 过滤 + 重排）

接口：`POST /api/v1/retrieval/search`

请求参数模型：`HybridSearchRequest`
- `query: str`（必填）
- `top_k: int = 10`（返回数量）
- `score_threshold: float = 0.2`（最低分阈值）
- `document_ids: List[int] | None`（按文档过滤）
- `fusion: { strategy: "rrf" | "linear", k?: int, w_keyword?: float, w_vector?: float }`
- `filters: [{ op, field, value|min|max }]`（DSL）
- `rerank_top_m: int = 10`（重排候选量）

#### 3.1 使用 RRF 融合

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "向量数据库原理",
    "top_k": 8,
    "score_threshold": 0.2,
    "fusion": {"strategy": "rrf", "k": 60},
    "filters": [{"op": "exists", "field": "metadata.page"}],
    "rerank_top_m": 8
  }'
```

#### 3.2 使用线性加权融合

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "中文检索效果评估",
    "top_k": 10,
    "fusion": {"strategy": "linear", "w_keyword": 0.4, "w_vector": 0.6},
    "filters": [
      {"op": "range", "field": "score", "min": 0.3},
      {"op": "in", "field": "metadata.section", "value": ["intro", "method"]}
    ],
    "rerank_top_m": 6
  }'
```

#### 3.3 关键词索引与性能提示

- 首次调用将自动构建 TF‑IDF 索引（字符 n‑gram = 2..4），耗时取决于分块数量
- 新增文档后索引会在下一次搜索前进行刷新或重建（具体实现见 `KeywordService.refresh()`）
- 若重排导致时延较高，可将 `rerank_top_m` 降低，或设为 `0` 关闭重排

### 4. 过滤 DSL 语法示例

```json
[
  { "op": "eq", "field": "document_id", "value": 123 },
  { "op": "in", "field": "metadata.page", "value": [1, 2, 3] },
  { "op": "range", "field": "score", "min": 0.3, "max": 0.9 },
  { "op": "exists", "field": "metadata.author" }
]
```

字段寻址支持 `metadata.foo.bar` 深层路径。不合法字段或未知操作符将被拒绝。

## 🧾 接口操作与 I/O 说明

以下为实验 2 中各核心端点的“操作步骤 + 输入/输出”说明，均给出请求示例与典型响应结构，便于快速对照与复测。

### 1) 系统健康检查
- 路径：`GET /api/v1/system/health`
- 输入：无（HTTP GET）
- 输出（示例）：
```json
{
  "success": true,
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "vector_store": "healthy",
    "embedding_service": "healthy",
    "volcengine_api": "healthy"
  }
}
```

### 2) 文档列表
- 路径：`GET /api/v1/documents/`
- 输入：无（HTTP GET）
- 输出（示例）：
```json
[
  {
    "id": "uuid",
    "filename": "test_document.txt",
    "title": null,
    "description": null,
    "file_size": 12345,
    "file_type": "text/plain",
    "is_processed": true,
    "is_vectorized": true
  }
]
```

### 3) 文档上传（与实验 1 一致）
- 路径：`POST /api/v1/documents/upload`
- 输入（multipart/form-data）：
  - `file`: 文档文件
  - `title`: 文档标题（可选）
  - `description`: 文档描述（可选）
- 输出（示例）：
```json
{
  "success": true,
  "id": "uuid",
  "filename": "your_document.pdf",
  "message": "上传成功"
}
```

### 4) 文档向量化（与实验 1 一致）
- 路径：`POST /api/v1/vectors/vectorize`
- 输入（JSON）：
```json
{
  "document_ids": ["uuid1", "uuid2"],
  "batch_size": 16
}
```
- 输出（示例）：
```json
{
  "success": true,
  "processed_count": 2,
  "failed_ids": []
}
```

### 5) 向量检索（对照组）
- 路径：`POST /api/v1/vectors/search`
- 输入（JSON）：
```json
{
  "query": "RAG系统",
  "limit": 6,
  "score_threshold": 0.2,
  "document_ids": null
}
```
- 输出（示例，关键字段）：
```json
{
  "success": true,
  "total_found": 5,
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "document_filename": "test_document.txt",
      "chunk_index": 0,
      "content": "...",
      "score": 0.78,
      "start_position": 0,
      "end_position": 488,
      "metadata": {}
    }
  ]
}
```

### 6) 混合检索（RRF 融合）
- 路径：`POST /api/v1/retrieval/search`
- 输入（JSON）：
```json
{
  "query": "RAG系统",
  "top_k": 6,
  "score_threshold": 0.2,
  "fusion": {"strategy": "rrf", "k": 60},
  "filters": [],
  "rerank_top_m": 6
}
```
- 输出（示例，关键字段）：
```json
{
  "success": true,
  "message": "检索成功",
  "query": "RAG系统",
  "results": [ { "chunk_id": "uuid", "document_filename": "...", "score": 0.02 } ],
  "total_found": 5,
  "processing_time": 0.35
}
```
- 说明：在无过滤、阈值较高时可能出现空结果；可降低 `score_threshold` 或结合 `filters` 使用（见下文）。

### 7) 混合检索（线性加权融合）
- 路径：`POST /api/v1/retrieval/search`
- 输入（JSON）：
```json
{
  "query": "RAG系统",
  "top_k": 6,
  "score_threshold": 0.0,
  "fusion": {"strategy": "linear", "w_keyword": 0.4, "w_vector": 0.6},
  "filters": [],
  "rerank_top_m": 6
}
```
- 输出（示例，关键字段）：
```json
{
  "success": true,
  "total_found": 5,
  "results": [
    {"document_filename": "test_document.txt", "score": 0.016},
    {"document_filename": "tmptpi5hpw0.pdf", "score": 0.015}
  ]
}
```

### 8) 过滤 DSL（eq / in / range / exists）
- 路径：`POST /api/v1/retrieval/search`
- 输入（JSON）：在 `filters` 中添加条件，字段支持顶层与 `metadata.*` 深层路径；例：
```json
{
  "query": "RAG系统",
  "top_k": 6,
  "fusion": {"strategy": "rrf", "k": 60},
  "filters": [
    {"op": "eq", "field": "document_filename", "value": "test_document.txt"},
    {"op": "in", "field": "document_filename", "value": ["test_document.txt", "tmptpi5hpw0.pdf"]},
    {"op": "range", "field": "score", "min": 0.01}
    // {"op": "exists", "field": "metadata.page"}
  ],
  "rerank_top_m": 6
}
```
- 输出：与检索输出一致（`results` 会被过滤后再融合/重排）；非法字段或操作符将被拒绝。

### 9) 重排服务（CrossEncoder）
- 路径：`POST /api/v1/retrieval/search`
- 输入关键参数：`rerank_top_m`
  - `> 0`：开启重排，值为参与重排的候选数（如 `6`）
  - `= 0`：关闭重排，仅按融合得分返回
- 典型效果：开启重排时候选排序变化、`processing_time` 略增（如 0.37s → 0.40s）。

### 10) 问答接口（RAG）
- 路径：`POST /api/v1/qa/ask`
- 输入（JSON）：
```json
{
  "question": "什么是RAG系统？",
  "top_k": 5,
  "score_threshold": 0.2,
  "context_size": 2
}
```
- 输出（示例，关键字段）：
```json
{
  "success": true,
  "message": "问答完成",
  "question": "什么是RAG系统？",
  "answer": "检索增强生成（RAG）是一种结合检索与生成的技术...",
  "sources": [
    {
      "document_id": "uuid",
      "document_filename": "test_document.txt",
      "chunk_id": "uuid",
      "chunk_index": 1,
      "score": 0.78,
      "content_preview": "...RAG系统..."
    }
  ],
  "context_used": 1,
  "processing_time": 7.59,
  "model_used": "doubao-seed-1-6-250615"
}
```

> 参数建议：当候选偏少或中文检索召回不足时，可适当降低 `score_threshold`，或先使用线性融合获取更稳定候选，再配合 `filters` 与重排提升最终排序质量。

## 🧪 实验验证

### 1. 功能验证清单

- [ ] 系统成功启动，健康检查返回正常
- [ ] 文档上传与向量化完成
- [ ] 关键词检索自动索引并返回结果
- [ ] RRF 与线性加权融合结果可用
- [ ] 过滤 DSL 生效并筛选候选
- [ ] CrossEncoder 重排生效，Top‑K 排序稳定

### 2. 对比与评估建议

```bash
# 仅向量检索（对照组）
curl -X GET "http://localhost:8000/api/v1/vectors/search?query=向量数据库原理&top_k=8"

# 混合检索（实验组，RRF）
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "向量数据库原理", "top_k": 8, "fusion": {"strategy": "rrf"}}'
```

评估指标：`Recall@K`、`MRR`、`nDCG@K`、`Latency`。

## 🐛 故障排除

### 1. 模型下载或网络问题

```bash
# 使用镜像源提高下载成功率
python - << 'PY'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('模型镜像下载完成！')
PY
```

### 2. Qdrant/数据库连接失败

```bash
docker-compose ps qdrant
docker-compose restart qdrant
curl http://localhost:6333/health

docker-compose ps postgres
docker-compose restart postgres
lsof -i :15432
```

### 3. 重排导致延迟过高

- 降低 `rerank_top_m`
- 关闭重排：将 `rerank_top_m=0`
- 预热模型：启动后先调用一次检索以加载 CrossEncoder

## 📊 性能基准（建议）

```bash
# 查询响应时间
time curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "测试问题", "top_k": 10, "rerank_top_m": 5}'
```

## 🎓 实验总结

完成本实验后，你将能够：
- 将关键词与向量检索进行有效融合，提高召回与精度
- 使用 DSL 精准筛选候选，提升业务可控性
- 通过 CrossEncoder 重排进一步提升最终排序质量

祝你实验顺利！🚀