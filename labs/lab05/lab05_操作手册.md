# RAG 系统实验 5 - 详细操作手册

## 📋 实验概述

本实验在前序实验基础上，强化“端到端数据流程与系统运维能力”，聚焦以下能力与验证项：
- 文档上传 → 分块处理 → 向量化 → 检索的完整链路
- 向量检索过滤 DSL（`eq`/`in`/`range`/`exists`）
- 全量重建与增量向量化（`/vectors/reindex` 与 `/vectors/vectorize`）
- 级联删除（删除文档后自动清理对应向量）
- 系统健康检查与统计接口（数据库、向量库、嵌入服务等）
- 列表接口的重定向规范（尾斜杠与 307 行为）

你将学会如何启动环境、配置应用、上传/向量化文档，并通过新端点验证过滤检索与删除级联等行为。

## 🎯 学习目标

- 掌握文档入库、分块与向量化的标准流程
- 使用过滤 DSL 对检索结果做精细筛选
- 理解并执行全量重建与增量向量化
- 验证删除文档的级联清理（向量自动移除）
- 熟悉健康检查、系统统计与列表接口规范

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
- 网络：首次运行可能需要下载嵌入模型（约 500MB）

### 2. 项目获取与结构

```bash
# 进入实验根目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab05

# 查看项目结构
ls -la
```

关键目录与文件（以参考为准，实际以代码为主）：
- `src/main.py`：应用入口，挂载所有路由
- `src/api/routes/documents.py`：文档上传/处理/删除
- `src/api/routes/vectors.py`：文档向量化/检索/重建
- `src/api/routes/ingestion.py`：外部数据源摄取（CSV/SQL/API）
- `src/api/routes/system.py`：健康检查与统计
- `scripts/setup_dev_uv.py` / `scripts/setup_dev.py`：开发环境与基础服务启动脚本
- `scripts/test_system.py`：系统端到端验证脚本

### 3. 依赖安装

#### 方法 1：自动安装（推荐）

```bash
# 在 Lab05 项目目录运行一次自动脚本
python scripts/setup_dev_uv.py
```

脚本会自动完成：
- 检查 Python 版本与 uv 安装
- 同步项目依赖
- 生成环境配置文件（`.env`）
- 验证安装结果

#### 方法 2：手动安装

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（开发环境）
uv sync --extra dev

# 安装嵌入模型依赖（如需）
uv add sentence-transformers torch
```

### 4. 环境配置

在 `labs/full/lab05` 目录准备 `.env`，可复用实验 1/2 的配置并补充以下建议项（若项目未读取某些变量，则作为文档指导项）：

```env
# ===== 应用与服务 =====
DATABASE_URL=postgresql://rag_user:rag_password@localhost:15432/rag_db
REDIS_URL=redis://localhost:16379/0
QDRANT_URL=http://localhost:6333
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true

# ===== 向量化（建议项） =====
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTORIZE_BATCH_SIZE=16

# ===== 检索过滤（建议项） =====
FILTER_DSL_ENABLED=true
```

### 5. 模型准备（可选预下载）

```bash
# 预下载嵌入模型
python - << 'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Embedding 模型下载完成！')
PY
```

## 🚀 系统启动

### 1. 启动基础服务

```bash
# 进入 labs 根目录（包含 docker-compose.yml）
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs

# 启动 PostgreSQL、Redis、Qdrant
docker compose up -d postgres redis qdrant

# 验证服务状态
docker compose ps
```

### 2. 数据库初始化（首次或迁移更新）

```bash
# 在 Lab05 项目目录运行 Alembic 迁移
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab05
uv run alembic upgrade head
```

### 3. 启动 RAG 应用（实验 5）

```bash
# 开发模式（热重载）
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 验证启动状态

- 健康检查: `http://localhost:8000/api/v1/system/health`
- API 文档: `http://localhost:8000/docs`
- ReDoc 文档: `http://localhost:8000/redoc`

## 📚 功能操作指南（实验 5 能力）

> 以下功能基于文档上传与向量化能力，请先完成入库与向量化。

### 1. 上传文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=测试文档" \
  -F "description=这是一个测试文档"
```

典型输出：

```json
{
  "success": true,
  "id": "uuid",
  "filename": "your_document.pdf",
  "message": "上传成功"
}
```

### 2. 查看分块与向量化状态

```bash
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/chunks"
```

典型字段：`chunk_id`、`chunk_index`、`is_vectorized`、`vector_id`、`content_preview` 等。

### 3. 向量化文档（增量）

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/vectorize" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["uuid1", "uuid2"], "batch_size": 16}'
```

典型输出：

```json
{
  "success": true,
  "processed_count": 2,
  "failed_ids": []
}
```

### 4. 全量重建（向量重索引）

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/reindex" \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

用于重建所有分块向量索引（当模型或索引策略变更时）。

### 5. 向量检索（支持过滤 DSL）

接口：`POST /api/v1/vectors/search`

请求示例：

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "RAG系统",
    "limit": 8,
    "score_threshold": 0.2,
    "filters": [
      {"op": "eq", "field": "document_filename", "value": "your_document.pdf"},
      {"op": "exists", "field": "metadata.page"}
    ]
  }'
```

过滤 DSL 语法示例：

```json
[
  { "op": "eq", "field": "document_id", "value": 123 },
  { "op": "in", "field": "metadata.page", "value": [1, 2, 3] },
  { "op": "range", "field": "score", "min": 0.3, "max": 0.9 },
  { "op": "exists", "field": "metadata.author" }
]
```

说明：字段寻址支持 `metadata.foo.bar` 深层路径；非法字段或未知操作符将被拒绝。

### 6. 删除文档（级联删除向量）

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/{document_id}"
```

预期：文档删除后，其所有分块与向量在向量库中被级联清理；可通过过滤检索或分块查询验证结果为 0。

### 7. 系统统计与列表接口

- 系统统计：

```bash
curl -X GET "http://localhost:8000/api/v1/system/stats"
```

- 文档列表（注意尾斜杠规范）：

```bash
# 无尾斜杠可能返回 307 重定向
curl -i -X GET "http://localhost:8000/api/v1/documents"

# 带尾斜杠返回 200
curl -i -X GET "http://localhost:8000/api/v1/documents/"
```

## 🧾 接口 I/O 速览

以下为实验 5 中各核心端点的“操作步骤 + 输入/输出”说明，便于快速对照与复测。

1) 系统健康检查（`GET /api/v1/system/health`）
```json
{
  "success": true,
  "status": "healthy",
  "components": {
    "database": "healthy",
    "vector_store": "healthy",
    "embedding_service": "healthy",
    "volcengine_api": "healthy"
  }
}
```

2) 文档列表（`GET /api/v1/documents/`）
```json
[
  {
    "id": "uuid",
    "filename": "test_document.txt",
    "is_processed": true,
    "is_vectorized": true
  }
]
```

3) 文档上传（`POST /api/v1/documents/upload`）
```json
{
  "success": true,
  "id": "uuid",
  "filename": "your_document.pdf"
}
```

4) 文档向量化（`POST /api/v1/vectors/vectorize`）
```json
{
  "success": true,
  "processed_count": 2,
  "failed_ids": []
}
```

5) 向量检索（`POST /api/v1/vectors/search`）
```json
{
  "success": true,
  "total_found": 5,
  "results": [
    {
      "chunk_id": "uuid",
      "document_filename": "test_document.txt",
      "score": 0.78
    }
  ]
}
```

6) 全量重建（`POST /api/v1/vectors/reindex`）
```json
{
  "success": true,
  "message": "重建完成",
  "processed": 123
}
```

7) 删除文档（`DELETE /api/v1/documents/{document_id}`）
```json
{
  "success": true,
  "message": "删除成功"
}
```

## 🧪 实验验证

### 1. 功能验证清单

- [ ] 系统成功启动，健康检查返回正常
- [ ] 文档上传与分块完成
- [ ] 指定文档向量化成功，分块 `is_vectorized=true`
- [ ] 向量检索返回结果，过滤 DSL 生效
- [ ] 全量重建完成，索引可用
- [ ] 删除文档后向量级联清理，检索返回 0
- [ ] 系统统计与文档列表返回结构正确（尾斜杠行为一致）

### 2. 对比与评估建议

```bash
# 仅向量检索（对照组）
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "向量数据库原理", "limit": 8}'

# 过滤 DSL（实验组）
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "向量数据库原理",
    "limit": 8,
    "filters": [{"op": "exists", "field": "metadata.page"}]
  }'
```

评估指标：`Recall@K`、`MRR`、`nDCG@K`、`Latency`。

## 🐛 故障排除

### 1. 模型下载或网络问题

```bash
# 使用镜像源提高下载成功率
python - << 'PY'
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('模型镜像下载完成！')
PY
```

### 2. Qdrant/数据库连接失败

```bash
docker compose ps qdrant
docker compose restart qdrant
curl http://localhost:6333/health

docker compose ps postgres
docker compose restart postgres
lsof -i :15432
```

### 3. 向量化 500 错误或分块未向量化

- 检查 `.env` 中的 `EMBEDDING_MODEL` 与网络下载是否正常
- 查看应用日志，定位具体异常与堆栈
- 先对指定文档执行增量向量化；若仍失败，尝试 `reindex`
- 通过 `GET /api/v1/documents/{id}/chunks` 检查 `is_vectorized` 与 `vector_id`

### 4. 列表接口 307 重定向

- 无尾斜杠时可能返回 307；建议使用带尾斜杠的资源路径（如 `.../documents/`）

## 📊 性能基准（建议）

```bash
# 查询响应时间
time curl -X POST "http://localhost:8000/api/v1/vectors/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "测试问题", "limit": 10}'
```

## 🎓 实验总结

完成本实验后，你将能够：
- 熟练完成文档上传、分块、向量化与检索的端到端流程
- 使用过滤 DSL 提升检索的可控性与精度
- 通过增量向量化与全量重建维护索引一致性
- 验证并依赖级联删除机制保持数据与向量库的整洁

祝你实验顺利！🚀