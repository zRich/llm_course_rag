# RAG 系统实验 4 - 详细操作手册（批量接入与增量更新）

## 📋 实验概述

本实验在实验 1/2 的基础上，新增并集成“批量数据接入 + 断点续传 + 幂等 + 增量更新与失效重建”能力：
- 结构化数据接入三类连接器：CSV / SQL / HTTP API
- 统一字段映射与元数据规范化（title/content/metadata）
- 清洗流水线（统一换行与空格、去噪、去空行）提高分块质量
- 批量接入支持断点续传与幂等键，保障重复调用安全
- 增量更新服务：内容哈希检测变化，自动失效重建分块与统计信息

你将学会如何启动环境、配置应用、批量接入结构化数据、以及对已有文档进行增量更新与重建。

## 🎯 学习目标

- 掌握 CSV/SQL/API 三类数据源的标准化接入
- 理解幂等键与检查点在批处理中的作用与使用方式
- 通过增量更新服务自动检测内容变化并重建分块
- 结合文本清洗流水线，提高分块与向量化质量

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
# 进入实验 4 目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab04

# 查看项目结构
ls -la
```

关键目录与文件：
- `src/main.py`：应用入口，挂载所有路由（含新增接入与增量路由）
- `src/api/routes/ingestion.py`：批量接入与增量更新 API 路由
- `src/services/ingestion.py`：检查点、幂等键、批量加载器与 DocumentSink
- `src/services/incremental.py`：增量更新与失效重建服务
- `src/services/cleaning.py`：文本清洗去噪服务（写库前统一规范化）
- `src/connectors/csv_connector.py`：CSV 连接器
- `src/connectors/sql_connector.py`：SQL 连接器（SQLAlchemy）
- `src/connectors/api_connector.py`：HTTP API 连接器（支持嵌套路径）

> 注：本实验延续实验 1/2 的文档上传、向量化与检索能力。若首次运行，请先完成基础环境与迁移。

### 3. 依赖安装

#### 方法 1：自动安装（推荐）

```bash
# 在实验 4 目录运行一次自动脚本（与实验1一致）
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

# 若需嵌入或重排模型（延续实验 1/2）
uv add sentence-transformers torch
```

### 4. 环境配置

在项目根或实验 4 目录准备 `.env`，可复用实验 1 的配置。建议项：

```env
# ===== 批量接入与增量（建议项） =====
INGESTION_BATCH_SIZE=64
INGESTION_MAX_RETRIES=3
IDEMPOTENCY_ENABLED=true

# ===== 数据库与服务（与实验1一致） =====
DATABASE_URL=postgresql://rag_user:rag_password@localhost:15432/rag_db
REDIS_URL=redis://localhost:16379/0
QDRANT_URL=http://localhost:6333
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
```

> 若项目暂未读取上述变量，可作为文档指导项；或在实际使用中按路由参数进行覆盖。

### 5. 模型准备（可选预下载）

延续实验 1/2 的嵌入与重排模型（可选）：

```bash
# 预下载嵌入模型（与实验1相同）
python - << 'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Embedding 模型下载完成！')
PY

# 预下载 CrossEncoder 重排模型（若与实验2联动）
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

# 启动 PostgreSQL、Redis、Qdrant（如实验 1 已启动可跳过）
docker-compose up -d postgres redis qdrant

docker-compose ps
```

### 2. 数据库初始化（如已在实验1执行，可跳过）

```bash
# 进入实验1目录，运行迁移（仅首次）
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab01
uv run alembic upgrade head
```

### 3. 启动 RAG 应用（实验 4）

```bash
# 进入实验4目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab04

# 开发模式（热重载）
uv run python src/main.py

# 或使用自定义脚本（若存在）：
# uv run rag-server
```

### 4. 验证启动状态

- 健康检查: `http://localhost:8000/api/v1/system/health`
- API 文档: `http://localhost:8000/docs`
- ReDoc 文档: `http://localhost:8000/redoc`

## 📚 功能操作指南（实验 4 新增能力）

> 以下功能在实验 1 的文档与向量能力基础上提供。建议先完成基本文档入库与向量化，以便后续增量与重建验证。

### A. 标准化字段说明

所有结构化数据在接入时将被映射为以下统一字段：
- `title: str | None` 文档标题（可选）
- `content: str` 文本内容（必填，用于分块与向量化）
- `metadata: dict` 任意元数据（如 `source`, `author`, `tags`, `page` 等）

清洗流水线会在写库前统一处理文本：
- 统一换行与空格、去除多余空格
- 去除行首尾空白
- 删除空行，提高分块密度与质量

### B. 连接器接入（CSV / SQL / API）

> 统一路由前缀：`/api/v1/ingestion/*`

#### 1) CSV 批量接入
- 路径：`POST /api/v1/ingestion/csv`
- 输入（JSON）：
```json
{
  "file_path": "/absolute/path/to/data.csv",
  "field_mapping": { "title": "title", "content": "body", "metadata": "meta" },
  "default_metadata": { "source": "csv_demo", "collection": "news" },
  "batch_size": 64,
  "checkpoint_id": null,
  "idempotency_key": null
}
```
- 说明：
  - `field_mapping` 将 CSV 列映射到统一字段；缺失字段会使用默认值或空字符串
  - 若提供 `checkpoint_id`，则在失败后可继续从断点恢复
  - 未提供 `idempotency_key` 时，系统可根据输入配置生成幂等键，避免重复入库
- 输出（示例）：
```json
{
  "success": true,
  "processed": 128,
  "failed": 0,
  "checkpoint_id": "ckpt_20241001_123456",
  "idempotency_key": "csv:5f7a..."
}
```

#### 2) SQL 批量接入（SQLAlchemy）
- 路径：`POST /api/v1/ingestion/sql`
- 输入（JSON）：
```json
{
  "connection_url": "postgresql://user:pass@localhost:15432/dbname",
  "query": "SELECT title, body AS content, to_json(meta) AS metadata FROM articles LIMIT 100",
  "field_mapping": { "title": "title", "content": "content", "metadata": "metadata" },
  "default_metadata": { "source": "postgres", "table": "articles" },
  "batch_size": 64,
  "checkpoint_id": null,
  "idempotency_key": null
}
```
- 说明：
  - `query` 返回的列需能映射到统一字段
  - 复杂结构可通过数据库函数（如 `to_json`) 统一为 JSON

#### 3) HTTP API 批量接入
- 路径：`POST /api/v1/ingestion/api`
- 输入（JSON）：
```json
{
  "url": "https://example.com/api/articles",
  "method": "GET",
  "headers": { "Authorization": "Bearer <TOKEN>" },
  "params": { "limit": 100 },
  "json": null,
  "data_path": "data.items",  
  "field_mapping": { "title": "title", "content": "text", "metadata": "_" },
  "default_metadata": { "source": "example_api" },
  "batch_size": 64,
  "checkpoint_id": null,
  "idempotency_key": null
}
```
- 说明：
  - `data_path` 指向响应中的数据数组（支持嵌套路径，如 `data.items`）
  - 当 `metadata` 映射为 `_` 时，表示保留整条原始记录作为元数据

### C. 批处理运行与断点续传

- 路径：`POST /api/v1/ingestion/batch/run`
- 输入（JSON）：
```json
{
  "source_type": "csv",  
  "source_config": { "file_path": "/path/to/data.csv", "field_mapping": {"title": "t", "content": "c"} },
  "default_metadata": { "source": "csv_demo" },
  "batch_size": 64,
  "checkpoint_id": null,
  "idempotency_key": null
}
```
- 说明：
  - `source_type`: `csv` | `sql` | `api`
  - 首次运行返回 `checkpoint_id`，失败重试时将自动续传
  - 幂等键可避免重复写入同一数据集（按配置生成）

### D. 增量更新与失效重建

> 统一路由前缀：`/api/v1/incremental/*`

#### 1) 文档内容更新（Upsert）
- 路径：`POST /api/v1/incremental/upsert`
- 输入（JSON）：
```json
{
  "document_id": "uuid-or-external-id",
  "title": "新标题（可选）",
  "content": "新的文档正文",
  "metadata": { "source": "update_demo" },
  "rechunk": true
}
```
- 行为：
  - 系统计算内容哈希，检测变化；若变化，则删除旧分块、重分块与重建统计
  - 未变化则跳过重建，仅更新元数据或标题（如提供）

#### 2) 文档失效重建（Rebuild）
- 路径：`POST /api/v1/incremental/rebuild`
- 输入（JSON）：
```json
{ "document_id": "uuid-or-external-id" }
```
- 行为：
  - 强制失效并重建文档分块与相关统计（不依赖内容变化）

### E. 与实验 1/2 的能力联动

- 上传文档（实验 1）：`POST /api/v1/documents/upload`
- 向量化（实验 1）：`POST /api/v1/vectors/vectorize`
- 检索（实验 2）：`POST /api/v1/retrieval/search`

> 批量接入产生的新文档将参与后续向量化与检索流程；建议在批量接入后运行一次向量化以便检索验证。

## 🧾 接口操作与 I/O 说明

以下为实验 4 中各核心端点的“操作步骤 + 输入/输出”说明，均给出请求示例与典型响应结构，便于快速对照与复测。

### 1) 系统健康检查
- 路径：`GET /api/v1/system/health`
- 输入：无（HTTP GET）
- 输出（示例）：
```json
{ "success": true, "status": "healthy", "version": "1.0.0" }
```

### 2) CSV 接入
- 路径：`POST /api/v1/ingestion/csv`
- 见上文示例；关键返回字段：`processed`, `failed`, `checkpoint_id`, `idempotency_key`

### 3) SQL 接入
- 路径：`POST /api/v1/ingestion/sql`
- 见上文示例；关键返回字段：`processed`, `failed`, `checkpoint_id`, `idempotency_key`

### 4) API 接入
- 路径：`POST /api/v1/ingestion/api`
- 见上文示例；关键返回字段：`processed`, `failed`, `checkpoint_id`, `idempotency_key`

### 5) 批处理运行（断点续传）
- 路径：`POST /api/v1/ingestion/batch/run`
- 见上文示例；关键返回字段：`processed`, `failed`, `checkpoint_id`, `idempotency_key`

### 6) 增量更新（Upsert）
- 路径：`POST /api/v1/incremental/upsert`
- 输出（示例）：
```json
{ "success": true, "updated": 1, "rechunked": true, "hash_changed": true }
```

### 7) 失效重建（Rebuild）
- 路径：`POST /api/v1/incremental/rebuild`
- 输出（示例）：
```json
{ "success": true, "rebuild": "done", "document_id": "uuid" }
```

## 🧪 实验验证

### 1. 功能验证清单

- [ ] 系统成功启动，健康检查返回正常
- [ ] 成功从 CSV/SQL/API 任一数据源批量接入
- [ ] 断点续传生效（人工制造错误后继续执行）
- [ ] 幂等键生效（重复请求不重复写入）
- [ ] 文本清洗生效（空行与多余空格被清理）
- [ ] 增量更新：内容变化触发重建，未变化仅更新元信息
- [ ] 失效重建：强制重建完成
- [ ] 后续向量化与检索联动正常（与实验 1/2）

### 2. 对比与评估建议

```bash
# 批量接入后进行向量化（示例）
curl -X POST http://localhost:8000/api/v1/vectors/vectorize \
  -H 'Content-Type: application/json' \
  -d '{"document_ids": ["uuid1", "uuid2"], "batch_size": 16}'

# 增量更新后进行检索验证（示例）
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "系统可靠性设计", "top_k": 8}'
```

评估指标：`Processed Count`、`Failed Count`、`Latency`（批次与单条）与检索 `Recall@K`。

## 🐛 故障排除

### 1. 数据源访问失败（路径/网络/权限）
- CSV：确认 `file_path` 为绝对路径且文件存在
- SQL：检查 `connection_url`、数据库联通与权限，尝试 `LIMIT` 验证查询
- API：检查 `url`、鉴权头与响应结构；合理设置 `data_path`

### 2. 断点续传未生效
- 确认返回的 `checkpoint_id` 已在下一次请求中传入
- 检查幂等键：确保相同数据集使用同一个 `idempotency_key`

### 3. 处理性能与超时
- 降低 `batch_size` 或在连接器层分页接入
- 打开 `DEBUG` 观察每批处理耗时；必要时调整分块大小

### 4. 增量更新无效或误重建
- 检查 `content` 是否真实变化；系统基于内容哈希判断
- 若需强制重建，请改用 `POST /api/v1/incremental/rebuild`

## 📊 性能基准（建议）

```bash
# 批量运行时延
TIMEFORMAT=$'批量执行耗时: %3R 秒'
{ time curl -X POST http://localhost:8000/api/v1/ingestion/batch/run \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"csv","source_config":{"file_path":"/path/to/data.csv","field_mapping":{"title":"t","content":"c"}},"batch_size":64}'; } 2>&1
```

## 🎓 实验总结

完成本实验后，你将能够：
- 将 CSV/SQL/API 三类结构化数据稳定接入到 RAG 系统
- 使用幂等键与检查点机制保障批处理安全与可恢复性
- 基于内容哈希进行增量更新与失效重建，维护数据一致性
- 借助清洗流水线提升分块质量，为后续向量化与检索打下基础

祝你实验顺利！🚀