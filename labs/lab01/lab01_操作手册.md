# RAG 系统实验 1 - 详细操作手册

## 📋 实验概述

本实验将带你构建一个完整的RAG（检索增强生成）系统，支持中文文档处理、向量化存储、语义搜索和智能问答。系统采用本地嵌入模型优先的策略，确保在课堂环境中能够稳定运行。

## 🎯 学习目标

- 理解RAG系统的核心架构和工作原理
- 掌握文档向量化和语义搜索技术
- 学会配置和使用本地嵌入模型
- 实践文档管理和智能问答功能
- 了解系统监控和故障排除方法

## 🛠️ 实验准备

### 1. 环境要求

**必需软件**：
- Python 3.12+
- Docker 和 Docker Compose
- uv (Python包管理器)
- Git

**系统要求**：
- 内存：至少 4GB RAM
- 存储：至少 2GB 可用空间
- 网络：首次运行需要下载模型（约500MB）

### 2. 项目获取

```bash
# 进入实验目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab01

# 查看项目结构
ls -la
```

### 3. 依赖安装

#### 方法1: 自动安装（推荐）

```bash
# 运行自动设置脚本
python scripts/setup_dev_uv.py
```

脚本会自动完成：
- 检查Python版本
- 安装uv包管理器
- 同步项目依赖
- 生成环境配置文件
- 验证安装结果

#### 方法2: 手动安装

```bash
# 1. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 同步依赖
uv sync --extra dev

# 3. 安装嵌入模型依赖
uv add sentence-transformers torch
```

### 4. 环境配置

#### 4.1 创建环境配置文件

```bash
# 复制配置模板
cp .env.example .env
```

#### 4.2 编辑配置文件

打开 `.env` 文件，配置以下关键参数：

```env
# ===== 嵌入模型配置 (本地模式，优化中文支持) =====
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LOCAL_EMBEDDING_DIMENSION=384

# ===== 火山引擎配置 (可选) =====
# 如果需要使用API模式，请配置以下参数
# VOLCENGINE_API_KEY=your-api-key-here
# VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
# VOLCENGINE_MODEL=doubao-seed-1-6-250615

# ===== 数据库配置 =====
DATABASE_URL=postgresql://rag_user:rag_password@localhost:15432/rag_db
REDIS_URL=redis://localhost:16379/0

# ===== Qdrant向量数据库配置 =====
QDRANT_URL=http://localhost:6333

# ===== 应用配置 =====
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
```

### 5. 嵌入模型准备

#### 5.1 推荐的中文嵌入模型

| 模型名称 | 维度 | 特点 | 推荐场景 |
|---------|------|------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 多语言支持，中英文效果好 | **默认推荐** |
| `shibing624/text2vec-base-chinese` | 768 | 专门的中文模型 | 纯中文文档 |
| `BAAI/bge-small-zh-v1.5` | 512 | BGE中文小模型 | 高质量需求 |
| `distiluse-base-multilingual-cased` | 512 | 多语言通用 | 平衡性能 |

#### 5.2 模型下载和缓存

首次运行时，系统会自动下载选定的嵌入模型：

```bash
# 预下载模型（可选）
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('模型下载完成！')
"
```

**注意事项**：
- 模型会缓存在 `~/.cache/huggingface/` 目录
- 首次下载约需要3-5分钟
- 下载完成后可离线使用

## 🚀 系统启动

### 1. 启动基础服务

```bash
# 进入实验根目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs

# 启动PostgreSQL、Redis、Qdrant服务
docker-compose up -d postgres redis qdrant

# 验证服务状态
docker-compose ps
```

预期输出：
```
NAME                COMMAND                  SERVICE             STATUS              PORTS
labs-postgres-1     "docker-entrypoint.s…"   postgres            running             0.0.0.0:15432->5432/tcp
labs-qdrant-1       "./qdrant"               qdrant              running             0.0.0.0:6333->6333/tcp, 0.0.0.0:6334->6334/tcp
labs-redis-1        "docker-entrypoint.s…"   redis               running             0.0.0.0:16379->6379/tcp
```

### 2. 数据库初始化

```bash
# 回到lab01目录
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs/full/lab01

# 运行数据库迁移
uv run alembic upgrade head
```

成功输出示例：
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> abc123, Initial migration
```

### 3. 启动RAG应用

#### 方法1: 开发模式（推荐）

```bash
# 启动开发服务器（支持热重载）
uv run python scripts/run_with_uv.py server --reload
```

#### 方法2: 直接启动

```bash
# 直接运行主程序
uv run python src/main.py
```

#### 方法3: 使用项目脚本

```bash
# 使用预定义的启动脚本
uv run rag-server
```

### 4. 验证启动状态

访问以下URL验证系统状态：

- **健康检查**: http://localhost:8000/api/v1/system/health
- **API文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc

预期健康检查响应：
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "qdrant": "connected",
    "embedding_model": "loaded"
  }
}
```

## 📚 功能操作指南

### 1. 文档上传功能

#### 1.1 通过API上传文档

```bash
# 上传PDF文档
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=测试文档" \
  -F "description=这是一个测试文档"
```

#### 1.2 通过Web界面上传

1. 访问 http://localhost:8000/docs
2. 找到 `POST /api/v1/documents/upload` 接口
3. 点击 "Try it out"
4. 选择PDF文件并填写标题、描述
5. 点击 "Execute" 执行上传

#### 1.3 上传响应示例

```json
{
  "id": "doc_123456",
  "title": "测试文档",
  "description": "这是一个测试文档",
  "filename": "document.pdf",
  "file_size": 1024000,
  "status": "processing",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### 2. 文档管理操作

#### 2.1 查看文档列表

```bash
# 获取所有文档
curl -X GET "http://localhost:8000/api/v1/documents/"

# 分页查询
curl -X GET "http://localhost:8000/api/v1/documents/?page=1&size=10"
```

#### 2.2 查看文档详情

```bash
# 获取特定文档信息
curl -X GET "http://localhost:8000/api/v1/documents/doc_123456"
```

#### 2.3 删除文档

```bash
# 删除文档
curl -X DELETE "http://localhost:8000/api/v1/documents/doc_123456"
```

#### 2.4 查看文档处理状态

文档上传后会经历以下状态：
- `uploading`: 上传中
- `processing`: 处理中（解析、分块、向量化）
- `completed`: 处理完成
- `failed`: 处理失败

### 3. 智能问答系统

#### 3.1 基础问答

```bash
# 发送问题
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是人工智能？",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

#### 3.2 指定文档问答

```bash
# 针对特定文档提问
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "文档中提到的主要观点是什么？",
    "document_ids": ["doc_123456"],
    "max_tokens": 500
  }'
```

#### 3.3 问答响应示例

```json
{
  "answer": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统...",
  "sources": [
    {
      "document_id": "doc_123456",
      "chunk_id": "chunk_789",
      "content": "相关的文档片段内容...",
      "similarity_score": 0.85
    }
  ],
  "processing_time": 1.23,
  "model_used": "doubao-seed-1-6-250615"
}
```

### 4. 语义搜索功能

#### 4.1 搜索文档片段

```bash
# 语义搜索
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "机器学习算法",
    "top_k": 5,
    "similarity_threshold": 0.7
  }'
```

#### 4.2 搜索响应示例

```json
{
  "results": [
    {
      "chunk_id": "chunk_001",
      "document_id": "doc_123456",
      "content": "机器学习是人工智能的一个重要分支...",
      "similarity_score": 0.92,
      "metadata": {
        "page": 1,
        "section": "introduction"
      }
    }
  ],
  "total_results": 5,
  "processing_time": 0.15
}
```

### 5. 系统监控和管理

#### 5.1 系统状态检查

```bash
# 检查系统健康状态
curl -X GET "http://localhost:8000/api/v1/system/health"

# 获取系统信息
curl -X GET "http://localhost:8000/api/v1/system/info"
```

#### 5.2 性能监控

```bash
# 查看系统统计信息
curl -X GET "http://localhost:8000/api/v1/system/stats"
```

响应示例：
```json
{
  "documents_count": 10,
  "chunks_count": 150,
  "total_queries": 25,
  "average_response_time": 1.2,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "model_status": "loaded"
}
```

## 🔧 高级配置

### 1. 嵌入模型切换

#### 1.1 修改配置文件

编辑 `.env` 文件：

```env
# 切换到专门的中文模型
LOCAL_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
LOCAL_EMBEDDING_DIMENSION=768
```

#### 1.2 重启应用

```bash
# 停止应用 (Ctrl+C)
# 重新启动
uv run python src/main.py
```

#### 1.3 验证模型切换

```bash
# 检查当前使用的模型
curl -X GET "http://localhost:8000/api/v1/system/info" | grep embedding_model
```

### 2. 性能优化配置

#### 2.1 调整文档处理参数

编辑 `.env` 文件：

```env
# 文档分块配置
CHUNK_SIZE=800          # 增大分块大小
CHUNK_OVERLAP=100       # 增加重叠大小
MAX_FILE_SIZE=20971520  # 20MB最大文件大小

# 检索配置
TOP_K=10                # 增加检索结果数量
SIMILARITY_THRESHOLD=0.6 # 降低相似度阈值
```

#### 2.2 调整LLM参数

```env
# LLM生成配置
MAX_TOKENS=3000         # 增加最大token数
TEMPERATURE=0.5         # 降低生成温度，提高一致性
```

### 3. 多模型配置

#### 3.1 配置模型池

创建 `config/models.json`：

```json
{
  "embedding_models": {
    "chinese": {
      "model": "shibing624/text2vec-base-chinese",
      "dimension": 768,
      "description": "专门的中文模型"
    },
    "multilingual": {
      "model": "paraphrase-multilingual-MiniLM-L12-v2",
      "dimension": 384,
      "description": "多语言模型"
    }
  }
}
```

## 🐛 故障排除

### 1. 常见问题及解决方案

#### 1.1 模型下载失败

**问题**：首次启动时模型下载超时或失败

**解决方案**：
```bash
# 手动下载模型
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
"
```

#### 1.2 数据库连接失败

**问题**：`connection to server at "localhost" (127.0.0.1), port 15432 failed`

**解决方案**：
```bash
# 检查数据库服务状态
docker-compose ps postgres

# 重启数据库服务
docker-compose restart postgres

# 检查端口占用
lsof -i :15432
```

#### 1.3 Qdrant连接失败

**问题**：`Failed to connect to Qdrant`

**解决方案**：
```bash
# 检查Qdrant服务
docker-compose ps qdrant

# 重启Qdrant
docker-compose restart qdrant

# 验证Qdrant API
curl http://localhost:6333/health
```

#### 1.4 内存不足

**问题**：`RuntimeError: CUDA out of memory` 或系统卡顿

**解决方案**：
```bash
# 使用更小的模型
# 在.env中修改：
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DIMENSION=384

# 或者限制批处理大小
BATCH_SIZE=16
```

### 2. 日志查看和调试

#### 2.1 应用日志

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

#### 2.2 数据库日志

```bash
# 查看PostgreSQL日志
docker-compose logs postgres

# 实时查看日志
docker-compose logs -f postgres
```

#### 2.3 Qdrant日志

```bash
# 查看Qdrant日志
docker-compose logs qdrant

# 检查Qdrant集合状态
curl http://localhost:6333/collections
```

### 3. 性能调优

#### 3.1 系统资源监控

```bash
# 监控系统资源
htop

# 监控Docker容器资源
docker stats
```

#### 3.2 数据库优化

```bash
# 连接到PostgreSQL
docker exec -it labs-postgres-1 psql -U rag_user -d rag_db

# 查看表大小
\dt+

# 查看索引使用情况
SELECT * FROM pg_stat_user_indexes;
```

#### 3.3 向量数据库优化

```bash
# 检查Qdrant集合信息
curl http://localhost:6333/collections/documents

# 查看集合统计
curl http://localhost:6333/collections/documents/cluster
```

## 📊 实验验证

### 1. 功能验证清单

- [ ] 系统成功启动，所有服务正常
- [ ] 嵌入模型成功加载
- [ ] 文档上传功能正常
- [ ] 文档解析和向量化完成
- [ ] 语义搜索返回相关结果
- [ ] 问答系统生成合理回答
- [ ] API接口响应正常

### 2. 性能基准测试

#### 2.1 文档处理性能

```bash
# 测试文档上传和处理时间
time curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@test_document.pdf"
```

#### 2.2 查询响应性能

```bash
# 测试问答响应时间
time curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "测试问题"}'
```

### 3. 中文处理验证

上传中文PDF文档，测试以下功能：
- 中文文本正确解析
- 中文向量化效果
- 中文语义搜索准确性
- 中文问答质量

## 🎓 实验总结

完成本实验后，你将掌握：

1. **RAG系统架构**：理解检索增强生成的工作原理
2. **向量化技术**：掌握文档向量化和相似度搜索
3. **本地模型部署**：学会配置和使用本地嵌入模型
4. **系统集成**：了解多组件系统的集成和配置
5. **性能优化**：掌握系统监控和性能调优方法

## 📚 扩展学习

- 尝试不同的嵌入模型，比较效果差异
- 实现自定义的文档预处理逻辑
- 添加更多的文档格式支持
- 优化检索策略和重排序算法
- 集成更多的大语言模型

## 🆘 获取帮助

如果在实验过程中遇到问题：

1. 查看本文档的故障排除部分
2. 检查系统日志和错误信息
3. 验证环境配置和依赖安装
4. 参考项目README和API文档
5. 寻求助教或同学帮助

祝你实验顺利！🚀