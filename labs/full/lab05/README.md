# RAG (检索增强生成) 系统

一个基于FastAPI构建的完整RAG系统，支持PDF文档处理、向量化存储、语义搜索和智能问答。

## 🚀 功能特性

### 核心功能
- **📄 文档管理**: 支持PDF文档上传、解析、分块处理
- **🔍 向量化**: 使用豆包嵌入模型将文档转换为向量
- **💾 向量存储**: 基于Qdrant的高性能向量数据库
- **🤖 智能问答**: 基于检索增强生成的问答系统
- **🔎 语义搜索**: 基于向量相似度的文档片段搜索

### 技术特性
- **⚡ 高性能**: 异步处理，支持并发请求
- **🔧 可扩展**: 模块化设计，易于扩展和维护
- **📊 监控**: 完整的日志记录和系统监控
- **🐳 容器化**: Docker和Docker Compose支持
- **📚 文档**: 完整的API文档和使用说明

## 🏗️ 系统架构

```
RAG系统
├── 文档处理层
│   ├── PDF解析 (PyPDF2, pdfplumber)
│   ├── 文本分块 (智能分块策略)
│   └── 内容清理和预处理
├── 向量化层
│   ├── 文本嵌入 (豆包嵌入模型)
│   ├── 向量存储 (Qdrant)
│   └── 相似度搜索
├── 问答层
│   ├── 检索增强 (RAG)
│   ├── 上下文构建
│   └── 回答生成 (豆包大模型)
└── API服务层
    ├── RESTful API (FastAPI)
    ├── 数据验证 (Pydantic)
    └── 异常处理
```

## 🛠️ 技术栈

### 后端框架
- **FastAPI**: 现代、快速的Web框架
- **SQLAlchemy**: ORM和数据库操作
- **Alembic**: 数据库迁移工具
- **Pydantic**: 数据验证和序列化

### 数据存储
- **PostgreSQL**: 关系型数据库
- **Qdrant**: 向量数据库
- **Redis**: 缓存和会话存储

### AI和机器学习
- **火山引擎豆包大模型**: 大语言模型和Embeddings
- **OpenAI SDK**: 兼容的API客户端
- **NumPy**: 数值计算
- **scikit-learn**: 机器学习工具

### 文档处理
- **PyPDF2**: PDF文档解析
- **pdfplumber**: 高级PDF处理
- **NLTK**: 自然语言处理

### 开发工具
- **Docker**: 容器化部署
- **pytest**: 单元测试
- **Black**: 代码格式化
- **Rich**: 命令行界面美化

## 📦 快速开始

### 1. 环境要求

- Python 3.12+
- Docker 和 Docker Compose
- uv (Python包管理器)

### 2. 项目克隆

```bash
git clone <repository-url>
cd rag-lab01
```

### 3. 开发环境设置

#### 方法1: 自动设置 (推荐)

使用uv管理依赖:

```bash
# 安装uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 运行自动设置脚本
python scripts/setup_dev_uv.py
```

#### 方法2: 手动设置

1. **安装uv**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者使用pip
pip install uv
```

2. **同步依赖**:
```bash
uv sync
# 安装开发依赖
uv sync --extra dev
```

3. **配置环境变量**:
```bash
cp .env.example .env
# 编辑.env文件，填入你的配置
```

编辑 `.env` 文件，设置必要的配置：

```env
# 嵌入模型配置 (推荐本地模式，优化中文支持)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LOCAL_EMBEDDING_DIMENSION=384

# 火山引擎豆包大模型配置 (可选，用于API模式)
VOLCENGINE_API_KEY=your-volcengine-api-key
VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCENGINE_MODEL=doubao-seed-1-6-250615
VOLCENGINE_EMBEDDING_MODEL=doubao-embedding-v1
VOLCENGINE_MAX_TOKENS=2000
VOLCENGINE_TEMPERATURE=0.7

# 数据库配置
DATABASE_URL=postgresql://rag_user:rag_password@localhost:15432/rag_db
REDIS_URL=redis://localhost:16379/0

# Qdrant配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

4. **安装本地嵌入模型依赖** (推荐):
```bash
# 安装sentence-transformers用于本地嵌入模型
uv add sentence-transformers
```

5. **启动数据库服务**:
```bash
# 启动数据库服务
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs
docker-compose up -d postgres redis qdrant
```

6. **运行数据库迁移**:
```bash
uv run alembic upgrade head
```

### 4. 启动应用

#### 使用uv运行

```bash
# 开发模式 (热重载)
uv run python scripts/run_with_uv.py server --reload

# 或者直接运行
uv run python src/main.py

# 使用项目脚本
uv run rag-server
```

应用将在 `http://localhost:8000` 启动。

### 5. 访问应用

- **API文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/v1/system/health

## 开发工具

### 使用uv管理依赖

```bash
# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 移除依赖
uv remove <package-name>

# 同步依赖
uv sync

# 更新锁文件
uv lock

# 查看依赖树
uv tree
```

### 代码质量工具

```bash
# 代码格式化
uv run black src/
uv run isort src/

# 代码检查
uv run flake8 src/

# 类型检查
uv run mypy src/

# 运行所有检查
uv run python scripts/run_with_uv.py format
uv run python scripts/run_with_uv.py lint
uv run python scripts/run_with_uv.py typecheck
```

### 运行测试

```bash
# 运行系统测试
uv run python scripts/test_system.py

# 运行pytest测试
uv run pytest

# 使用脚本运行测试
uv run python scripts/run_with_uv.py test
```

运行完整的系统测试：

```bash
python scripts/test_system.py
```

这个脚本会测试：
- 系统健康状态
- 文档上传和处理
- 向量化和搜索
- 问答功能

## 📖 API使用示例

### 上传文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

### 向量化文档

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/vectorize" \
     -H "Content-Type: application/json" \
     -d '{"document_ids": ["doc-id"]}'
```

### 搜索文档

```bash
curl -X POST "http://localhost:8000/api/v1/vectors/search" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "搜索内容",
           "limit": 5,
           "filters": [
             {"op": "eq", "field": "document_title", "value": "指南"},
             {"op": "exists", "field": "metadata.category"}
           ]
         }'
```

### 问答

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "这个文档讲了什么？"}'
```

### 删除文档（级联删除向量）

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/{document_id}"
```

说明：Lab05 起点已实现“级联删除”，当调用文档删除接口时，将先清理该文档在向量存储中的所有向量，再删除数据库中的文档与分块记录，确保数据一致性。

## 🐳 基础服务部署

本实验的应用直接在操作系统上运行，但需要使用Docker Compose启动基础服务（PostgreSQL、Redis、Qdrant）：

```bash
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs
docker-compose up -d postgres redis qdrant
```

## 📁 项目结构

```
lab01/
├── .env.example           # 环境变量模板
├── .gitignore            # Git忽略文件

├── README.md             # 项目文档
├── alembic.ini           # 数据库迁移配置
├── alembic/              # 数据库迁移脚本
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── pyproject.toml        # Python项目配置和依赖
├── uv.lock              # uv依赖锁定文件
├── scripts/              # 脚本工具
│   ├── run_with_uv.py    # 使用uv运行应用的脚本
│   ├── setup_dev.py      # 开发环境设置脚本
│   ├── setup_dev_uv.py   # 使用uv的开发环境设置脚本
│   └── test_system.py    # 系统测试脚本
├── src/                  # 源代码
│   ├── __init__.py
│   ├── main.py           # 应用入口
│   ├── api/              # API路由和模式
│   │   ├── __init__.py
│   │   ├── dependencies.py # 依赖注入
│   │   ├── routes/       # API路由
│   │   └── schemas.py    # 数据模式
│   ├── config/           # 配置文件
│   │   ├── __init__.py
│   │   └── settings.py   # 应用设置
│   ├── models/           # 数据模型
│   │   ├── __init__.py
│   │   ├── chunk.py      # 文档块模型
│   │   ├── database.py   # 数据库配置
│   │   └── document.py   # 文档模型
│   ├── services/         # 业务逻辑
│   │   ├── __init__.py
│   │   ├── document_processor.py # 文档处理服务
│   │   ├── embedding_service.py  # 嵌入服务
│   │   ├── pdf_parser.py         # PDF解析器
│   │   ├── qa_service.py         # 问答服务
│   │   ├── text_splitter.py      # 文本分割器
│   │   ├── vector_service.py     # 向量服务
│   │   └── vector_store.py       # 向量存储
│   └── utils/            # 工具函数
│       ├── __init__.py
│       └── logger.py     # 日志工具
└── tests/                # 测试文件
```

## 🔧 配置说明

### 嵌入模型配置

本系统支持两种嵌入模型模式，**推荐课堂实验使用本地模式**：

#### 1. 本地模式 (推荐)
```bash
# 设置为本地模式 (推荐中文RAG实验)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LOCAL_EMBEDDING_DIMENSION=384
```

**支持的本地模型** (按中文支持程度排序)：
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384维) - **推荐** 多语言释义模型，优秀的中英文支持
- `shibing624/text2vec-base-chinese` (768维) - 专门的中文嵌入模型，中文语义理解能力强
- `BAAI/bge-small-zh-v1.5` (512维) - BGE中文小模型，高质量中文嵌入，性能优异
- `sentence-transformers/distiluse-base-multilingual-cased` (512维) - 多语言通用模型，支持中文，平衡性能和速度
- `sentence-transformers/all-MiniLM-L6-v2` (384维) - 轻量级多语言模型，适合快速实验，基础中文支持
- `sentence-transformers/all-mpnet-base-v2` (768维) - 高质量英文模型，性能较好，有限中文支持

**本地模式优势**：
- ✅ 无需API密钥，适合课堂环境
- ✅ 离线运行，不依赖网络
- ✅ 成本低，无API调用费用
- ✅ 响应快，本地计算

#### 2. API模式
```bash
# 设置为API模式
EMBEDDING_PROVIDER=api
VOLCENGINE_API_KEY=your-api-key
VOLCENGINE_EMBEDDING_MODEL=doubao-embedding-v1
```

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `EMBEDDING_PROVIDER` | 嵌入模型提供者 | `local` |
| `LOCAL_EMBEDDING_MODEL` | 本地嵌入模型名称 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `LOCAL_EMBEDDING_DIMENSION` | 本地模型向量维度 | `384` |
| `VOLCENGINE_API_KEY` | 火山引擎API密钥 | 可选 |
| `VOLCENGINE_BASE_URL` | 火山引擎API地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLCENGINE_MODEL` | 豆包大模型名称 | `doubao-seed-1-6-250615` |
| `VOLCENGINE_EMBEDDING_MODEL` | 豆包嵌入模型名称 | `doubao-embedding-v1` |
| `DATABASE_URL` | 数据库连接URL | `postgresql://...` |
| `REDIS_URL` | Redis连接URL | `redis://localhost:16379/0` |
| `QDRANT_HOST` | Qdrant主机地址 | `localhost` |
| `QDRANT_PORT` | Qdrant端口 | `6333` |
| `CHUNK_SIZE` | 文本分块大小 | `1000` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `200` |

### 文档处理配置

- **支持格式**: PDF
- **最大文件大小**: 10MB
- **分块策略**: 智能分块，支持句子和段落边界
- **向量维度**: 根据选择的模型自动设置

## 🚨 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查PostgreSQL服务是否启动
   - 验证数据库连接字符串
   - 确认数据库用户权限

2. **向量存储连接失败**
   - 检查Qdrant服务状态
   - 验证主机和端口配置
   - 查看Qdrant日志

3. **火山引擎API错误**
   - 验证API密钥有效性
   - 检查API配额和限制
   - 确认网络连接和API地址

4. **文档上传失败**
   - 检查文件格式和大小
   - 验证上传目录权限
   - 查看应用日志

### 日志查看

```bash
# 查看应用日志
tail -f logs/rag_system.log

# 查看基础服务日志
cd /Users/richzhao/dev/llm_courses/courses/11_rag/labs
docker-compose logs -f postgres redis qdrant
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Qdrant](https://qdrant.tech/) - 向量数据库
- [火山引擎](https://www.volcengine.com/) - 豆包大模型和API
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL工具包

## 📞 支持

如果您有任何问题或建议，请：

1. 查看[常见问题](#故障排除)
2. 搜索现有的[Issues](../../issues)
3. 创建新的[Issue](../../issues/new)

---

**Happy Coding! 🚀**