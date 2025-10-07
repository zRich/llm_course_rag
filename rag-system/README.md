# RAG System - 企业级检索增强生成系统

## 项目简介

本项目是一个企业级的RAG（Retrieval-Augmented Generation）系统，采用现代化的技术栈和架构设计，旨在提供高性能、可扩展的智能问答解决方案。

## 技术架构

### 核心技术栈

- **Python 3.12+**: 现代Python版本，提供更好的性能和类型支持
- **FastAPI**: 高性能异步Web框架
- **PostgreSQL 17**: 主数据库，支持向量扩展
- **Qdrant**: 专业向量数据库
- **Redis**: 缓存和会话管理
- **MinIO**: 对象存储服务
- **Docker**: 容器化部署

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户接口层     │    │    业务逻辑层    │    │    数据存储层    │
│                │    │                │    │                │
│ • Web UI       │────│ • 文档处理      │────│ • PostgreSQL   │
│ • REST API     │    │ • 向量检索      │    │ • Qdrant       │
│ • WebSocket    │    │ • 生成回答      │    │ • Redis        │
│                │    │ • 用户管理      │    │ • MinIO        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 项目结构

```
rag-system/
├── src/                    # 源代码目录
│   ├── api/                # API接口
│   │   └── health.py       # 健康检查接口
│   ├── core/               # 核心业务逻辑
│   ├── models/             # 数据模型
│   ├── services/           # 业务服务
│   └── utils/              # 工具函数
├── tests/                  # 测试代码
├── docs/                   # 项目文档
├── scripts/                # 脚本文件
│   └── verify_environment.py  # 环境验证脚本
├── config/                 # 配置文件
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── embeddings/        # 向量数据
├── logs/                   # 日志文件
├── docker/                 # Docker相关文件
├── pyproject.toml         # 项目配置
├── .env.example           # 环境变量模板
├── .gitignore             # Git忽略文件
├── docker-compose.yml     # Docker编排文件
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境准备

确保系统已安装以下软件：
- Python 3.12+
- Docker & Docker Compose
- Git
- uv (Python包管理器)

### 2. 环境验证

运行环境验证脚本：

```bash
python scripts/verify_environment.py
```

### 3. 项目初始化

```bash
# 克隆项目
git clone <repository-url>
cd rag-system

# 复制环境变量文件
cp .env.example .env

# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 4. 启动服务

#### 开发模式

```bash
# 启动API服务
uvicorn src.api.health:app --reload --host 0.0.0.0 --port 8000
```

#### 生产模式

```bash
# 使用Docker Compose启动所有服务
docker-compose up -d
```

### 5. 验证部署

访问以下端点验证服务状态：

- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 根路径: http://localhost:8000/

## 开发指南

### 代码规范

项目使用以下工具确保代码质量：

```bash
# 代码格式化
black src/ tests/

# 导入排序
isort src/ tests/

# 类型检查
mypy src/

# 运行测试
pytest tests/
```

### 环境变量配置

复制 `.env.example` 为 `.env` 并根据实际环境修改配置：

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db

# 向量数据库配置
QDRANT_URL=http://localhost:6333

# 缓存配置
REDIS_URL=redis://localhost:6379

# 对象存储配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

## 部署说明

### Docker部署

```bash
# 构建镜像
docker build -t rag-system .

# 启动服务栈
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 生产环境注意事项

1. **安全配置**: 修改默认密码和密钥
2. **资源限制**: 根据负载调整容器资源限制
3. **监控告警**: 配置Prometheus和Grafana监控
4. **备份策略**: 定期备份数据库和向量数据
5. **日志管理**: 配置日志轮转和集中收集

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目链接: [https://github.com/yourusername/rag-system](https://github.com/yourusername/rag-system)

## 更新日志

### v0.1.0 (2024-01-XX)

- 初始项目结构
- 基础API框架
- Docker容器化支持
- 环境验证脚本