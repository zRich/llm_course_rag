# 企业级RAG项目结构演示

## 项目目录结构

```
exercise/
├── app/                          # 应用主目录
│   ├── __init__.py              # Python包标识
│   ├── main.py                  # FastAPI应用入口
│   ├── api/                     # API路由模块
│   │   ├── __init__.py
│   │   ├── health.py            # 健康检查接口
│   │   ├── auth.py              # 认证接口（后续课程）
│   │   ├── documents.py         # 文档管理接口（后续课程）
│   │   ├── embeddings.py        # 向量化接口（后续课程）
│   │   ├── retrieval.py         # 检索接口（后续课程）
│   │   └── generation.py        # 生成接口（后续课程）
│   ├── core/                    # 核心配置模块
│   │   ├── __init__.py
│   │   ├── config.py            # 应用配置
│   │   ├── logging.py           # 日志配置
│   │   ├── security.py          # 安全配置（后续课程）
│   │   └── database.py          # 数据库配置（后续课程）
│   ├── models/                  # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py              # 基础模型
│   │   ├── document.py          # 文档模型（后续课程）
│   │   ├── embedding.py         # 向量模型（后续课程）
│   │   └── user.py              # 用户模型（后续课程）
│   ├── services/                # 业务逻辑服务
│   │   ├── __init__.py
│   │   ├── document_service.py  # 文档服务（后续课程）
│   │   ├── embedding_service.py # 向量化服务（后续课程）
│   │   ├── retrieval_service.py # 检索服务（后续课程）
│   │   └── llm_service.py       # LLM服务（后续课程）
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── text_processing.py   # 文本处理（后续课程）
│   │   ├── vector_utils.py      # 向量工具（后续课程）
│   │   └── file_utils.py        # 文件工具（后续课程）
│   └── schemas/                 # Pydantic数据模式
│       ├── __init__.py
│       ├── health.py            # 健康检查模式
│       ├── document.py          # 文档模式（后续课程）
│       ├── embedding.py         # 向量模式（后续课程）
│       └── response.py          # 响应模式
├── tests/                       # 测试目录
│   ├── __init__.py
│   ├── conftest.py              # pytest配置
│   ├── test_health.py           # 健康检查测试
│   ├── test_api/                # API测试
│   │   ├── __init__.py
│   │   └── test_endpoints.py
│   ├── test_services/           # 服务测试
│   │   ├── __init__.py
│   │   └── test_document_service.py
│   └── test_utils/              # 工具测试
│       ├── __init__.py
│       └── test_text_processing.py
├── docs/                        # 文档目录
│   ├── api.md                   # API文档
│   ├── deployment.md            # 部署文档
│   └── development.md           # 开发文档
├── scripts/                     # 脚本目录
│   ├── start_dev.sh             # 开发启动脚本
│   ├── run_tests.sh             # 测试运行脚本
│   └── deploy.sh                # 部署脚本
├── logs/                        # 日志目录
│   └── .gitkeep
├── data/                        # 数据目录（后续课程）
│   ├── documents/               # 文档存储
│   ├── embeddings/              # 向量存储
│   └── models/                  # 模型存储
├── .env                         # 环境变量（不提交到git）
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git忽略文件
├── pyproject.toml               # 项目配置和依赖
├── README.md                    # 项目说明
└── Dockerfile                   # Docker配置（后续课程）
```

## 目录结构设计原则

### 1. 分层架构原则
- **API层** (`app/api/`): 处理HTTP请求和响应
- **服务层** (`app/services/`): 业务逻辑处理
- **模型层** (`app/models/`): 数据模型定义
- **工具层** (`app/utils/`): 通用工具函数

### 2. 关注点分离
- **配置管理** (`app/core/`): 集中管理应用配置
- **数据模式** (`app/schemas/`): 定义API输入输出格式
- **测试代码** (`tests/`): 独立的测试目录
- **文档资料** (`docs/`): 项目文档集中管理

### 3. 可扩展性设计
- 模块化设计，便于添加新功能
- 清晰的依赖关系，避免循环依赖
- 标准化的命名约定
- 完整的测试覆盖

## 企业级项目特点

### 1. 标准化结构
```python
# 每个模块都有清晰的职责
app/
├── api/          # 接口层 - 只处理HTTP相关逻辑
├── services/     # 服务层 - 核心业务逻辑
├── models/       # 模型层 - 数据结构定义
└── utils/        # 工具层 - 通用功能函数
```

### 2. 配置管理
```python
# app/core/config.py
class Settings(BaseSettings):
    app_name: str = "RAG Course API"
    debug: bool = False
    database_url: str
    
    class Config:
        env_file = ".env"
```

### 3. 依赖注入
```python
# 通过依赖注入实现松耦合
from app.core.config import settings
from app.services.document_service import DocumentService

def get_document_service() -> DocumentService:
    return DocumentService(settings.database_url)
```

### 4. 测试驱动开发
```python
# tests/test_health.py
def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## RAG系统特定结构

### 1. 文档处理模块
```
app/services/
├── document_service.py      # 文档上传、解析、存储
├── text_processing.py       # 文本清洗、分块
└── file_handler.py          # 文件格式处理
```

### 2. 向量化模块
```
app/services/
├── embedding_service.py     # 文本向量化
├── vector_store.py          # 向量存储管理
└── similarity_search.py     # 相似度搜索
```

### 3. 检索增强模块
```
app/services/
├── retrieval_service.py     # 检索策略
├── ranking_service.py       # 结果排序
└── context_builder.py       # 上下文构建
```

### 4. 生成模块
```
app/services/
├── llm_service.py           # LLM调用服务
├── prompt_template.py       # 提示词模板
└── response_processor.py    # 响应后处理
```

## 开发工作流

### 1. 功能开发流程
1. **需求分析** → 确定功能边界
2. **接口设计** → 定义API规范
3. **模型设计** → 设计数据结构
4. **服务实现** → 编写业务逻辑
5. **测试编写** → 保证代码质量
6. **文档更新** → 维护项目文档

### 2. 代码提交流程
1. **功能分支** → 从main分支创建feature分支
2. **本地开发** → 在feature分支开发功能
3. **测试验证** → 运行单元测试和集成测试
4. **代码审查** → 提交Pull Request
5. **合并主干** → 审查通过后合并到main

### 3. 质量保证
- **代码规范**: 使用ruff进行代码格式化和检查
- **类型检查**: 使用mypy进行静态类型检查
- **测试覆盖**: 保持80%以上的测试覆盖率
- **文档同步**: 代码变更时同步更新文档

## 课堂演示要点

### 1. 目录创建演示
```bash
# 创建项目结构
mkdir -p app/{api,core,models,services,utils,schemas}
mkdir -p tests/{test_api,test_services,test_utils}
mkdir -p {docs,scripts,logs,data}

# 创建__init__.py文件
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
# ... 其他__init__.py文件
```

### 2. 依赖管理演示
```bash
# 使用uv管理依赖
uv init
uv add fastapi uvicorn[standard]
uv add --dev pytest ruff mypy
```

### 3. 配置文件演示
```bash
# 创建配置文件
cp .env.example .env
# 编辑环境变量
```

### 4. 开发服务器启动
```bash
# 启动开发服务器
uv run uvicorn app.main:app --reload
```

## 常见问题与解答

### Q1: 为什么要这样组织目录结构？
**A**: 这种结构遵循了软件工程的最佳实践：
- **分层架构**: 清晰的职责分离
- **模块化设计**: 便于维护和扩展
- **标准化**: 团队成员容易理解和协作

### Q2: 如何处理模块间的依赖关系？
**A**: 遵循依赖倒置原则：
- 高层模块不依赖低层模块
- 通过接口定义依赖关系
- 使用依赖注入管理对象创建

### Q3: 测试代码应该如何组织？
**A**: 测试目录结构应该镜像源代码结构：
- 每个源文件对应一个测试文件
- 使用相同的目录层次结构
- 分离单元测试和集成测试

### Q4: 如何管理不同环境的配置？
**A**: 使用环境变量和配置文件：
- 开发环境: `.env`文件
- 测试环境: `.env.test`文件
- 生产环境: 环境变量或配置管理系统