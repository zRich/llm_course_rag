# 环境配置检查清单

## 课前准备检查清单

### 系统要求检查
- [ ] **操作系统**: macOS 12+, Windows 10+, 或 Ubuntu 20.04+
- [ ] **内存**: 至少8GB RAM (推荐16GB)
- [ ] **存储空间**: 至少5GB可用空间
- [ ] **网络连接**: 稳定的互联网连接

### 基础软件安装
- [ ] **Python 3.12+**: 
  - 安装命令: [根据操作系统选择]
  - 验证命令: `python --version`
  - 预期输出: `Python 3.12.x` 或更高版本

- [ ] **Git**: 
  - 安装命令: [根据操作系统选择]
  - 验证命令: `git --version`
  - 预期输出: `git version 2.x.x`

- [ ] **代码编辑器**: 
  - [ ] VS Code (推荐)
  - [ ] PyCharm
  - [ ] Cursor
  - [ ] 其他: ___________

## 开发环境配置

### Python环境管理
- [ ] **uv工具安装**:
  ```bash
  # macOS/Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Windows
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- [ ] **验证安装**: `uv --version`
- [ ] **预期输出**: `uv 0.x.x`

### 项目初始化
- [ ] **创建项目目录**:
  ```bash
  mkdir exercise
  cd exercise
  ```

- [ ] **初始化uv项目**:
  ```bash
  uv init
  ```

- [ ] **验证项目结构**:
  ```
  exercise/
  ├── pyproject.toml
  ├── README.md
  └── src/
      └── exercise/
          └── __init__.py
  ```

### 依赖管理配置

#### 国内源配置（重要）
- [ ] **配置uv国内源**:
  ```bash
  # 创建配置目录
  mkdir -p ~/.config/uv
  
  # 配置清华大学镜像源
  cat > ~/.config/uv/uv.toml << EOF
  [global]
  index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
  extra-index-url = [
      "https://mirrors.aliyun.com/pypi/simple/",
      "https://pypi.douban.com/simple/"
  ]
  EOF
  ```

- [ ] **验证源配置**:
  ```bash
  # 测试安装速度
  uv pip install requests
  # 预期：安装速度明显提升
  ```

#### 项目依赖安装
- [ ] **安装核心依赖**:
  ```bash
  uv add fastapi
  uv add "uvicorn[standard]"
  uv add pydantic-settings
  uv add python-dotenv
  ```

- [ ] **安装开发依赖**:
  ```bash
  uv add --dev pytest
  uv add --dev ruff
  uv add --dev mypy
  uv add --dev httpx
  ```

- [ ] **验证依赖安装**:
  ```bash
  uv pip list
  ```

## 项目结构创建

### 核心目录结构
- [ ] **创建应用目录**:
  ```bash
  mkdir -p app/{api,core,models,services,utils,schemas}
  ```

- [ ] **创建测试目录**:
  ```bash
  mkdir -p tests/{test_api,test_services,test_utils}
  ```

- [ ] **创建其他目录**:
  ```bash
  mkdir -p {docs,scripts,logs}
  ```

### Python包初始化
- [ ] **创建__init__.py文件**:
  ```bash
  touch app/__init__.py
  touch app/api/__init__.py
  touch app/core/__init__.py
  touch app/models/__init__.py
  touch app/services/__init__.py
  touch app/utils/__init__.py
  touch app/schemas/__init__.py
  touch tests/__init__.py
  touch tests/test_api/__init__.py
  touch tests/test_services/__init__.py
  touch tests/test_utils/__init__.py
  ```

### 配置文件创建
- [ ] **创建环境变量文件**:
  ```bash
  touch .env.example
  cp .env.example .env
  ```

- [ ] **创建Git忽略文件**:
  ```bash
  touch .gitignore
  ```

- [ ] **Git忽略文件内容**:
  ```gitignore
  # Python
  __pycache__/
  *.py[cod]
  *$py.class
  *.so
  .Python
  build/
  develop-eggs/
  dist/
  downloads/
  eggs/
  .eggs/
  lib/
  lib64/
  parts/
  sdist/
  var/
  wheels/
  *.egg-info/
  .installed.cfg
  *.egg

  # Virtual Environment
  .env
  .venv
  env/
  venv/
  ENV/
  env.bak/
  venv.bak/

  # IDE
  .vscode/
  .idea/
  *.swp
  *.swo
  *~

  # Logs
  logs/
  *.log

  # Database
  *.db
  *.sqlite3

  # OS
  .DS_Store
  Thumbs.db
  ```

## FastAPI应用创建

### 主应用文件
- [ ] **创建app/main.py**:
  ```python
  from fastapi import FastAPI
  from app.api.health import router as health_router

  app = FastAPI(
      title="RAG Course API",
      description="企业级RAG系统开发课程",
      version="1.0.0"
  )

  app.include_router(health_router, prefix="/api/v1", tags=["健康检查"])

  @app.get("/")
  async def root():
      return {"message": "欢迎使用RAG课程API"}
  ```

### 健康检查接口
- [ ] **创建app/api/health.py**:
  ```python
  from fastapi import APIRouter
  from datetime import datetime

  router = APIRouter()

  @router.get("/health")
  async def health_check():
      return {
          "status": "healthy",
          "message": "RAG系统运行正常",
          "timestamp": datetime.now()
      }
  ```

### 配置管理
- [ ] **创建app/core/config.py**:
  ```python
  from pydantic_settings import BaseSettings

  class Settings(BaseSettings):
      app_name: str = "RAG Course API"
      app_version: str = "1.0.0"
      debug: bool = True
      
      class Config:
          env_file = ".env"

  settings = Settings()
  ```

## 测试环境配置

### 测试文件创建
- [ ] **创建tests/test_health.py**:
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from app.main import app

  client = TestClient(app)

  def test_root_endpoint():
      response = client.get("/")
      assert response.status_code == 200
      assert response.json()["message"] == "欢迎使用RAG课程API"

  def test_health_check():
      response = client.get("/api/v1/health")
      assert response.status_code == 200
      data = response.json()
      assert data["status"] == "healthy"
  ```

### pytest配置
- [ ] **创建tests/conftest.py**:
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from app.main import app

  @pytest.fixture
  def client():
      return TestClient(app)
  ```

## 开发工具配置

### 代码格式化配置
- [ ] **ruff配置** (在pyproject.toml中):
  ```toml
  [tool.ruff]
  line-length = 88
  target-version = "py311"

  [tool.ruff.lint]
  select = ["E", "F", "I", "N", "W"]
  ignore = []
  ```

### 类型检查配置
- [ ] **mypy配置** (在pyproject.toml中):
  ```toml
  [tool.mypy]
  python_version = "3.12"
  warn_return_any = true
  warn_unused_configs = true
  disallow_untyped_defs = true
  ```

### pytest配置
- [ ] **pytest配置** (在pyproject.toml中):
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  python_classes = ["Test*"]
  python_functions = ["test_*"]
  ```

## 功能验证测试

### 应用启动测试
- [ ] **启动开发服务器**:
  ```bash
  uv run uvicorn app.main:app --reload
  ```

- [ ] **验证启动成功**:
  - 访问: http://localhost:8000
  - 预期响应: `{"message": "欢迎使用RAG课程API"}`

- [ ] **验证健康检查**:
  - 访问: http://localhost:8000/api/v1/health
  - 预期响应: `{"status": "healthy", ...}`

### API文档验证
- [ ] **Swagger UI访问**:
  - 访问: http://localhost:8000/docs
  - 验证文档正常显示

- [ ] **ReDoc访问**:
  - 访问: http://localhost:8000/redoc
  - 验证文档正常显示

### 测试运行验证
- [ ] **运行单元测试**:
  ```bash
  uv run pytest
  ```

- [ ] **验证测试通过**:
  - 所有测试用例通过
  - 无错误或警告

### 代码质量检查
- [ ] **代码格式化**:
  ```bash
  uv run ruff format .
  ```

- [ ] **代码检查**:
  ```bash
  uv run ruff check .
  ```

- [ ] **类型检查**:
  ```bash
  uv run mypy app/
  ```

## 常见问题排查

### Python版本问题
- **问题**: `python: command not found`
- **解决**: 
  - 确认Python已正确安装
  - 检查PATH环境变量
  - 尝试使用`python3`命令

### uv安装问题
- **问题**: `uv: command not found`
- **解决**:
  - 重新运行安装脚本
  - 重启终端
  - 检查PATH环境变量

### 依赖安装问题
- **问题**: 依赖安装失败
- **解决**:
  - 检查网络连接
  - 清除uv缓存: `uv cache clean`
  - 使用国内镜像源

### 应用启动问题
- **问题**: FastAPI应用无法启动
- **解决**:
  - 检查代码语法错误
  - 确认所有依赖已安装
  - 查看错误日志详细信息

### 端口占用问题
- **问题**: `Address already in use`
- **解决**:
  - 更换端口: `--port 8001`
  - 杀死占用进程
  - 重启系统

## 环境验证完成确认

### 最终检查清单
- [ ] Python 3.12+ 正确安装
- [ ] uv工具正确安装
- [ ] 项目结构完整创建
- [ ] 所有依赖正确安装
- [ ] FastAPI应用正常启动
- [ ] 健康检查接口正常工作
- [ ] API文档正常访问
- [ ] 测试用例全部通过
- [ ] 代码质量检查通过

### 环境信息记录
- **Python版本**: ___________
- **uv版本**: ___________
- **操作系统**: ___________
- **代码编辑器**: ___________
- **项目路径**: ___________

### 完成时间记录
- **开始时间**: ___________
- **完成时间**: ___________
- **总耗时**: ___________

---

**恭喜！您已成功完成环境配置，可以开始正式的课程学习了！**

如果在配置过程中遇到任何问题，请：
1. 仔细阅读错误信息
2. 查看常见问题排查部分
3. 搜索相关技术文档
4. 向老师或同学求助