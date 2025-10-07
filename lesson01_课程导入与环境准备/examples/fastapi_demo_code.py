# FastAPI演示代码示例

"""
这是Lesson 01课堂演示用的完整FastAPI应用代码
包含项目结构和基础功能实现
"""

# ===== app/main.py =====
from fastapi import FastAPI
from app.api.health import router as health_router

# 创建FastAPI应用实例
app = FastAPI(
    title="RAG Course API",
    description="企业级RAG系统开发课程 - 第一课演示应用",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 文档地址
    redoc_url="/redoc"  # ReDoc 文档地址
)

# 包含健康检查路由
app.include_router(health_router, prefix="/api/v1", tags=["健康检查"])

# 根路径欢迎信息
@app.get("/", tags=["根路径"])
async def root():
    return {
        "message": "欢迎使用RAG课程API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# 应用启动入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True  # 开发模式，代码变更自动重载
    )


# ===== app/api/health.py =====
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import platform
import sys

# 创建路由器
router = APIRouter()

# 响应模型定义
class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    message: str
    timestamp: datetime
    system_info: dict

class DetailedHealthResponse(BaseModel):
    """详细健康检查响应模型"""
    status: str
    message: str
    timestamp: datetime
    system_info: dict
    dependencies: dict

# 基础健康检查接口
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    基础健康检查接口
    返回系统运行状态和基本信息
    """
    return HealthResponse(
        status="healthy",
        message="RAG系统运行正常",
        timestamp=datetime.now(),
        system_info={
            "python_version": sys.version,
            "platform": platform.system(),
            "architecture": platform.architecture()[0]
        }
    )

# 详细健康检查接口
@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    详细健康检查接口
    返回系统详细状态信息
    """
    return DetailedHealthResponse(
        status="healthy",
        message="RAG系统运行正常 - 详细信息",
        timestamp=datetime.now(),
        system_info={
            "python_version": sys.version,
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor()
        },
        dependencies={
            "fastapi": "已安装",
            "uvicorn": "已安装",
            "pydantic": "已安装"
        }
    )

# 系统状态检查
@router.get("/status")
async def system_status():
    """
    系统状态检查接口
    用于监控系统运行状态
    """
    return {
        "status": "running",
        "uptime": "正常运行",
        "memory": "内存使用正常",
        "cpu": "CPU使用正常"
    }


# ===== app/core/config.py =====
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = "RAG Course API"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 数据库配置（后续课程使用）
    database_url: Optional[str] = None
    
    # Redis配置（后续课程使用）
    redis_url: Optional[str] = None
    
    # 向量数据库配置（后续课程使用）
    qdrant_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 创建配置实例
settings = Settings()


# ===== app/core/logging.py =====
from loguru import logger
import sys

def setup_logging():
    """配置日志系统"""
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件输出
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG"
    )
    
    return logger


# ===== tests/test_health.py =====
import pytest
from fastapi.testclient import TestClient
from app.main import app

# 创建测试客户端
client = TestClient(app)

def test_root_endpoint():
    """测试根路径接口"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "欢迎使用RAG课程API"
    assert data["version"] == "1.0.0"

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "RAG系统运行正常"
    assert "timestamp" in data
    assert "system_info" in data

def test_detailed_health_check():
    """测试详细健康检查接口"""
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "dependencies" in data
    assert "system_info" in data

def test_system_status():
    """测试系统状态接口"""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


# ===== .env.example =====
"""
# 应用配置
APP_NAME=RAG Course API
APP_VERSION=1.0.0
DEBUG=true

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置（后续课程使用）
# DATABASE_URL=postgresql://user:password@localhost:5432/rag_db

# Redis配置（后续课程使用）
# REDIS_URL=redis://localhost:6379

# 向量数据库配置（后续课程使用）
# QDRANT_URL=http://localhost:6333
"""


# ===== pyproject.toml =====
"""
[project]
name = "exercise"
version = "1.0.0"
description = "企业级RAG系统开发课程"
authors = [
    {name = "RAG Course Team"}
]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "pytest>=7.4.0",
    "httpx>=0.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
"""


# ===== 启动脚本示例 =====
"""
# 开发环境启动
uv run python -m app.main

# 或者使用uvicorn直接启动
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .

# 代码检查
uv run ruff check .
"""

# ===== 课堂演示要点 =====
"""
1. 项目结构演示
   - 展示清晰的目录组织
   - 解释每个文件的作用
   - 强调企业级项目规范

2. FastAPI特性演示
   - 自动API文档生成
   - 类型注解和数据验证
   - 异步支持
   - 高性能特性

3. 开发工具演示
   - uv的快速依赖管理
   - 热重载功能
   - 测试驱动开发
   - 代码质量工具

4. 实际运行演示
   - 启动应用
   - 访问API文档
   - 测试接口功能
   - 查看日志输出
"""