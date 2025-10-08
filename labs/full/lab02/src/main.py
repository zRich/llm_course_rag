"""
RAG系统主应用文件
基于FastAPI构建的检索增强生成系统
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from models.database import init_db, check_db_connection
from api.routes import documents, vectors, qa, system, retrieval
from utils.logger import setup_logger, get_logger

# 设置日志
setup_logger()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("正在启动RAG系统...")
    
    try:
        # 初始化数据库
        logger.info("初始化数据库连接...")
        init_db()
        
        # 检查数据库连接
        if not check_db_connection():
            logger.error("数据库连接失败")
            raise Exception("数据库连接失败")
        
        logger.info("数据库连接成功")
        
        # 检查向量存储连接
        try:
            from services.vector_store import VectorStore
            vector_store = VectorStore()
            if vector_store.health_check():
                logger.info("向量存储连接成功")
            else:
                logger.warning("向量存储连接失败，某些功能可能不可用")
        except Exception as e:
            logger.warning(f"向量存储初始化失败: {e}")
        
        # 检查火山引擎API
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.volcengine_api_key,
                base_url=settings.volcengine_base_url,
                timeout=60.0
            )
            # 使用简单的聊天完成测试连接
            response = client.chat.completions.create(
                model=settings.volcengine_model,
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10
            )
            logger.info("火山引擎API连接成功")
        except Exception as e:
            logger.warning(f"火山引擎API连接失败: {e}")
        
        # 创建必要的目录
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        
        logger.info("RAG系统启动完成")
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("正在关闭RAG系统...")
    logger.info("RAG系统已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    # RAG (检索增强生成) 系统

    这是一个基于FastAPI构建的检索增强生成系统，提供以下功能：

    ## 主要功能
    - **文档管理**: 上传、处理、管理PDF文档
    - **向量化**: 将文档转换为向量并存储到Qdrant
    - **智能问答**: 基于文档内容回答用户问题
    - **向量搜索**: 基于语义相似度搜索相关文档片段
    - **系统管理**: 健康检查、统计信息、系统维护

    ## 技术栈
    - **后端框架**: FastAPI
    - **数据库**: PostgreSQL
    - **向量数据库**: Qdrant
    - **缓存**: Redis
    - **AI模型**: 火山引擎豆包大模型 & 嵌入模型
    - **文档处理**: PyPDF2, pdfplumber
    - **包管理**: uv

    ## API文档
    - **Swagger UI**: `/docs` (当前页面)
    - **ReDoc**: `/redoc`
    - **OpenAPI Schema**: `/openapi.json`
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS if settings.ALLOWED_HOSTS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()
    
    # 记录请求信息
    logger.info(f"请求开始: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        logger.info(
            f"请求完成: {request.method} {request.url} - "
            f"状态码: {response.status_code} - "
            f"处理时间: {process_time:.3f}s"
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        # 记录错误信息
        process_time = time.time() - start_time
        logger.error(
            f"请求失败: {request.method} {request.url} - "
            f"错误: {str(e)} - "
            f"处理时间: {process_time:.3f}s"
        )
        raise


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time()
        }
    )


# 注册路由
app.include_router(documents.router, prefix="/api/v1")
app.include_router(vectors.router, prefix="/api/v1")
app.include_router(qa.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(retrieval.router, prefix="/api/v1")


# 根路径
@app.get("/")
async def root():
    """根路径，返回系统信息"""
    return {
        "success": True,
        "message": "欢迎使用RAG系统",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_prefix": "/api/v1",
        "timestamp": time.time()
    }


# API版本信息
@app.get("/api/v1")
async def api_info():
    """API版本信息"""
    return {
        "success": True,
        "message": "RAG系统 API v1",
        "version": settings.APP_VERSION,
        "endpoints": {
            "documents": "/api/v1/documents",
            "vectors": "/api/v1/vectors", 
            "qa": "/api/v1/qa",
            "system": "/api/v1/system"
        },
        "timestamp": time.time()
    }


# 静态文件服务（如果需要）
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def main():
    """主函数，启动应用"""
    logger.info(f"启动RAG系统 - 环境: {settings.ENVIRONMENT}")
    
    # 开发环境配置
    if settings.ENVIRONMENT == "development":
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            log_level="info",
            access_log=True
        )
    else:
        # 生产环境配置
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            workers=settings.WORKERS,
            log_level="warning",
            access_log=False
        )


if __name__ == "__main__":
    main()