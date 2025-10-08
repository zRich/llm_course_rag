"""
系统管理API路由
处理健康检查、统计信息等系统功能
"""

import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import (
    HealthCheckResponse, SystemStatsResponse, SystemStats, BaseResponse
)
from api.dependencies import (
    get_db, get_vector_service, get_qa_service, get_document_processor,
    check_service_health
)
from models.database import check_db_connection
from models.document import Document
from models.chunk import Chunk
from services.vector_service import VectorService
from services.qa_service import QAService
from services.document_processor import DocumentProcessor
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/system", tags=["系统管理"])

# 应用启动时间
app_start_time = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    系统健康检查
    
    检查各个组件的运行状态
    """
    try:
        components = {}
        
        # 检查数据库连接
        try:
            db_healthy = check_db_connection()
            components["database"] = "healthy" if db_healthy else "unhealthy"
        except Exception as e:
            components["database"] = f"error: {str(e)}"
        
        # 检查向量存储
        try:
            from services.vector_store import VectorStore
            vector_store = VectorStore()
            vector_healthy = vector_store.health_check()
            components["vector_store"] = "healthy" if vector_healthy else "unhealthy"
        except Exception as e:
            components["vector_store"] = f"error: {str(e)}"
        
        # 检查嵌入服务
        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            # 简单测试嵌入服务
            test_result = embedding_service.get_embedding("test")
            components["embedding_service"] = "healthy" if test_result else "unhealthy"
        except Exception as e:
            components["embedding_service"] = f"error: {str(e)}"
        
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
            components["volcengine_api"] = "healthy" if response else "unhealthy"
        except Exception as e:
            components["volcengine_api"] = f"error: {str(e)}"
        
        # 计算运行时间
        uptime = time.time() - app_start_time
        
        # 判断整体状态
        overall_status = "healthy"
        for component, status in components.items():
            if not status.startswith("healthy"):
                overall_status = "degraded"
                break
        
        return HealthCheckResponse(
            success=True,
            message="健康检查完成",
            status=overall_status,
            version=settings.APP_VERSION,
            uptime=uptime,
            components=components
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthCheckResponse(
            success=False,
            message=f"健康检查失败: {str(e)}",
            status="unhealthy",
            version=settings.APP_VERSION,
            uptime=time.time() - app_start_time,
            components={"error": str(e)}
        )


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db: Session = Depends(get_db),
    vector_service: VectorService = Depends(get_vector_service),
    qa_service: QAService = Depends(get_qa_service),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    获取系统统计信息
    
    包括数据库、向量存储、服务等统计数据
    """
    try:
        # 数据库统计
        database_stats = {
            "total_documents": db.query(Document).count(),
            "processed_documents": db.query(Document).filter(Document.is_processed == True).count(),
            "vectorized_documents": db.query(Document).filter(Document.is_vectorized == True).count(),
            "total_chunks": db.query(Chunk).count(),
            "vectorized_chunks": db.query(Chunk).filter(Chunk.is_embedded == 1).count(),
        }
        
        # 向量存储统计
        try:
            vector_stats = vector_service.get_vectorization_stats()
            vector_store_stats = {
                "collection_exists": vector_stats.get("collection_exists", False),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "collection_info": vector_stats.get("collection_info", {}),
            }
        except Exception as e:
            vector_store_stats = {"error": str(e)}
        
        # 服务统计
        try:
            qa_stats = qa_service.get_service_stats()
            service_stats = {
                "qa_service": qa_stats,
                "uptime": time.time() - app_start_time,
                "app_version": settings.APP_VERSION,
            }
        except Exception as e:
            service_stats = {"error": str(e)}
        
        # 系统健康状态
        system_health = {
            "database_connection": check_db_connection(),
            "vector_store_connection": vector_service.vector_store.health_check(),
            "memory_usage": "N/A",  # 可以添加内存使用统计
            "disk_usage": "N/A",    # 可以添加磁盘使用统计
        }
        
        stats = SystemStats(
            database_stats=database_stats,
            vector_store_stats=vector_store_stats,
            service_stats=service_stats,
            system_health=system_health
        )
        
        return SystemStatsResponse(
            success=True,
            message="获取系统统计信息成功",
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"获取系统统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统统计信息失败: {str(e)}"
        )


@router.post("/reset", response_model=BaseResponse)
async def reset_system(
    confirm: bool = False,
    db: Session = Depends(get_db),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    重置系统
    
    清空所有数据（需要确认）
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请设置 confirm=true 来确认重置操作"
        )
    
    try:
        # 清空向量存储
        await vector_service.vector_store.clear_collection()
        
        # 清空数据库
        db.query(Chunk).delete()
        db.query(Document).delete()
        db.commit()
        
        logger.info("系统重置完成")
        
        return BaseResponse(
            success=True,
            message="系统重置完成，所有数据已清空"
        )
        
    except Exception as e:
        logger.error(f"系统重置失败: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"系统重置失败: {str(e)}"
        )


@router.get("/info", response_model=BaseResponse)
async def get_system_info():
    """
    获取系统基本信息
    """
    try:
        info = {
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "start_time": datetime.fromtimestamp(app_start_time).isoformat(),
            "uptime_seconds": time.time() - app_start_time,
            "supported_file_types": [".pdf", ".txt"],
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.volcengine_embedding_model,
            "llm_model": settings.volcengine_model,
        }
        
        return BaseResponse(
            success=True,
            message="获取系统信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统信息失败: {str(e)}"
        )


@router.post("/maintenance", response_model=BaseResponse)
async def maintenance_mode(
    enable: bool = True
):
    """
    维护模式开关
    
    启用或禁用维护模式
    """
    try:
        # 这里可以实现维护模式的逻辑
        # 例如设置全局标志、拒绝某些请求等
        
        status_text = "启用" if enable else "禁用"
        
        return BaseResponse(
            success=True,
            message=f"维护模式已{status_text}"
        )
        
    except Exception as e:
        logger.error(f"设置维护模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置维护模式失败: {str(e)}"
        )


@router.get("/logs", response_model=BaseResponse)
async def get_recent_logs(
    lines: int = 100
):
    """
    获取最近的日志
    
    返回最近的系统日志
    """
    try:
        # 这里可以实现读取日志文件的逻辑
        # 由于日志配置可能不同，这里只返回一个示例响应
        
        return BaseResponse(
            success=True,
            message=f"获取最近 {lines} 行日志成功"
        )
        
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取日志失败: {str(e)}"
        )