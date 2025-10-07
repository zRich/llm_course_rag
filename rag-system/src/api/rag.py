"""RAG API接口"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging
import time

from ..rag.rag_service import RAGService, RAGRequest, RAGResponse
from ..rag.retriever import DocumentRetriever
from ..rag.qa_generator import QAGenerator
from ..embedding.embedder import TextEmbedder
from ..vector_store.qdrant_client import QdrantVectorStore

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/rag", tags=["RAG"])

# 全局RAG服务实例
_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """获取RAG服务实例"""
    global _rag_service
    if _rag_service is None:
        try:
            # 初始化组件
            embedder = TextEmbedder()
            vector_store = QdrantVectorStore()
            retriever = DocumentRetriever(embedder=embedder, vector_store=vector_store)
            qa_generator = QAGenerator()
            
            # 创建RAG服务
            _rag_service = RAGService(
                embedder=embedder,
                vector_store=vector_store,
                retriever=retriever,
                qa_generator=qa_generator
            )
            logger.info("RAG服务初始化成功")
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise HTTPException(status_code=500, detail=f"RAG服务初始化失败: {str(e)}")
    
    return _rag_service

# Pydantic模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    collection_name: str = Field(default="documents", description="集合名称")
    top_k: int = Field(default=5, description="返回文档数量", ge=1, le=20)
    score_threshold: float = Field(default=0.7, description="相似度阈值", ge=0.0, le=1.0)
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="对话历史")
    include_metadata: bool = Field(default=True, description="是否包含元数据")

class QueryResponse(BaseModel):
    """查询响应模型"""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    retrieved_documents: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]
    followup_questions: List[str]

class BatchQueryRequest(BaseModel):
    """批量查询请求模型"""
    queries: List[QueryRequest] = Field(..., description="查询列表", min_items=1, max_items=10)

class BatchQueryResponse(BaseModel):
    """批量查询响应模型"""
    results: List[QueryResponse]
    total_queries: int
    processing_time: float

class ValidationResponse(BaseModel):
    """验证响应模型"""
    is_valid: bool
    issues: List[str]
    suggestions: List[str]

class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    service_status: str
    components: Dict[str, str]
    vector_store_collections: Optional[int] = None
    vector_store_status: Optional[str] = None
    timestamp: float

@router.post("/query", response_model=QueryResponse, summary="RAG查询")
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> QueryResponse:
    """
    执行RAG查询
    
    - **question**: 用户问题
    - **collection_name**: 向量集合名称
    - **top_k**: 返回的相关文档数量
    - **score_threshold**: 相似度阈值
    - **conversation_history**: 对话历史（可选）
    - **include_metadata**: 是否包含检索文档的元数据
    """
    try:
        logger.info(f"收到RAG查询请求: {request.question[:50]}...")
        
        # 验证问题
        validation = rag_service.validate_query(request.question)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"问题验证失败: {', '.join(validation['issues'])}"
            )
        
        # 创建RAG请求
        rag_request = RAGRequest(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            conversation_history=request.conversation_history,
            include_metadata=request.include_metadata
        )
        
        # 执行查询
        response = await rag_service.query(rag_request)
        
        # 转换为API响应格式
        return QueryResponse(
            question=response.question,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            retrieved_documents=response.retrieved_documents,
            processing_time=response.processing_time,
            metadata=response.metadata,
            followup_questions=response.followup_questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@router.post("/query/sync", response_model=QueryResponse, summary="同步RAG查询")
def query_sync(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> QueryResponse:
    """
    执行同步RAG查询
    
    与异步查询功能相同，但使用同步方式处理
    """
    try:
        logger.info(f"收到同步RAG查询请求: {request.question[:50]}...")
        
        # 验证问题
        validation = rag_service.validate_query(request.question)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"问题验证失败: {', '.join(validation['issues'])}"
            )
        
        # 创建RAG请求
        rag_request = RAGRequest(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            conversation_history=request.conversation_history,
            include_metadata=request.include_metadata
        )
        
        # 执行同步查询
        response = rag_service.query_sync(rag_request)
        
        # 转换为API响应格式
        return QueryResponse(
            question=response.question,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            retrieved_documents=response.retrieved_documents,
            processing_time=response.processing_time,
            metadata=response.metadata,
            followup_questions=response.followup_questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同步RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@router.post("/query/batch", response_model=BatchQueryResponse, summary="批量RAG查询")
def batch_query(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
) -> BatchQueryResponse:
    """
    执行批量RAG查询
    
    - **queries**: 查询请求列表（最多10个）
    """
    try:
        start_time = time.time()
        logger.info(f"收到批量RAG查询请求: {len(request.queries)} 个查询")
        
        # 转换为RAG请求列表
        rag_requests = []
        for query_req in request.queries:
            # 验证每个问题
            validation = rag_service.validate_query(query_req.question)
            if not validation["is_valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"问题验证失败: {query_req.question[:50]}... - {', '.join(validation['issues'])}"
                )
            
            rag_request = RAGRequest(
                question=query_req.question,
                collection_name=query_req.collection_name,
                top_k=query_req.top_k,
                score_threshold=query_req.score_threshold,
                conversation_history=query_req.conversation_history,
                include_metadata=query_req.include_metadata
            )
            rag_requests.append(rag_request)
        
        # 执行批量查询
        responses = rag_service.batch_query(rag_requests)
        
        # 转换为API响应格式
        query_responses = []
        for response in responses:
            query_response = QueryResponse(
                question=response.question,
                answer=response.answer,
                confidence=response.confidence,
                sources=response.sources,
                retrieved_documents=response.retrieved_documents,
                processing_time=response.processing_time,
                metadata=response.metadata,
                followup_questions=response.followup_questions
            )
            query_responses.append(query_response)
        
        processing_time = time.time() - start_time
        
        return BatchQueryResponse(
            results=query_responses,
            total_queries=len(request.queries),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量查询处理失败: {str(e)}")

@router.post("/validate", response_model=ValidationResponse, summary="验证查询问题")
def validate_query(
    question: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> ValidationResponse:
    """
    验证查询问题的有效性
    
    - **question**: 要验证的问题
    """
    try:
        validation = rag_service.validate_query(question)
        return ValidationResponse(
            is_valid=validation["is_valid"],
            issues=validation["issues"],
            suggestions=validation["suggestions"]
        )
    except Exception as e:
        logger.error(f"问题验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证处理失败: {str(e)}")

@router.get("/status", response_model=SystemStatusResponse, summary="获取系统状态")
def get_system_status(
    rag_service: RAGService = Depends(get_rag_service)
) -> SystemStatusResponse:
    """
    获取RAG系统状态信息
    """
    try:
        status = rag_service.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"状态获取失败: {str(e)}")

@router.get("/collections/{collection_name}/stats", summary="获取集合统计信息")
def get_collection_stats(
    collection_name: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    获取指定集合的统计信息
    
    - **collection_name**: 集合名称
    """
    try:
        stats = rag_service.get_collection_stats(collection_name)
        return stats
    except Exception as e:
        logger.error(f"获取集合统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"统计信息获取失败: {str(e)}")

@router.get("/health", summary="RAG服务健康检查")
def health_check() -> Dict[str, Any]:
    """
    RAG服务健康检查
    """
    try:
        # 尝试获取RAG服务
        rag_service = get_rag_service()
        status = rag_service.get_system_status()
        
        return {
            "status": "healthy",
            "service": "rag",
            "timestamp": time.time(),
            "components_status": status.get("components", {})
        }
    except Exception as e:
        logger.error(f"RAG服务健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "service": "rag",
            "timestamp": time.time(),
            "error": str(e)
        }