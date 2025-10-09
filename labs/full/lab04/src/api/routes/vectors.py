"""
向量化API路由
处理文档向量化、向量搜索等操作
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from api.schemas import (
    VectorizeRequest, VectorizeResponse, SearchRequest, SearchResponse,
    SearchResult, BaseResponse
)
from api.dependencies import (
    get_db, get_vector_service, validate_document_exists,
    validate_search_params
)
from services.vector_service import VectorService
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/vectors", tags=["向量化"])


@router.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_documents(
    request: VectorizeRequest = VectorizeRequest(),
    db: Session = Depends(get_db),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    对文档进行向量化处理
    
    - **document_ids**: 要处理的文档ID列表，为空则处理所有未向量化文档
    - **force_revectorize**: 是否强制重新向量化已处理的文档
    """
    try:
        # 验证文档存在（如果指定了文档ID）
        if request.document_ids:
            for doc_id in request.document_ids:
                validate_document_exists(doc_id, db)
        
        # 执行向量化
        result = await vector_service.vectorize_documents(
            document_ids=request.document_ids,
            force_revectorize=request.force_revectorize
        )
        
        return VectorizeResponse(
            success=True,
            message=f"向量化完成，处理了 {result['processed_documents']} 个文档，{result['processed_chunks']} 个分块",
            processed_documents=result["processed_documents"],
            processed_chunks=result["processed_chunks"],
            processing_time=result["processing_time"],
            failed_documents=result.get("failed_documents", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"向量化处理失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量化处理失败: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_vectors(
    request: SearchRequest,
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    向量搜索
    
    根据查询文本搜索相似的文档分块
    """
    try:
        # 验证搜索参数
        validated_params = validate_search_params(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        # 执行搜索
        results = vector_service.search_similar_chunks(
            query=validated_params["query"],
            document_ids=request.document_ids,
            limit=validated_params["limit"],
            score_threshold=validated_params["score_threshold"]
        )
        
        # 转换为响应模型
        search_results = []
        for result in results:
            search_result = SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_filename=result["document_filename"],
                document_title=result.get("document_title"),
                chunk_index=result["chunk_index"],
                content=result["content"],
                score=result["score"],
                start_position=result["start_pos"],
                end_position=result["end_pos"],
                metadata=result.get("metadata", {})
            )
            search_results.append(search_result)
        
        return SearchResponse(
            success=True,
            message=f"搜索完成，找到 {len(search_results)} 个相关结果",
            query=validated_params["query"],
            results=search_results,
            total_found=len(search_results),
            processing_time=0.0  # 暂时设为0，后续可以添加计时功能
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量搜索失败: {str(e)}"
        )


@router.get("/search", response_model=SearchResponse)
async def search_vectors_get(
    query: str = Query(..., description="搜索查询"),
    document_ids: Optional[List[int]] = Query(None, description="限制搜索的文档ID列表"),
    limit: int = Query(10, ge=1, le=100, description="返回结果数量"),
    score_threshold: float = Query(0.5, ge=0.0, le=1.0, description="相似度阈值"),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    向量搜索（GET方法）
    
    根据查询文本搜索相似的文档分块
    """
    try:
        # 验证搜索参数
        validated_params = validate_search_params(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # 执行搜索
        results = vector_service.search_similar_chunks(
            query=validated_params["query"],
            document_ids=document_ids,
            limit=validated_params["limit"],
            score_threshold=validated_params["score_threshold"]
        )
        
        # 转换为响应模型
        search_results = []
        for result in results:
            search_result = SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                document_filename=result["document_filename"],
                document_title=result.get("document_title"),
                chunk_index=result["chunk_index"],
                content=result["content"],
                score=result["score"],
                start_position=result["start_pos"],
                end_position=result["end_pos"],
                metadata=result.get("metadata", {})
            )
            search_results.append(search_result)
        
        return SearchResponse(
            success=True,
            message=f"搜索完成，找到 {len(search_results)} 个相关结果",
            query=validated_params["query"],
            results=search_results,
            total_found=len(search_results),
            processing_time=0.0  # 暂时设为0，后续可以添加计时功能
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量搜索失败: {str(e)}"
        )


@router.delete("/document/{document_id}", response_model=BaseResponse)
async def delete_document_vectors(
    document_id: int,
    db: Session = Depends(get_db),
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    删除指定文档的所有向量
    """
    try:
        # 验证文档存在
        validate_document_exists(document_id, db)
        
        # 删除向量
        result = await vector_service.delete_document_vectors(document_id)
        
        return BaseResponse(
            success=True,
            message=f"删除文档向量成功，删除了 {result['deleted_count']} 个向量"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档向量失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文档向量失败: {str(e)}"
        )


@router.post("/reindex", response_model=VectorizeResponse)
async def reindex_all_documents(
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    重新索引所有文档
    
    删除现有向量并重新创建所有文档的向量
    """
    try:
        result = await vector_service.reindex_all_documents()
        
        return VectorizeResponse(
            success=True,
            message=f"重新索引完成，处理了 {result['processed_documents']} 个文档，{result['processed_chunks']} 个分块",
            processed_documents=result["processed_documents"],
            processed_chunks=result["processed_chunks"],
            processing_time=result["processing_time"],
            failed_documents=result.get("failed_documents", [])
        )
        
    except Exception as e:
        logger.error(f"重新索引失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新索引失败: {str(e)}"
        )


@router.get("/stats", response_model=BaseResponse)
async def get_vector_stats(
    vector_service: VectorService = Depends(get_vector_service)
):
    """
    获取向量化统计信息
    """
    try:
        stats = await vector_service.get_vectorization_stats()
        
        return BaseResponse(
            success=True,
            message="获取向量化统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取向量化统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取向量化统计信息失败: {str(e)}"
        )