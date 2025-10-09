"""
文档管理API路由
处理文档上传、处理、查询等操作
"""

import os
import tempfile
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Form
from sqlalchemy.orm import Session

from api.schemas import (
    DocumentUploadResponse, DocumentListResponse, DocumentProcessResponse,
    DocumentInfo, DocumentUploadRequest, DocumentProcessRequest,
    ChunkListResponse, ChunkInfo, BaseResponse, ErrorResponse
)
from api.dependencies import (
    get_db, get_document_processor, validate_file_upload,
    validate_document_exists, validate_pagination_params,
    get_upload_directory, cleanup_temp_file
)
from models.document import Document
from models.chunk import Chunk
from services.document_processor import DocumentProcessor
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF或TXT文档文件"),
    title: Optional[str] = Form(None, description="文档标题"),
    description: Optional[str] = Form(None, description="文档描述"),
    db: Session = Depends(get_db),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    上传PDF或TXT文档
    
    - **file**: PDF或TXT文档文件
    - **title**: 文档标题（可选）
    - **description**: 文档描述（可选）
    """
    # 验证文件
    file = validate_file_upload(file)
    
    # 创建临时文件
    upload_dir = get_upload_directory()
    temp_file_path = None
    
    try:
        # 保存上传的文件到临时位置
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename)[1],
            dir=upload_dir
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"文件已保存到临时位置: {temp_file_path}")
        
        # 处理文档
        from pathlib import Path
        document, chunks = document_processor.process_document(
            file_path=Path(temp_file_path),
            db=db
        )
        
        # 更新文档信息
        if title:
            document.title = title
        if description:
            document.description = description
        document.filename = file.filename
        db.commit()
        
        result = {
            "document": document,
            "chunks_created": len(chunks)
        }
        
        # 转换为响应模型
        document_info = DocumentInfo(
            id=str(result["document"].id),
            filename=result["document"].filename,
            title=result["document"].title or "",
            description=result["document"].description or "",
            file_size=result["document"].file_size,
            file_type=result["document"].file_type,
            content_hash=result["document"].content_hash or "",
            char_count=len(result["document"].content) if result["document"].content else 0,
            word_count=len(result["document"].content.split()) if result["document"].content else 0,
            estimated_tokens=result["document"].total_tokens or 0,
            chunk_count=result["document"].total_chunks or 0,
            is_processed=result["document"].status == "completed",
            is_vectorized=result["document"].is_vectorized or False,
            processed_at=result["document"].processed_at,
            vectorized_at=result["document"].vectorized_at,
            created_at=result["document"].created_at,
            updated_at=result["document"].updated_at,
            metadata=str(result["document"].metadata_) if result["document"].metadata_ else ""
        )
        
        return DocumentUploadResponse(
            success=True,
            message=f"文档上传成功，创建了 {result['chunks_created']} 个分块",
            document=document_info
        )
        
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档上传失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        if temp_file_path:
            cleanup_temp_file(temp_file_path)


@router.get("/", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    filename: Optional[str] = Query(None, description="文件名过滤"),
    file_type: Optional[str] = Query(None, description="文件类型过滤"),
    is_processed: Optional[bool] = Query(None, description="是否已处理过滤"),
    is_vectorized: Optional[bool] = Query(None, description="是否已向量化过滤"),
    db: Session = Depends(get_db)
):
    """
    获取文档列表
    
    支持分页和过滤功能
    """
    try:
        # 构建查询
        query = db.query(Document)
        
        # 应用过滤条件
        if filename:
            query = query.filter(Document.filename.ilike(f"%{filename}%"))
        if file_type:
            query = query.filter(Document.file_type == file_type)
        if is_processed is not None:
            query = query.filter(Document.status == ("completed" if is_processed else "pending"))
        if is_vectorized is not None:
            query = query.filter(Document.is_vectorized == is_vectorized)
        
        # 获取总数
        total = query.count()
        
        # 应用分页
        offset = (page - 1) * page_size
        documents = query.order_by(Document.created_at.desc()).offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        document_list = []
        for doc in documents:
            document_info = DocumentInfo(
                id=str(doc.id),
                filename=doc.filename,
                title=doc.title or "",
                description=doc.description or "",
                file_size=doc.file_size,
                file_type=doc.file_type,
                content_hash=doc.content_hash or "",
                char_count=len(doc.content) if doc.content else 0,
                word_count=len(doc.content.split()) if doc.content else 0,
                estimated_tokens=doc.total_tokens or 0,
                chunk_count=doc.total_chunks or 0,
                is_processed=doc.status == "completed",
                is_vectorized=doc.is_vectorized or False,
                processed_at=doc.processed_at,
                vectorized_at=doc.vectorized_at,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                metadata=str(doc.metadata_) if doc.metadata_ else ""
            )
            document_list.append(document_info)
        
        return DocumentListResponse(
            success=True,
            message=f"获取到 {len(document_list)} 个文档",
            documents=document_list,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档列表失败: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentUploadResponse)
def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    获取单个文档详情
    """
    try:
        document = validate_document_exists(document_id, db)
        
        document_info = DocumentInfo(
            id=str(document.id),
            filename=document.filename,
            title=document.title or "",
            description=document.description or "",
            file_size=document.file_size,
            file_type=document.file_type,
            content_hash=document.content_hash or "",
            char_count=len(document.content) if document.content else 0,
            word_count=len(document.content.split()) if document.content else 0,
            estimated_tokens=document.total_tokens or 0,
            chunk_count=document.total_chunks or 0,
            is_processed=document.status == "completed",
            is_vectorized=document.is_vectorized or False,
            processed_at=document.processed_at,
            vectorized_at=document.vectorized_at,
            created_at=document.created_at,
            updated_at=document.updated_at,
            metadata=str(document.metadata_) if document.metadata_ else ""
        )
        
        return DocumentUploadResponse(
            success=True,
            message="获取文档详情成功",
            document=document_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档详情失败: {str(e)}"
        )


@router.post("/{document_id}/reprocess", response_model=DocumentProcessResponse)
async def reprocess_document(
    document_id: str,
    request: DocumentProcessRequest = DocumentProcessRequest(),
    db: Session = Depends(get_db),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    重新处理文档
    
    重新进行文本分块处理
    """
    try:
        document = validate_document_exists(document_id, db)
        
        result = document_processor.reprocess_document(
            document_id=document_id,
            db=db
        )
        
        document, chunks = result
        
        return DocumentProcessResponse(
            success=True,
            message=f"文档重新处理成功，创建了 {len(chunks)} 个分块",
            document_id=document_id,
            chunks_created=len(chunks),
            processing_time=0.0  # 暂时设为0，后续可以添加计时功能
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新处理文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新处理文档失败: {str(e)}"
        )


@router.get("/{document_id}/chunks", response_model=ChunkListResponse)
def get_document_chunks(
    document_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: Session = Depends(get_db)
):
    """
    获取文档的分块列表
    """
    try:
        # 验证文档存在
        validate_document_exists(document_id, db)
        
        # 构建查询
        query = db.query(Chunk).filter(Chunk.document_id == document_id)
        
        # 获取总数
        total = query.count()
        
        # 应用分页
        offset = (page - 1) * page_size
        chunks = query.order_by(Chunk.chunk_index).offset(offset).limit(page_size).all()
        
        # 转换为响应模型
        chunk_list = []
        for chunk in chunks:
            # 处理metadata字段
            metadata_dict = {}
            if chunk.metadata_:
                try:
                    import json
                    if isinstance(chunk.metadata_, str):
                        metadata_dict = json.loads(chunk.metadata_)
                    elif isinstance(chunk.metadata_, dict):
                        metadata_dict = chunk.metadata_
                except:
                    metadata_dict = {}
            
            chunk_info = ChunkInfo(
                id=str(chunk.id),
                document_id=str(chunk.document_id),
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_hash=chunk.content_hash,
                start_position=chunk.start_pos,
                end_position=chunk.end_pos,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                vector_id=chunk.vector_id,
                vector_model=chunk.embedding_model,
                vector_dimensions=chunk.embedding_dimension,
                is_vectorized=chunk.is_embedded == 1,
                vectorized_at=chunk.embedded_at,
                created_at=chunk.created_at,
                metadata=metadata_dict
            )
            chunk_list.append(chunk_info)
        
        return ChunkListResponse(
            success=True,
            message=f"获取到 {len(chunk_list)} 个分块",
            chunks=chunk_list,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档分块失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档分块失败: {str(e)}"
        )


@router.delete("/{document_id}", response_model=BaseResponse)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """
    删除文档及其所有分块
    """
    try:
        # 验证文档存在
        validate_document_exists(document_id, db)
        
        # 删除文档及其分块
        success = document_processor.delete_document(document_id, db)
        
        if success:
            return BaseResponse(
                success=True,
                message="文档删除成功"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档删除失败"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文档失败: {str(e)}"
        )