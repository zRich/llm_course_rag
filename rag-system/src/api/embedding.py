"""Embedding相关API接口"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import tempfile
import shutil
from pathlib import Path

from src.embedding.embedder import TextEmbedder
from src.vector_store.qdrant_client import QdrantVectorStore
from src.vector_store.document_vectorizer import DocumentVectorizer

router = APIRouter(prefix="/embedding", tags=["embedding"])

# 全局变量存储组件实例
_embedder = None
_vector_store = None
_vectorizer = None

def get_embedder():
    """获取向量化器实例"""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder(model_name="BAAI/bge-m3")
    return _embedder

def get_vector_store():
    """获取向量存储实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            host="localhost",
            port=6333,
            timeout=10
        )
    return _vector_store

def get_vectorizer(collection_name: str = "documents"):
    """获取文档向量化管理器实例"""
    global _vectorizer
    if _vectorizer is None or _vectorizer.collection_name != collection_name:
        _vectorizer = DocumentVectorizer(
            embedder=get_embedder(),
            vector_store=get_vector_store(),
            collection_name=collection_name,
            chunk_size=300,
            chunk_overlap=50
        )
    return _vectorizer

# Pydantic模型定义
class EmbeddingRequest(BaseModel):
    """文本向量化请求"""
    text: str = Field(..., description="要向量化的文本")
    model_name: Optional[str] = Field(None, description="使用的模型名称")

class EmbeddingResponse(BaseModel):
    """文本向量化响应"""
    vector: List[float] = Field(..., description="向量结果")
    dimension: int = Field(..., description="向量维度")
    model_name: str = Field(..., description="使用的模型名称")
    processing_time: float = Field(..., description="处理时间（秒）")

class BatchEmbeddingRequest(BaseModel):
    """批量文本向量化请求"""
    texts: List[str] = Field(..., description="要向量化的文本列表")
    model_name: Optional[str] = Field(None, description="使用的模型名称")

class BatchEmbeddingResponse(BaseModel):
    """批量文本向量化响应"""
    vectors: List[List[float]] = Field(..., description="向量结果列表")
    dimension: int = Field(..., description="向量维度")
    model_name: str = Field(..., description="使用的模型名称")
    count: int = Field(..., description="处理的文本数量")
    processing_time: float = Field(..., description="处理时间（秒）")

class SimilarityRequest(BaseModel):
    """相似度计算请求"""
    text1: str = Field(..., description="第一个文本")
    text2: str = Field(..., description="第二个文本")
    model_name: Optional[str] = Field(None, description="使用的模型名称")

class SimilarityResponse(BaseModel):
    """相似度计算响应"""
    similarity: float = Field(..., description="相似度分数")
    model_name: str = Field(..., description="使用的模型名称")
    processing_time: float = Field(..., description="处理时间（秒）")

class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    success: bool = Field(..., description="是否成功")
    file_path: str = Field(..., description="文件路径")
    chunks_count: int = Field(..., description="文本块数量")
    vectors_count: int = Field(..., description="向量数量")
    processing_time: float = Field(..., description="处理时间（秒）")
    error: Optional[str] = Field(None, description="错误信息")

class SearchRequest(BaseModel):
    """文档搜索请求"""
    query: str = Field(..., description="搜索查询")
    collection_name: Optional[str] = Field("documents", description="集合名称")
    limit: Optional[int] = Field(5, description="返回结果数量")
    score_threshold: Optional[float] = Field(0.3, description="分数阈值")

class SearchResult(BaseModel):
    """搜索结果项"""
    score: float = Field(..., description="相似度分数")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    chunk_text: str = Field(..., description="文本块内容")
    chunk_index: int = Field(..., description="文本块索引")
    chunk_length: int = Field(..., description="文本块长度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class SearchResponse(BaseModel):
    """文档搜索响应"""
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(..., description="搜索结果")
    total_results: int = Field(..., description="结果总数")
    processing_time: float = Field(..., description="处理时间（秒）")

class CollectionStatsResponse(BaseModel):
    """集合统计响应"""
    collection_name: str = Field(..., description="集合名称")
    total_vectors: int = Field(..., description="总向量数")
    indexed_vectors: int = Field(..., description="已索引向量数")
    total_points: int = Field(..., description="总点数")
    vector_dimension: int = Field(..., description="向量维度")
    distance_metric: str = Field(..., description="距离度量")
    status: str = Field(..., description="状态")

# API接口实现
@router.post("/encode", response_model=EmbeddingResponse)
async def encode_text(request: EmbeddingRequest):
    """文本向量化接口"""
    try:
        embedder = get_embedder()
        
        import time
        start_time = time.time()
        
        # 执行向量化
        vector = embedder.encode(request.text)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            vector=vector.tolist(),
            dimension=len(vector),
            model_name=embedder.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")

@router.post("/batch_encode", response_model=BatchEmbeddingResponse)
async def batch_encode_texts(request: BatchEmbeddingRequest):
    """批量文本向量化接口"""
    try:
        embedder = get_embedder()
        
        import time
        start_time = time.time()
        
        # 执行批量向量化
        vectors = embedder.batch_encode(request.texts)
        
        processing_time = time.time() - start_time
        
        return BatchEmbeddingResponse(
            vectors=[vector.tolist() for vector in vectors],
            dimension=len(vectors[0]) if vectors else 0,
            model_name=embedder.model_name,
            count=len(vectors),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量向量化失败: {str(e)}")

@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """计算文本相似度接口"""
    try:
        embedder = get_embedder()
        
        import time
        start_time = time.time()
        
        # 计算相似度
        similarity = embedder.similarity(request.text1, request.text2)
        
        processing_time = time.time() - start_time
        
        return SimilarityResponse(
            similarity=float(similarity),
            model_name=embedder.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似度计算失败: {str(e)}")

@router.post("/upload_document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("documents")
):
    """文档上传和向量化接口"""
    try:
        # 检查文件类型
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_extension}。支持的类型: {', '.join(allowed_extensions)}"
            )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # 保存上传的文件
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 获取文档向量化管理器
            vectorizer = get_vectorizer(collection_name)
            
            # 处理文档
            result = vectorizer.process_document(temp_file_path)
            
            return DocumentUploadResponse(
                success=result['success'],
                file_path=result['file_path'],
                chunks_count=result['chunks_count'],
                vectors_count=result['vectors_count'],
                processing_time=result['processing_time'],
                error=result.get('error')
            )
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """文档搜索接口"""
    try:
        vectorizer = get_vectorizer(request.collection_name)
        
        import time
        start_time = time.time()
        
        # 执行搜索
        results = vectorizer.search_documents(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        processing_time = time.time() - start_time
        
        # 转换结果格式
        search_results = [
            SearchResult(
                score=result['score'],
                file_name=result['file_name'],
                file_path=result['file_path'],
                chunk_text=result['chunk_text'],
                chunk_index=result['chunk_index'],
                chunk_length=result['chunk_length'],
                metadata=result.get('metadata', {})
            )
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/collections/{collection_name}/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(collection_name: str):
    """获取集合统计信息接口"""
    try:
        vectorizer = get_vectorizer(collection_name)
        
        # 获取统计信息
        stats = vectorizer.get_collection_stats()
        
        if 'error' in stats:
            raise HTTPException(status_code=404, detail=stats['error'])
        
        return CollectionStatsResponse(
            collection_name=stats['collection_name'],
            total_vectors=stats['total_vectors'],
            indexed_vectors=stats['indexed_vectors'],
            total_points=stats['total_points'],
            vector_dimension=stats['vector_dimension'],
            distance_metric=stats['distance_metric'],
            status=stats['status']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.get("/collections")
async def list_collections():
    """列出所有集合接口"""
    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()
        
        return {
            "collections": collections,
            "total_count": len(collections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合列表失败: {str(e)}")

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """删除集合接口"""
    try:
        vector_store = get_vector_store()
        success = vector_store.delete_collection(collection_name)
        
        if success:
            return {"message": f"集合 {collection_name} 删除成功"}
        else:
            raise HTTPException(status_code=404, detail=f"集合 {collection_name} 不存在或删除失败")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除集合失败: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """获取模型信息接口"""
    try:
        embedder = get_embedder()
        model_info = embedder.get_model_info()
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")