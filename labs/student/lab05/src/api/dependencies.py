"""
API依赖项
提供数据库会话、服务实例等依赖注入
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, UploadFile
from sqlalchemy.orm import Session
import os

from models.database import get_db
from services.document_processor import DocumentProcessor
from services.vector_service import VectorService
from services.qa_service import QAService
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.keyword_service import KeywordSearchService
from services.rerank_service import RerankService
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局服务实例
_document_processor: Optional[DocumentProcessor] = None
_vector_service: Optional[VectorService] = None
_qa_service: Optional[QAService] = None
_embedding_service: Optional[EmbeddingService] = None
_vector_store: Optional[VectorStore] = None
_keyword_service: Optional[KeywordSearchService] = None
_rerank_service: Optional[RerankService] = None


def get_document_processor(db: Session = Depends(get_db)) -> DocumentProcessor:
    """获取文档处理器实例"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor


def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_vector_store() -> VectorStore:
    """获取向量存储实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_vector_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store)
) -> VectorService:
    """获取向量服务实例"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service


def get_qa_service(
    vector_service: VectorService = Depends(get_vector_service)
) -> QAService:
    """获取问答服务实例"""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service


def get_keyword_service() -> KeywordSearchService:
    """获取关键词检索服务实例"""
    global _keyword_service
    if _keyword_service is None:
        _keyword_service = KeywordSearchService()
    return _keyword_service


def get_rerank_service() -> RerankService:
    """获取重排序服务实例"""
    global _rerank_service
    if _rerank_service is None:
        _rerank_service = RerankService()
    return _rerank_service


def validate_file_upload(file: UploadFile) -> UploadFile:
    """验证上传文件"""
    # 检查文件类型
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件名不能为空"
        )
    
    # 检查文件扩展名
    allowed_extensions = {'.pdf', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_extensions)}"
        )
    
    # 检查文件大小
    if hasattr(file, 'size') and file.size:
        max_size = settings.max_file_size  # 使用配置中的字节值
        if file.size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件大小超过限制: {max_size // (1024*1024)}MB"
            )
    
    return file


def validate_document_exists(
    document_id: int,
    db: Session = Depends(get_db)
):
    """验证文档是否存在"""
    from models.document import Document
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_id}"
        )
    return document


def validate_pagination_params(page: int = 1, page_size: int = 20):
    """验证分页参数"""
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="页码必须大于0"
        )
    
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="每页大小必须在1-100之间"
        )
    
    return {"page": page, "page_size": page_size}


def validate_search_params(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.5
):
    """验证搜索参数"""
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="搜索查询不能为空"
        )
    
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="返回结果数量必须在1-100之间"
        )
    
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="相似度阈值必须在0.0-1.0之间"
        )
    
    return {
        "query": query.strip(),
        "limit": limit,
        "score_threshold": score_threshold
    }


def validate_qa_params(
    question: str,
    top_k: int = 10,
    score_threshold: float = 0.7,
    context_size: int = 2
):
    """验证问答参数"""
    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="问题不能为空"
        )
    
    if len(question.strip()) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="问题长度不能超过1000个字符"
        )
    
    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="检索结果数量必须在1-50之间"
        )
    
    if score_threshold < 0.0 or score_threshold > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="相似度阈值必须在0.0-1.0之间"
        )
    
    if context_size < 0 or context_size > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上下文大小必须在0-5之间"
        )
    
    return {
        "question": question.strip(),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "context_size": context_size
    }


def check_service_health():
    """检查服务健康状态"""
    try:
        # 检查数据库连接
        from models.database import check_db_connection
        if not check_db_connection():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库连接失败"
            )
        
        # 检查向量存储连接
        vector_store = get_vector_store()
        if not vector_store.health_check():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="向量存储连接失败"
            )
        
        return True
    except Exception as e:
        logger.error(f"服务健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务不可用"
        )


def get_upload_directory() -> str:
    """获取上传目录"""
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def cleanup_temp_file(file_path: str):
    """清理临时文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败 {file_path}: {e}")


class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        self._services = {}
    
    def get_service(self, service_name: str, factory_func):
        """获取服务实例（单例模式）"""
        if service_name not in self._services:
            self._services[service_name] = factory_func()
        return self._services[service_name]
    
    def reset_services(self):
        """重置所有服务实例"""
        self._services.clear()
        global _document_processor, _vector_service, _qa_service, _embedding_service, _vector_store, _keyword_service, _rerank_service
        _document_processor = None
        _vector_service = None
        _qa_service = None
        _embedding_service = None
        _vector_store = None
        _keyword_service = None
        _rerank_service = None


# 全局服务管理器实例
service_manager = ServiceManager()


def get_service_manager() -> ServiceManager:
    """获取服务管理器"""
    return service_manager