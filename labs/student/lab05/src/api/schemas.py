"""
API数据模式定义
使用Pydantic定义请求和响应模型
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(description="请求是否成功")
    message: str = Field(description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error_code: Optional[str] = Field(None, description="错误代码")
    error_details: Optional[Dict] = Field(None, description="错误详情")


# 文档相关模型
class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    title: Optional[str] = Field(None, description="文档标题")
    description: Optional[str] = Field(None, description="文档描述")
    metadata: Optional[Dict] = Field(None, description="文档元数据")


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str = Field(description="文档ID")
    filename: str = Field(description="文件名")
    title: Optional[str] = Field(description="文档标题")
    description: Optional[str] = Field(description="文档描述")
    file_size: int = Field(description="文件大小（字节）")
    file_type: str = Field(description="文件类型")
    content_hash: str = Field(description="内容哈希")
    char_count: int = Field(description="字符数")
    word_count: int = Field(description="词数")
    estimated_tokens: int = Field(description="预估token数")
    chunk_count: int = Field(description="分块数量")
    is_processed: bool = Field(description="是否已处理")
    is_vectorized: bool = Field(description="是否已向量化")
    processed_at: Optional[datetime] = Field(description="处理时间")
    vectorized_at: Optional[datetime] = Field(description="向量化时间")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    metadata: Optional[str] = Field(description="元数据")


class DocumentUploadResponse(BaseResponse):
    """文档上传响应"""
    document: Optional[DocumentInfo] = Field(description="文档信息")


class DocumentListResponse(BaseResponse):
    """文档列表响应"""
    documents: List[DocumentInfo] = Field(description="文档列表")
    total: int = Field(description="总数量")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")


class DocumentProcessRequest(BaseModel):
    """文档处理请求"""
    chunk_size: Optional[int] = Field(1000, description="分块大小")
    chunk_overlap: Optional[int] = Field(200, description="分块重叠")
    force_reprocess: Optional[bool] = Field(False, description="是否强制重新处理")


class DocumentProcessResponse(BaseResponse):
    """文档处理响应"""
    document_id: str = Field(description="文档ID")
    chunks_created: int = Field(description="创建的分块数量")
    processing_time: float = Field(description="处理时间（秒）")


# 分块相关模型
class ChunkInfo(BaseModel):
    """分块信息"""
    id: str = Field(description="分块ID")
    document_id: str = Field(description="文档ID")
    chunk_index: int = Field(description="分块索引")
    content: str = Field(description="分块内容")
    content_hash: str = Field(description="内容哈希")
    start_position: int = Field(description="开始位置")
    end_position: int = Field(description="结束位置")
    token_count: int = Field(description="token数量")
    char_count: int = Field(description="字符数量")
    vector_id: Optional[str] = Field(description="向量ID")
    vector_model: Optional[str] = Field(description="向量模型")
    vector_dimensions: Optional[int] = Field(description="向量维度")
    is_vectorized: bool = Field(description="是否已向量化")
    vectorized_at: Optional[datetime] = Field(description="向量化时间")
    created_at: datetime = Field(description="创建时间")
    metadata: Optional[Dict] = Field(description="元数据")


class ChunkListResponse(BaseResponse):
    """分块列表响应"""
    chunks: List[ChunkInfo] = Field(description="分块列表")
    total: int = Field(description="总数量")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")


# 向量化相关模型
class VectorizeRequest(BaseModel):
    """向量化请求"""
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表，为空则处理所有未向量化文档")
    force_revectorize: Optional[bool] = Field(False, description="是否强制重新向量化")


class VectorizeResponse(BaseResponse):
    """向量化响应"""
    processed_documents: int = Field(description="处理的文档数量")
    processed_chunks: int = Field(description="处理的分块数量")
    processing_time: float = Field(description="处理时间（秒）")
    failed_documents: List[str] = Field(default_factory=list, description="失败的文档ID列表")


# 问答相关模型
class QuestionRequest(BaseModel):
    """问答请求"""
    question: str = Field(description="用户问题", min_length=1, max_length=1000)
    document_ids: Optional[List[int]] = Field(None, description="限制搜索的文档ID列表")
    top_k: Optional[int] = Field(10, description="检索结果数量", ge=1, le=50)
    score_threshold: Optional[float] = Field(0.7, description="相似度阈值", ge=0.0, le=1.0)
    context_size: Optional[int] = Field(2, description="上下文大小", ge=0, le=5)
    conversation_history: Optional[List[Dict]] = Field(None, description="对话历史")

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('问题不能为空')
        return v.strip()


class SourceInfo(BaseModel):
    """来源信息"""
    document_id: str = Field(description="文档ID")
    document_filename: str = Field(description="文档文件名")
    document_title: Optional[str] = Field(description="文档标题")
    chunk_id: str = Field(description="分块ID")
    chunk_index: int = Field(description="分块索引")
    score: float = Field(description="相似度分数")
    content_preview: str = Field(description="内容预览")


class QuestionResponse(BaseResponse):
    """问答响应"""
    question: str = Field(description="用户问题")
    answer: str = Field(description="回答")
    sources: List[SourceInfo] = Field(description="来源信息列表")
    context_used: int = Field(description="使用的上下文片段数量")
    processing_time: float = Field(description="处理时间（秒）")
    model_used: str = Field(description="使用的模型")
    generation_info: Optional[Dict] = Field(description="生成信息")


class BatchQuestionRequest(BaseModel):
    """批量问答请求"""
    questions: List[str] = Field(description="问题列表", min_items=1, max_items=10)
    document_ids: Optional[List[int]] = Field(None, description="限制搜索的文档ID列表")
    top_k: Optional[int] = Field(10, description="检索结果数量", ge=1, le=50)
    score_threshold: Optional[float] = Field(0.7, description="相似度阈值", ge=0.0, le=1.0)

    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('问题列表不能为空')
        for i, question in enumerate(v):
            if not question.strip():
                raise ValueError(f'第{i+1}个问题不能为空')
        return [q.strip() for q in v]


class BatchQuestionResponse(BaseResponse):
    """批量问答响应"""
    results: List[QuestionResponse] = Field(description="问答结果列表")
    total_processing_time: float = Field(description="总处理时间（秒）")


# 搜索相关模型
class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(description="搜索查询", min_length=1, max_length=500)
    document_ids: Optional[List[int]] = Field(None, description="限制搜索的文档ID列表")
    limit: Optional[int] = Field(10, description="返回结果数量", ge=1, le=100)
    score_threshold: Optional[float] = Field(0.5, description="相似度阈值", ge=0.0, le=1.0)
    # 元数据过滤条件（Lab05 增强）：与混合检索保持一致的DSL
    class SearchFilterCondition(BaseModel):
        op: str = Field(description="操作符：eq|in|range|exists")
        field: str = Field(description="字段名，支持 metadata.foo.bar 深层路径")
        value: Optional[Any] = Field(None, description="用于 eq/in 的值")
        min: Optional[float] = Field(None, description="用于 range 的最小值")
        max: Optional[float] = Field(None, description="用于 range 的最大值")

    filters: Optional[List[SearchFilterCondition]] = Field(
        default=None,
        description="检索后过滤的DSL条件列表"
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('搜索查询不能为空')
        return v.strip()


class SearchResult(BaseModel):
    """搜索结果"""
    chunk_id: str = Field(description="分块ID")
    document_id: str = Field(description="文档ID")
    document_filename: str = Field(description="文档文件名")
    document_title: Optional[str] = Field(description="文档标题")
    chunk_index: int = Field(description="分块索引")
    content: str = Field(description="分块内容")
    score: float = Field(description="相似度分数")
    start_position: int = Field(description="开始位置")
    end_position: int = Field(description="结束位置")
    metadata: Optional[Dict] = Field(description="元数据")


class SearchResponse(BaseResponse):
    """搜索响应"""
    query: str = Field(description="搜索查询")
    results: List[SearchResult] = Field(description="搜索结果列表")
    total_found: int = Field(description="找到的结果总数")
    processing_time: float = Field(description="处理时间（秒）")


# 统计相关模型
class SystemStats(BaseModel):
    """系统统计信息"""
    database_stats: Dict = Field(description="数据库统计")
    vector_store_stats: Dict = Field(description="向量存储统计")
    service_stats: Dict = Field(description="服务统计")
    system_health: Dict = Field(description="系统健康状态")


class SystemStatsResponse(BaseResponse):
    """系统统计响应"""
    stats: SystemStats = Field(description="统计信息")


# 健康检查模型
class HealthCheckResponse(BaseResponse):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    version: str = Field(description="版本号")
    uptime: float = Field(description="运行时间（秒）")
    components: Dict[str, str] = Field(description="组件状态")


# 分页参数模型
class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(1, description="页码", ge=1)
    page_size: int = Field(20, description="每页大小", ge=1, le=100)


# 排序参数模型
class SortParams(BaseModel):
    """排序参数"""
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", description="排序顺序", pattern="^(asc|desc)$")


# 过滤参数模型
class DocumentFilterParams(BaseModel):
    """文档过滤参数"""
    filename: Optional[str] = Field(None, description="文件名过滤")
    file_type: Optional[str] = Field(None, description="文件类型过滤")
    is_processed: Optional[bool] = Field(None, description="是否已处理过滤")
    is_vectorized: Optional[bool] = Field(None, description="是否已向量化过滤")
    created_after: Optional[datetime] = Field(None, description="创建时间起始过滤")
    created_before: Optional[datetime] = Field(None, description="创建时间结束过滤")