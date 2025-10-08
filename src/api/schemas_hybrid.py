"""
Lab02 混合检索相关的 Pydantic 模型
包含融合参数、过滤条件 DSL、混合检索请求。
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, validator


class FusionParams(BaseModel):
    """融合参数"""
    strategy: Literal['rrf', 'linear'] = Field('rrf', description="融合策略：rrf 或 linear")
    keyword_weight: Optional[float] = Field(0.5, description="关键词检索权重（仅线性加权使用）")
    vector_weight: Optional[float] = Field(0.5, description="向量检索权重（仅线性加权使用）")

    @validator('keyword_weight', 'vector_weight')
    def validate_weights(cls, v):
        if v is None:
            return v
        if v < 0 or v > 1:
            raise ValueError('权重必须在 [0,1] 范围内')
        return v


class FilterCondition(BaseModel):
    """过滤条件 DSL 元素"""
    op: Literal['eq', 'in', 'range', 'exists'] = Field(description="操作符")
    field: str = Field(description="字段路径，如 metadata.category 或 document_title")
    value: Optional[Any] = Field(None, description="比较值（eq/in）")
    min: Optional[float] = Field(None, description="最小值（range）")
    max: Optional[float] = Field(None, description="最大值（range）")


class HybridSearchRequest(BaseModel):
    """混合检索请求：支持关键词检索、向量检索、融合、过滤与重排"""
    query: str = Field(description="搜索查询", min_length=1, max_length=500)
    document_ids: Optional[List[int]] = Field(None, description="限制搜索的文档ID列表")
    limit: Optional[int] = Field(10, description="返回结果数量", ge=1, le=100)
    score_threshold: Optional[float] = Field(0.5, description="相似度阈值（针对向量检索）", ge=0.0, le=1.0)

    enable_keyword: bool = Field(True, description="启用关键词检索")
    enable_vector: bool = Field(True, description="启用向量检索")
    fusion: Optional[FusionParams] = Field(None, description="融合参数；当两路均启用时可用")
    filters: Optional[List[FilterCondition]] = Field(None, description="过滤条件 DSL")

    rerank: bool = Field(False, description="启用 Cross-Encoder 重排序")
    rerank_top_m: Optional[int] = Field(10, description="参与重排的候选数量", ge=1, le=50)

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('搜索查询不能为空')
        return v.strip()