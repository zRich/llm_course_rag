"""
混合检索相关的请求模型
避免改动现有 schemas.py，独立新增 Lesson 7-10 需要的结构。
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FusionParams(BaseModel):
    strategy: Literal["rrf", "linear"] = Field(default="rrf", description="融合策略")
    k: int = Field(default=60, description="RRF 常数 k")
    w_keyword: float = Field(default=0.5, description="线性融合关键词权重")
    w_vector: float = Field(default=0.5, description="线性融合向量权重")


class FilterCondition(BaseModel):
    op: Literal["eq", "in", "range", "exists"]
    field: str
    value: Optional[object] = None
    min: Optional[float] = None
    max: Optional[float] = None


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: float = 0.2
    document_ids: Optional[List[int]] = None
    fusion: FusionParams = FusionParams()
    filters: Optional[List[FilterCondition]] = None
    rerank_top_m: int = 10