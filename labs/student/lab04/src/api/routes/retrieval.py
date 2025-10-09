"""
混合检索路由：集成关键词检索、向量检索、融合、过滤与重排
"""

import time
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import SearchResponse, SearchResult
from api.schemas_hybrid import HybridSearchRequest
from api.dependencies import (
    get_vector_service,
    get_keyword_service,
    get_rerank_service,
)
from services.vector_service import VectorService
from services.keyword_service import KeywordSearchService
from services.rerank_service import RerankService
from services.fusion_service import rrf_fuse, linear_fuse
from services.filter_dsl import apply_filters


router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


def _to_search_result(r: Dict) -> SearchResult:
    return SearchResult(
        chunk_id=str(r.get("chunk_id")),
        document_id=str(r.get("document_id")),
        document_filename=str(r.get("document_filename")),
        document_title=r.get("document_title"),
        chunk_index=int(r.get("chunk_index", 0)),
        content=str(r.get("content", "")),
        score=float(r.get("score", 0.0)),
        start_position=int(r.get("start_pos", r.get("start_position", 0))),
        end_position=int(r.get("end_pos", r.get("end_position", 0))),
        metadata=r.get("metadata"),
    )


@router.post("/search", response_model=SearchResponse)
async def hybrid_search(
    req: HybridSearchRequest,
    vector_service: VectorService = Depends(get_vector_service),
    keyword_service: KeywordSearchService = Depends(get_keyword_service),
    rerank_service: RerankService = Depends(get_rerank_service),
):
    start = time.time()

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")

    # 1) 向量检索
    vec_results: List[Dict] = vector_service.search_similar_chunks(
        query=req.query,
        limit=req.top_k,
        score_threshold=req.score_threshold,
        document_ids=req.document_ids,
    ) or []

    # 2) 关键词检索
    kw_results: List[Dict] = []
    try:
        kw_results = keyword_service.search(
            query=req.query,
            limit=req.top_k,
            document_ids=req.document_ids,
        ) or []
    except Exception:
        # 关键词索引可能尚未构建，忽略异常以保持鲁棒性
        kw_results = []

    # 确保两侧结果按分数降序（融合函数期望如此）
    vec_results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    kw_results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # 3) 融合
    if req.fusion.strategy == "rrf":
        fused = rrf_fuse(kw_results, vec_results, k=req.fusion.k)
    else:
        fused = linear_fuse(kw_results, vec_results, req.fusion.w_keyword, req.fusion.w_vector)

    # 4) 过滤（可选）
    fused_filtered = apply_filters(fused, [f.dict() for f in (req.filters or [])])

    # 5) 重排（可选）
    final_candidates = fused_filtered
    if req.rerank_top_m and req.rerank_top_m > 0:
        try:
            final_candidates = rerank_service.rerank(req.query, fused_filtered, top_m=req.rerank_top_m)
        except Exception:
            # 如果模型不可用或首次加载失败，保持原排序
            final_candidates = fused_filtered

    # 截断到 top_k
    final_candidates = final_candidates[:req.top_k]

    # 构造响应
    results = [_to_search_result(r) for r in final_candidates]
    return SearchResponse(
        success=True,
        message="检索成功",
        query=req.query,
        results=results,
        total_found=len(fused),
        processing_time=time.time() - start,
    )