"""
混合检索路由（Lab02 Lesson 7-10）
提供关键词检索 + 向量检索 + 融合 + 过滤 + 重排 的端点。
"""

import time
from typing import List, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import SearchResponse, SearchResult
from api.schemas_hybrid import HybridSearchRequest
from api.dependencies import get_db, validate_search_params, get_vector_service

from services.keyword_service import KeywordSearchService
from services.fusion_service import rrf_fuse, linear_fuse
from services.filter_dsl import apply_filters
from services.rerank_service import RerankService


router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


# 路由级服务实例（避免修改全局依赖，保持对现有功能的零影响）
_keyword_service: KeywordSearchService | None = None
_rerank_service: RerankService | None = None


def _get_keyword_service(db: Session) -> KeywordSearchService:
    global _keyword_service
    if _keyword_service is None:
        _keyword_service = KeywordSearchService()
        try:
            _keyword_service.initialize(db)
        except Exception:
            pass
    return _keyword_service


def _get_rerank_service() -> RerankService:
    global _rerank_service
    if _rerank_service is None:
        _rerank_service = RerankService()
    return _rerank_service


@router.post("/search", response_model=SearchResponse)
def hybrid_search(
    req: HybridSearchRequest,
    db: Session = Depends(get_db),
    vector_service = Depends(get_vector_service)
):
    """混合检索入口"""
    start = time.time()

    # 验证基础搜索参数（沿用现有校验逻辑）
    params = validate_search_params(req.query, req.limit or 10, req.score_threshold or 0.5)
    query = params["query"]
    limit = params["limit"]
    score_th = params["score_threshold"]

    keyword_results: List[Dict] = []
    vector_results: List[Dict] = []

    # 关键词检索
    if req.enable_keyword:
        kw = _get_keyword_service(db)
        keyword_results = kw.search(query=query, limit=limit, document_ids=req.document_ids, db=db)

    # 向量检索
    if req.enable_vector:
        vector_results = vector_service.search_similar_chunks(
            query=query,
            limit=limit,
            score_threshold=score_th,
            document_ids=req.document_ids,
            db=db,
        )

    # 融合
    combined: List[Dict] = []
    if req.enable_keyword and req.enable_vector:
        if req.fusion and req.fusion.strategy == 'linear':
            kw_w = float(req.fusion.keyword_weight or 0.5)
            vec_w = float(req.fusion.vector_weight or 0.5)
            combined = linear_fuse(keyword_results, vector_results, w_keyword=kw_w, w_vector=vec_w)
        else:
            combined = rrf_fuse(keyword_results, vector_results)
    else:
        combined = vector_results if req.enable_vector else keyword_results

    # 过滤
    if req.filters:
        combined = apply_filters(combined, req.filters)

    # 重排序
    if req.rerank:
        reranker = _get_rerank_service()
        combined = reranker.rerank(query, combined, top_m=req.rerank_top_m or 10)

    # 限制返回数量
    combined = combined[:limit]

    # 构建响应结果
    results: List[SearchResult] = []
    for r in combined:
        results.append(
            SearchResult(
                chunk_id=str(r.get("chunk_id")),
                document_id=str(r.get("document_id")),
                document_filename=str(r.get("document_filename") or ""),
                document_title=r.get("document_title"),
                chunk_index=int(r.get("chunk_index") or 0),
                content=str(r.get("content") or ""),
                score=float(r.get("score") or 0.0),
                start_position=int(r.get("start_pos") or 0),
                end_position=int(r.get("end_pos") or 0),
                metadata=r.get("metadata") or {},
            )
        )

    return SearchResponse(
        success=True,
        message="混合检索完成",
        timestamp=time.time(),
        query=query,
        results=results,
        total_found=len(results),
        processing_time=time.time() - start,
    )