#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lesson 10 示例：重排序（Rerank）接入

特性：
- 可选启用 Cross-Encoder（需安装 sentence-transformers），否则回退到启发式（词重合）。
- 归一化（min-max）与线性融合（alpha）；稳定排序并列打断。
- 仅对 Top-K 候选进行重排；计算 MRR/nDCG 指标；统一错误结构。
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


def error_response(code: int, message: str, details: Dict[str, Any], hints: Sequence[str]) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details, "hints": list(hints)}}


class Reranker:
    def __init__(self, use_ce: bool = False, model_name: Optional[str] = None):
        self.method = "heuristic"
        self.ce = None
        if use_ce:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self.method = "cross_encoder"
                self.ce = CrossEncoder(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as exc:
                self.method = "heuristic"
                self.ce = None
                self._init_exc = exc
        else:
            self._init_exc = None

    def score(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        if self.method == "cross_encoder" and self.ce is not None:
            pairs = [(query, d.get("text", "")) for d in docs]
            try:
                scores = self.ce.predict(pairs).tolist()
                return [float(s) for s in scores]
            except Exception as exc:
                # 回退到启发式
                self.method = "heuristic"
                self._init_exc = exc
        # 启发式：词重合计数
        q_tokens = [t for t in query.lower().split() if t]
        scores = []
        for d in docs:
            text = str(d.get("text", "")).lower()
            doc_tokens = [t for t in text.split() if t]
            overlap = sum(1 for t in q_tokens if t in doc_tokens)
            scores.append(float(overlap))
        return scores


def min_max_normalize(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def stable_sort(items: List[Dict[str, Any]], key_fields: Tuple[str, ...], reverse: bool = True) -> List[Dict[str, Any]]:
    # 并列打断：依次按 key_fields 排序；最后按 id 保证稳定
    def sort_key(d: Dict[str, Any]):
        keys = [d.get(k) for k in key_fields]
        keys.append(d.get("id"))
        return tuple(keys)
    return sorted(items, key=sort_key, reverse=reverse)


def mrr_at_k(relevant_ids: Sequence[int], ranked_ids: Sequence[int], k: int = 10) -> float:
    rel = set(relevant_ids)
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevant_ids: Sequence[int], ranked_ids: Sequence[int], k: int = 10) -> float:
    import math
    rel = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        gain = 1.0 if doc_id in rel else 0.0
        dcg += (gain) / math.log2(i + 1)
    # IDCG（单一相关）
    idcg = 1.0 / math.log2(1 + 1)
    return dcg / idcg if idcg > 0 else 0.0


def demo(args: argparse.Namespace) -> None:
    # 示例候选（含基线分数）
    docs = [
        {"id": 1, "title": "RAG 简介", "text": "RAG overview and pipeline", "base_score": 10.2},
        {"id": 2, "title": "融合与过滤", "text": "hybrid retrieval and metadata filtering", "base_score": 9.1},
        {"id": 3, "title": "Rerank 接入最佳实践", "text": "Rerank integration with linear fusion and cross encoder", "base_score": 7.5},
        {"id": 4, "title": "Embedding 选择", "text": "bi encoder selection and vector search", "base_score": 9.3},
        {"id": 5, "title": "答案抽取", "text": "answer extraction after rerank stage", "base_score": 6.8},
    ]
    query = "Rerank integration with linear fusion"
    relevant_ids = [3]

    # 参数校验
    if not (0.0 <= args.alpha <= 1.0):
        print(json.dumps(error_response(400, "Invalid alpha", {"alpha": args.alpha}, ["alpha 应在 [0,1] 区间", "建议 0.3~0.7"]), ensure_ascii=False, indent=2))
        return
    if args.top_k < 1:
        print(json.dumps(error_response(400, "Invalid top_k", {"top_k": args.top_k}, ["top_k 应为正整数", "建议 10~100"]), ensure_ascii=False, indent=2))
        return

    # 基线排序
    baseline = stable_sort(docs, ("base_score",))
    baseline_ids = [d["id"] for d in baseline]

    # Rerank 评分（仅前 top_k）
    start = time.time()
    rr = Reranker(use_ce=args.use_ce, model_name=args.model_name)
    top_slice = baseline[: min(args.top_k, len(baseline))]
    rr_scores = rr.score(query, top_slice)
    rr_scores_norm = min_max_normalize(rr_scores)

    # 融合分数与稳定排序
    merged = []
    for i, d in enumerate(baseline):
        rr_norm = rr_scores_norm[i] if i < len(rr_scores_norm) else 0.0
        final = args.alpha * float(d["base_score"]) + (1.0 - args.alpha) * rr_norm
        merged.append({**d, "rerank_score": rr_norm, "final_score": final})

    reranked = stable_sort(merged, ("final_score", "base_score"))
    reranked_ids = [d["id"] for d in reranked]
    elapsed_ms = (time.time() - start) * 1000.0

    # 指标
    mrr_base = mrr_at_k(relevant_ids, baseline_ids, k=10)
    mrr_rerank = mrr_at_k(relevant_ids, reranked_ids, k=10)
    ndcg_base = ndcg_at_k(relevant_ids, baseline_ids, k=10)
    ndcg_rerank = ndcg_at_k(relevant_ids, reranked_ids, k=10)

    output = {
        "query": query,
        "params": {"alpha": args.alpha, "top_k": args.top_k, "use_ce": args.use_ce},
        "latency_ms": round(elapsed_ms, 2),
        "baseline_top10": [{"id": d["id"], "title": d["title"], "score": d["base_score"]} for d in baseline[:10]],
        "reranked_top10": [{"id": d["id"], "title": d["title"], "score": d["final_score"], "rerank": d["rerank_score"]} for d in reranked[:10]],
        "metrics": {
            "mrr_base": mrr_base,
            "mrr_rerank": mrr_rerank,
            "ndcg_base": ndcg_base,
            "ndcg_rerank": ndcg_rerank,
            "mrr_gain_pct": ((mrr_rerank - mrr_base) / (mrr_base + 1e-9)) * 100.0 if mrr_base > 0 else (100.0 if mrr_rerank > 0 else 0.0),
        },
        "notes": [],
    }

    # 依赖失败提示（若存在）
    if getattr(rr, "_init_exc", None) is not None:
        output["notes"].append("Cross-Encoder 初始化失败，已回退启发式。")

    print(json.dumps(output, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lesson 10 Rerank demo")
    p.add_argument("--alpha", type=float, default=0.5, help="融合权重（0~1）")
    p.add_argument("--top_k", type=int, default=100, help="重排的候选范围")
    p.add_argument("--use_ce", action="store_true", help="启用 Cross-Encoder（需 sentence-transformers）")
    p.add_argument("--model_name", type=str, default=None, help="Cross-Encoder 模型名")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args)