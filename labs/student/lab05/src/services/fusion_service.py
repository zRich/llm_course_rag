"""
融合策略服务：RRF 与线性加权
覆盖 Lesson 8，支持将向量检索与关键词检索结果进行融合。
"""

from typing import List, Dict, Tuple


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _rank_map(results: List[Dict]) -> Dict[str, int]:
    # 假设已按分数降序排列
    ranks = {}
    for idx, r in enumerate(results):
        cid = str(r.get("chunk_id"))
        ranks[cid] = idx + 1
    return ranks


def _dedup_merge(base: List[Dict], extra: List[Dict]) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for r in base + extra:
        cid = str(r.get("chunk_id"))
        if cid not in merged:
            merged[cid] = dict(r)
        else:
            # 保留已有字段，补充缺失字段
            for k, v in r.items():
                if k not in merged[cid] or merged[cid][k] in (None, ""):
                    merged[cid][k] = v
    return merged


def rrf_fuse(keyword_results: List[Dict], vector_results: List[Dict], k: int = 60) -> List[Dict]:
    """RRF 融合：score = Σ 1 / (k + rank)"""
    if not keyword_results and not vector_results:
        return []

    # 计算 rank
    kw_ranks = _rank_map(keyword_results)
    vec_ranks = _rank_map(vector_results)

    merged = _dedup_merge(keyword_results, vector_results)

    fused: List[Tuple[str, float]] = []
    for cid in merged.keys():
        score = 0.0
        if cid in kw_ranks:
            score += 1.0 / (k + kw_ranks[cid])
        if cid in vec_ranks:
            score += 1.0 / (k + vec_ranks[cid])
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    # 回填字段
    out: List[Dict] = []
    for cid, score in fused:
        r = merged[cid]
        r["score"] = float(score)
        r["fusion_strategy"] = "rrf"
        out.append(r)
    return out


def linear_fuse(keyword_results: List[Dict], vector_results: List[Dict], w_keyword: float = 0.5, w_vector: float = 0.5) -> List[Dict]:
    """线性加权融合：score = w_k * norm_k + w_v * norm_v"""
    if not keyword_results and not vector_results:
        return []

    # 取分数向量并归一化
    kw_scores = [float(r.get("score", 0.0)) for r in keyword_results]
    vec_scores = [float(r.get("score", 0.0)) for r in vector_results]
    kw_norm = _normalize(kw_scores)
    vec_norm = _normalize(vec_scores)

    kw_norm_map: Dict[str, float] = {}
    for r, s in zip(keyword_results, kw_norm):
        kw_norm_map[str(r.get("chunk_id"))] = float(s)

    vec_norm_map: Dict[str, float] = {}
    for r, s in zip(vector_results, vec_norm):
        vec_norm_map[str(r.get("chunk_id"))] = float(s)

    merged = _dedup_merge(keyword_results, vector_results)

    fused: List[Tuple[str, float]] = []
    for cid in merged.keys():
        score = w_keyword * kw_norm_map.get(cid, 0.0) + w_vector * vec_norm_map.get(cid, 0.0)
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    out: List[Dict] = []
    for cid, score in fused:
        r = merged[cid]
        r["score"] = float(score)
        r["fusion_strategy"] = "linear"
        out.append(r)
    return out