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
    """
    RRF 融合：score = Σ 1 / (k + rank)
    # TODO(lab02-L8): 在此实现 RRF 融合逻辑（按名次映射并汇总）
    """
    raise NotImplementedError("TODO(lab02-L8): 实现 rrf_fuse")


def linear_fuse(keyword_results: List[Dict], vector_results: List[Dict], w_keyword: float = 0.5, w_vector: float = 0.5) -> List[Dict]:
    """
    线性加权融合：score = w_k * norm_k + w_v * norm_v
    # TODO(lab02-L8): 在此实现线性加权融合（归一化并加权合成）
    """
    raise NotImplementedError("TODO(lab02-L8): 实现 linear_fuse")