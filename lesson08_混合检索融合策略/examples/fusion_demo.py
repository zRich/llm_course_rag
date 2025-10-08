#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson 08 示例：关键词检索与向量检索融合演示
支持两种方法：RRF（reciprocal rank fusion）与线性加权（weighted）

运行示例：
  python examples/fusion_demo.py --method rrf --top_k 10
  python examples/fusion_demo.py --method weighted --top_k 10 --w_k 0.6 --w_v 0.4
"""

import argparse
from typing import List, Dict, Tuple


def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def rrf_fuse(keyword: List[Dict], vector: List[Dict], k: int = 60, top_k: int = 10) -> Tuple[List[Dict], List[str]]:
    warnings = []
    # 构建 rank 映射
    kw_rank = {item["id"]: item.get("rank", i + 1) for i, item in enumerate(keyword)}
    vt_rank = {item["id"]: item.get("rank", i + 1) for i, item in enumerate(vector)}

    all_ids = set(kw_rank.keys()) | set(vt_rank.keys())
    fused = []
    for _id in all_ids:
        score = 0.0
        if _id in kw_rank:
            score += 1.0 / (k + kw_rank[_id])
        if _id in vt_rank:
            score += 1.0 / (k + vt_rank[_id])
        # 取一个代表项作为输出基础
        src = next((x for x in keyword if x["id"] == _id), None) or next((x for x in vector if x["id"] == _id), None)
        fused.append({
            "id": _id,
            "title": src.get("title", ""),
            "source": src.get("source", ""),
            "fused_score": score,
            "method": "rrf",
        })

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:top_k], warnings


def weighted_fuse(keyword: List[Dict], vector: List[Dict], w_k: float = 0.5, w_v: float = 0.5, top_k: int = 10) -> Tuple[List[Dict], List[str]]:
    warnings = []
    # 归一化两路分数
    kw_scores = {x["id"]: x.get("score", 0.0) for x in keyword}
    vt_scores = {x["id"]: x.get("score", 0.0) for x in vector}
    kw_norm = dict(zip(kw_scores.keys(), min_max_normalize(list(kw_scores.values()))))
    vt_norm = dict(zip(vt_scores.keys(), min_max_normalize(list(vt_scores.values()))))

    all_ids = set(kw_scores.keys()) | set(vt_scores.keys())
    fused = []
    for _id in all_ids:
        s_k = kw_norm.get(_id, None)
        s_v = vt_norm.get(_id, None)
        if s_k is None and s_v is None:
            warnings.append(f"id={_id} has neither keyword nor vector score, skipped")
            continue
        score = (w_k * s_k if s_k is not None else 0.0) + (w_v * s_v if s_v is not None else 0.0)
        src = next((x for x in keyword if x["id"] == _id), None) or next((x for x in vector if x["id"] == _id), None)
        fused.append({
            "id": _id,
            "title": src.get("title", ""),
            "source": src.get("source", ""),
            "fused_score": score,
            "method": "weighted",
        })

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused[:top_k], warnings


def demo_data() -> Tuple[List[Dict], List[Dict]]:
    # 模拟两路检索结果（真实项目中应从 DB/向量库获取）
    keyword = [
        {"id": "A", "title": "Postgres 全文检索概述", "score": 12.3, "source": "docs/pgfts.md", "rank": 1},
        {"id": "B", "title": "BM25 与 TF-IDF", "score": 9.4, "source": "docs/bm25.md", "rank": 2},
        {"id": "C", "title": "中文分词与词典", "score": 7.7, "source": "docs/jieba.md", "rank": 3},
    ]
    vector = [
        {"id": "C", "title": "中文分词与词典", "score": 0.71, "source": "docs/jieba.md", "rank": 1},
        {"id": "A", "title": "Postgres 全文检索概述", "score": 0.66, "source": "docs/pgfts.md", "rank": 2},
        {"id": "D", "title": "语义向量与相似度", "score": 0.62, "source": "docs/embeddings.md", "rank": 3},
    ]
    return keyword, vector


def main():
    parser = argparse.ArgumentParser(description="Fusion demo for keyword + vector retrieval")
    parser.add_argument("--method", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--k", type=int, default=60, help="RRF k parameter")
    parser.add_argument("--w_k", type=float, default=0.5, help="weight for keyword score")
    parser.add_argument("--w_v", type=float, default=0.5, help="weight for vector score")
    args = parser.parse_args()

    keyword, vector = demo_data()

    if args.method == "rrf":
        fused, warnings = rrf_fuse(keyword, vector, k=args.k, top_k=args.top_k)
    else:
        fused, warnings = weighted_fuse(keyword, vector, w_k=args.w_k, w_v=args.w_v, top_k=args.top_k)

    print(f"Method: {args.method}\nTop-{args.top_k} results:")
    for i, item in enumerate(fused, 1):
        print(f"{i:2d}. id={item['id']:<2} score={item['fused_score']:.4f} title={item['title']} source={item['source']}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("- ", w)


if __name__ == "__main__":
    main()