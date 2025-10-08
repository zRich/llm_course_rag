#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lesson 09 示例：元数据过滤

演示过滤算子（eq/in/contains/range/exists）与布尔组合（and/or），
覆盖字符串、列表、数值与日期类型，并统一错误返回结构。
"""

from __future__ import annotations

import json
import datetime as dt
from typing import Any, Dict, List, Union


def parse_date(value: Union[str, dt.datetime]) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            # 支持日期与日期时间（ISO 8601）
            if len(value) == 10:
                return dt.datetime.strptime(value, "%Y-%m-%d")
            return dt.datetime.fromisoformat(value)
        except Exception as exc:
            raise ValueError(f"Invalid date: {value}") from exc
    raise TypeError(f"Unsupported date type: {type(value)}")


def normalize(value: Any) -> Any:
    # 简化示例：直接返回；可扩展为大小写归一化、去空白等
    return value


def op_eq(doc: Dict[str, Any], field: str, value: Any) -> bool:
    return normalize(doc.get(field)) == normalize(value)


def op_in(doc: Dict[str, Any], field: str, values: List[Any]) -> bool:
    # 约定：空列表视为不命中
    if not isinstance(values, list) or not values:
        return False
    return normalize(doc.get(field)) in [normalize(v) for v in values]


def op_contains(doc: Dict[str, Any], field: str, value: Any) -> bool:
    target = doc.get(field)
    if target is None:
        return False
    # 支持字符串包含与列表包含
    if isinstance(target, str) and isinstance(value, str):
        return normalize(value) in normalize(target)
    if isinstance(target, list):
        return normalize(value) in [normalize(v) for v in target]
    return False


def op_range(doc: Dict[str, Any], field: str, gte: Any = None, lte: Any = None) -> bool:
    val = doc.get(field)
    if val is None:
        return False
    # 日期范围
    if isinstance(val, str) and (isinstance(gte, str) or isinstance(lte, str)):
        try:
            val_dt = parse_date(val)
            if gte is not None:
                gte_dt = parse_date(gte)
                if val_dt < gte_dt:
                    return False
            if lte is not None:
                lte_dt = parse_date(lte)
                if val_dt > lte_dt:
                    return False
            return True
        except Exception:
            return False
    # 数值范围
    try:
        num = float(val)
        if gte is not None and num < float(gte):
            return False
        if lte is not None and num > float(lte):
            return False
        return True
    except Exception:
        return False


def op_exists(doc: Dict[str, Any], field: str) -> bool:
    return field in doc and doc[field] is not None


def eval_condition(doc: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    if "eq" in cond:
        c = cond["eq"]
        return op_eq(doc, c["field"], c["value"])
    if "in" in cond:
        c = cond["in"]
        return op_in(doc, c["field"], c.get("values", []))
    if "contains" in cond:
        c = cond["contains"]
        return op_contains(doc, c["field"], c["value"])
    if "range" in cond:
        c = cond["range"]
        return op_range(doc, c["field"], c.get("gte"), c.get("lte"))
    if "exists" in cond:
        c = cond["exists"]
        return op_exists(doc, c["field"]) 
    raise KeyError("Unknown operator in condition")


def filter_docs(docs: List[Dict[str, Any]], dsl: Dict[str, Any]) -> Dict[str, Any]:
    try:
        def eval_logic(doc: Dict[str, Any], node: Dict[str, Any]) -> bool:
            if "and" in node:
                return all(eval_condition(doc, c) for c in node["and"])
            if "or" in node:
                return any(eval_condition(doc, c) for c in node["or"])
            # 默认：将节点视为单条件
            return eval_condition(doc, node)

        hits = []
        for d in docs:
            if eval_logic(d, dsl):
                hits.append(d)
        return {"hits": hits, "count": len(hits)}
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "error": {
                "code": 400,
                "message": "Invalid filter DSL",
                "details": {"exception": type(exc).__name__, "str": str(exc)},
                "hints": [
                    "检查算子是否为 eq/in/contains/range/exists",
                    "确保 range 的 gte/lte 类型与字段一致（日期或数值）",
                    "in.values 必须为非空列表",
                ],
            }
        }


def demo() -> None:
    docs = [
        {
            "id": 1,
            "title": "RAG 简介",
            "source": "docs",
            "doc_type": "guide",
            "tags": ["rag", "入门", "元数据"],
            "language": "zh",
            "published": True,
            "created_at": "2024-03-12",
            "updated_at": "2024-07-01",
            "views": 420,
        },
        {
            "id": 2,
            "title": "FAQ 索引维护",
            "source": "kb",
            "doc_type": "faq",
            "tags": ["索引", "维护"],
            "language": "zh",
            "published": False,
            "created_at": "2023-11-20",
            "updated_at": None,
            "views": 120,
        },
        {
            "id": 3,
            "title": "Metadata Filtering Guide",
            "source": "docs",
            "doc_type": "guide",
            "tags": ["metadata", "filter"],
            "language": "en",
            "published": True,
            "created_at": "2024-08-10",
            "updated_at": "2024-09-15",
            "views": 980,
        },
        {
            "id": 4,
            "title": "运营手册",
            "source": "internal",
            "doc_type": "policy",
            "tags": ["运营", "内部"],
            "language": "zh",
            "published": True,
            "created_at": "2022-05-01",
            "updated_at": "2024-01-02",
            "views": 50,
        },
    ]

    dsl = {
        "and": [
            {"eq": {"field": "language", "value": "zh"}},
            {"in": {"field": "doc_type", "values": ["guide", "faq"]}},
            {"contains": {"field": "tags", "value": "元数据"}},
            {"range": {"field": "created_at", "gte": "2024-01-01", "lte": "2024-12-31"}},
            {"eq": {"field": "published", "value": True}},
            {"range": {"field": "views", "gte": 100}},
        ]
    }

    result = filter_docs(docs, dsl)
    print("== Hits (AND combo) ==")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 错误示例：未知算子
+    bad_dsl = {"and": [{"unknown": {"field": "language", "value": "zh"}}]}
    print("\n== Error (unknown operator) ==")
    print(json.dumps(filter_docs(docs, bad_dsl), ensure_ascii=False, indent=2))

    # 对比 contains：字符串目标 vs 列表目标
    contains_string = {"contains": {"field": "title", "value": "RAG"}}
    contains_list = {"contains": {"field": "tags", "value": "维护"}}
    print("\n== contains on string ==")
    print(json.dumps(filter_docs(docs, contains_string), ensure_ascii=False, indent=2))
    print("\n== contains on list ==")
    print(json.dumps(filter_docs(docs, contains_list), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()