"""
元数据过滤 DSL
覆盖 Lesson 9：基于简单 DSL 的元数据过滤，对检索结果进行二次过滤。

DSL 格式（示例）：
[
  {"op": "eq", "field": "document_title", "value": "指南"},
  {"op": "in", "field": "metadata.category", "value": ["spec", "tutorial"]},
  {"op": "range", "field": "metadata.page", "min": 1, "max": 10},
  {"op": "exists", "field": "metadata.author"}
]
"""

from typing import Any, Dict, List


def _get_field(data: Dict, field: str) -> Any:
    # 支持 "metadata.foo.bar" 深层路径
    parts = field.split(".")
    cur: Any = data
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _op_eq(val: Any, expected: Any) -> bool:
    return val == expected


def _op_in(val: Any, options: List[Any]) -> bool:
    if options is None:
        return False
    if isinstance(options, list):
        return val in options
    return False


def _op_range(val: Any, min_v: Any, max_v: Any) -> bool:
    try:
        if min_v is not None and val < min_v:
            return False
        if max_v is not None and val > max_v:
            return False
        return True
    except Exception:
        return False


def _op_exists(val: Any) -> bool:
    return val is not None and val != ""


def apply_filters(results: List[Dict], filters: List[Dict]) -> List[Dict]:
    """
    # TODO(lab02-L9): 实现过滤 DSL（eq/in/range/exists），支持 metadata 深层寻址。
    当前占位实现：当传入非空 filters 时抛出未实现；为空时透传。
    """
    if not filters:
        return results
    raise NotImplementedError("TODO(lab02-L9): 实现 apply_filters")