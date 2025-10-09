from typing import Dict, Any, List, Optional
import csv


def load_csv(file_path: str, field_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """从CSV加载数据并映射到统一字段。
    field_mapping: {"id": "id_col", "title": "title_col", "content": "content_col", "tags": "tags_col"}
    """
    field_mapping = field_mapping or {}
    id_col = field_mapping.get("id", "id")
    title_col = field_mapping.get("title", "title")
    content_col = field_mapping.get("content", "content")
    tags_col = field_mapping.get("tags", "tags")

    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "id": r.get(id_col),
                "title": r.get(title_col) or "",
                "content": r.get(content_col) or "",
                "tags": (r.get(tags_col) or "").split(";") if r.get(tags_col) else [],
                "source": "csv",
            })
    return rows