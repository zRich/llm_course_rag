from typing import Dict, Any, List, Optional

def fetch_sql(db_uri: str, query: str, field_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """从SQL数据源查询并映射到统一字段。需安装sqlalchemy。
    field_mapping: {"id": "id_col", "title": "title_col", "content": "content_col", "tags": "tags_col"}
    """
    field_mapping = field_mapping or {}
    id_col = field_mapping.get("id", "id")
    title_col = field_mapping.get("title", "title")
    content_col = field_mapping.get("content", "content")
    tags_col = field_mapping.get("tags", "tags")

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            rs = conn.execute(text(query))
            rows = []
            for r in rs.mappings():
                rows.append({
                    "id": r.get(id_col),
                    "title": r.get(title_col) or "",
                    "content": r.get(content_col) or "",
                    "tags": (r.get(tags_col) or []),
                    "source": "sql",
                })
            return rows
    except Exception:
        return []