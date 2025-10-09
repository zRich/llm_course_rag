from typing import Dict, Any, List, Optional

def fetch_api(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None, path: Optional[str] = None, field_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """从HTTP API获取数据并映射到统一字段。需安装requests。
    path: 如果响应为嵌套结构，指定列表路径，如"data.items"
    """
    field_mapping = field_mapping or {}
    id_col = field_mapping.get("id", "id")
    title_col = field_mapping.get("title", "title")
    content_col = field_mapping.get("content", "content")
    tags_col = field_mapping.get("tags", "tags")

    try:
        import requests
        resp = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data
        if path:
            for p in path.split('.'):
                items = items.get(p, [])
        rows: List[Dict[str, Any]] = []
        for r in items:
            rows.append({
                "id": r.get(id_col),
                "title": r.get(title_col) or "",
                "content": r.get(content_col) or "",
                "tags": r.get(tags_col) or [],
                "source": "api",
            })
        return rows
    except Exception:
        return []