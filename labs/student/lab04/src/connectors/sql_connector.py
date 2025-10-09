from typing import Dict, Any, List, Optional

def fetch_sql(db_uri: str, query: str, field_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """从 SQL 数据源查询并映射到统一字段结构。

    课程 16 实验任务：实现 SQL 连接器查询与字段映射。
    - 输入：数据库 URI、SQL 文本、可选字段映射
    - 依赖：`sqlalchemy`
    - 输出：统一结构字典列表：[{id,title,content,tags,source}]

    TODO(lab04-lesson16):
    1) 使用 `create_engine(db_uri)` 创建引擎并执行 `text(query)`
    2) 使用 `rs.mappings()` 迭代结果并按 `field_mapping` 提取字段
    3) `tags` 字段保持为列表或空列表，`source` 设置为 "sql"
    4) 合理处理异常，避免抛出到上层
    """
    raise NotImplementedError("TODO(lab04-lesson16): 实现 SQL 连接器 fetch_sql()")