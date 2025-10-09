from typing import Dict, Any, List, Optional

def fetch_api(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
    field_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """从 HTTP API 获取数据并映射到统一字段结构。

    课程 16 实验任务：实现 API 连接器请求、嵌套路径解析与字段映射。
    - 输入：`url`、可选 `headers`/`params`、嵌套路径 `path`（如 "data.items"）、字段映射
    - 依赖：`requests`
    - 输出：统一结构字典列表：[{id,title,content,tags,source}]

    TODO(lab04-lesson16):
    1) 发起 GET 请求并校验响应状态
    2) 解析 JSON，并根据 `path` 逐级取出列表节点
    3) 按 `field_mapping` 提取字段，`source` 设为 "api"
    4) 合理处理异常，避免抛出到上层
    """
    raise NotImplementedError("TODO(lab04-lesson16): 实现 API 连接器 fetch_api()")