from typing import Dict, Any, List, Optional
import csv


def load_csv(file_path: str, field_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """从 CSV 加载数据并映射到统一字段结构。

    课程 16 实验任务：实现 CSV 连接器数据加载与字段映射。
    - 输入：`file_path` 指向 CSV 文件路径；`field_mapping` 提供列名映射
      示例：{"id": "id_col", "title": "title_col", "content": "content_col", "tags": "tags_col"}
    - 输出：统一结构字典列表：[{id,title,content,tags,source}]

    TODO(lab04-lesson16):
    1) 使用 `csv.DictReader` 读取文件
    2) 根据 `field_mapping` 提取并归一化字段
    3) 将分号分隔的标签列转为列表
    4) 设置 `source` 为 "csv"
    """
    # 提示：实现完成后，删除下面的异常并返回处理结果
    raise NotImplementedError("TODO(lab04-lesson16): 实现 CSV 连接器 load_csv()")