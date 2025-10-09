"""
文本清洗与去噪流水线
提供轻量级的规范化、去噪与归一化能力。
"""

import re
from typing import Optional


class CleaningService:
    def __init__(self, normalize_whitespace: bool = True, drop_empty_lines: bool = True, strip_lines: bool = True):
        self.normalize_whitespace = normalize_whitespace
        self.drop_empty_lines = drop_empty_lines
        self.strip_lines = strip_lines

    def clean_text(self, text: Optional[str]) -> str:
        """Lesson 17：实现文本清洗与去噪流水线。

        TODO(lab04-lesson17):
        1) 统一换行与空格（\r/\r\n→\n，制表/不间断空格→普通空格，合并多余空格）
        2) 去除行首尾空白
        3) 删除空行
        返回清洗后的文本字符串。
        """
        raise NotImplementedError("TODO(lab04-lesson17): 实现 CleaningService.clean_text()")