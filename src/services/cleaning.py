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
        if not text:
            return ""
        out = text
        if self.normalize_whitespace:
            # 统一换行与空格
            out = out.replace("\r\n", "\n").replace("\r", "\n")
            out = re.sub(r"[\t\u00A0]+", " ", out)
            out = re.sub(r"[ ]{2,}", " ", out)
        if self.strip_lines:
            out = "\n".join(line.strip() for line in out.splitlines())
        if self.drop_empty_lines:
            out = "\n".join(line for line in out.splitlines() if line)
        return out