"""
增量更新与失效重建服务
根据内容哈希判断变化，执行有序失效与重建。
"""

import hashlib
import json
from typing import Optional, List
from sqlalchemy.orm import Session

from services.text_splitter import TextSplitter, SplitStrategy
from services.cleaning import CleaningService
from models.document import Document
from models.chunk import Chunk


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IncrementalService:
    def __init__(self, db: Session, split_strategy: SplitStrategy = SplitStrategy.FIXED_SIZE):
        self.db = db
        self.splitter = TextSplitter()
        self.strategy = split_strategy
        self.cleaner = CleaningService()

    def _delete_chunks(self, doc_id: int):
        self.db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

    def rebuild_document(self, doc_id: int) -> Optional[Document]:
        """Lesson 18：失效重建文档分块。

        TODO(lab04-lesson18):
        1) 读取文档并清洗内容
        2) 删除旧分块，按策略重新分块
        3) 批量写入新分块，更新统计与状态
        返回更新后的文档。
        """
        raise NotImplementedError("TODO(lab04-lesson18): 实现 IncrementalService.rebuild_document()")

    def upsert_document(self, doc_id: int, title: Optional[str], content: str, metadata: Optional[str] = None) -> Optional[Document]:
        """Lesson 18：增量更新（内容哈希驱动）。

        TODO(lab04-lesson18):
        1) 读取文档，清洗新内容并计算哈希
        2) 若哈希未变化直接返回；否则更新字段与状态
        3) 调用 `rebuild_document` 完成失效重建
        返回更新后的文档。
        """
        raise NotImplementedError("TODO(lab04-lesson18): 实现 IncrementalService.upsert_document()")