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
        d: Optional[Document] = self.db.query(Document).filter(Document.id == doc_id).first()
        if not d:
            return None
        content = self.cleaner.clean_text(d.content)
        self._delete_chunks(doc_id)
        chunks = self.splitter.split_text(content, strategy=self.strategy, metadata={"document_id": str(doc_id), "source": d.file_type})
        rows: List[Chunk] = []
        for ch in chunks:
            rows.append(Chunk(
                document_id=doc_id,
                chunk_index=ch.chunk_index,
                content=ch.content,
                content_hash=ch.content_hash,
                start_pos=ch.start_position,
                end_pos=ch.end_position,
                token_count=ch.estimated_tokens,
                char_count=ch.char_count,
                metadata_=json.dumps(ch.metadata, ensure_ascii=False),
            ))
        if rows:
            self.db.add_all(rows)
        d.total_chunks = len(rows)
        d.total_tokens = sum(r.token_count for r in rows)
        d.status = "completed"
        self.db.commit()
        return d

    def upsert_document(self, doc_id: int, title: Optional[str], content: str, metadata: Optional[str] = None) -> Optional[Document]:
        d: Optional[Document] = self.db.query(Document).filter(Document.id == doc_id).first()
        if not d:
            return None
        new_content = self.cleaner.clean_text(content)
        new_hash = sha256(new_content)
        # 无变化则跳过
        if d.content_hash and d.content_hash == new_hash:
            return d
        # 更新文档并重建
        if title:
            d.title = title
        d.content = new_content
        d.content_hash = new_hash
        if metadata:
            d.metadata_ = metadata
        d.status = "processing"
        self.db.commit()
        return self.rebuild_document(doc_id)