"""
实验4 第一阶段：批量断点续传与幂等键
提供检查点管理、幂等键生成与可重入批量加载管道。
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Iterable, Protocol

from sqlalchemy.orm import Session

from config.settings import settings
from utils.cache import get_cache, make_key
from services.text_splitter import TextSplitter, SplitStrategy
from services.cleaning import CleaningService
from models.document import Document
from models.chunk import Chunk


@dataclass
class Checkpoint:
    batch_id: str
    offset: int
    status: str  # pending|processing|completed|failed
    updated_at: float


class CheckpointManager:
    """使用缓存后端（内存/Redis）管理检查点。"""

    def __init__(self):
        self.cache = get_cache()
        # 使用较长TTL，避免检查点意外过期；复用 embedding 层 TTL
        self.ttl = int(getattr(settings, "ttl_embedding", 604800))

    def _key(self, pipeline_id: str) -> str:
        return make_key({"layer": "checkpoint", "pipeline": pipeline_id}, namespace=getattr(settings, "cache_namespace", "lab04"))

    def load(self, pipeline_id: str) -> Optional[Checkpoint]:
        data = self.cache.get(self._key(pipeline_id))
        if not data:
            return None
        try:
            if isinstance(data, str):
                data = json.loads(data)
            return Checkpoint(**data)
        except Exception:
            return None

    def save(self, pipeline_id: str, ckpt: Checkpoint):
        self.cache.set(self._key(pipeline_id), asdict(ckpt), self.ttl)

    def mark(self, pipeline_id: str, status: str, offset: int):
        ckpt = self.load(pipeline_id) or Checkpoint(batch_id=pipeline_id, offset=0, status="pending", updated_at=time.time())
        ckpt.status = status
        ckpt.offset = offset
        ckpt.updated_at = time.time()
        self.save(pipeline_id, ckpt)


def idempotency_key(doc: Dict[str, Any], stable_fields: List[str]) -> str:
    """按稳定字段组合生成幂等键（SHA256）。"""
    payload = "|".join(str(doc.get(f, "")) for f in stable_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Sink(Protocol):
    async def exists(self, key: str) -> bool: ...
    async def write(self, key: str, doc: Dict[str, Any]) -> None: ...


class DocumentSink:
    """将结构化文档写入 Document/Chunk 表。"""

    def __init__(self, db: Session, split_strategy: SplitStrategy = SplitStrategy.FIXED_SIZE, cleaner: Optional[CleaningService] = None):
        self.db = db
        self.splitter = TextSplitter()
        self.strategy = split_strategy
        self.cleaner = cleaner

    async def exists(self, key: str) -> bool:
        # 通过 metadata_ 中的 idempotency_key 或 content_hash 判断重复
        q = self.db.query(Document).filter(
            (Document.content_hash == key) | (Document.metadata_.ilike(f"%\"idempotency_key\": \"{key}\"%"))
        )
        return self.db.query(q.exists()).scalar() or False

    async def write(self, key: str, doc: Dict[str, Any]) -> None:
        title = (doc.get("title") or "").strip() or doc.get("id") or "untitled"
        content = (doc.get("content") or "").strip()
        if self.cleaner:
            content = self.cleaner.clean_text(content)
        source = doc.get("source") or "structured"
        # 补齐必需字段：以结构化来源构造占位信息
        filename = f"ingested:{source}:{doc.get('id', key)}"
        file_path = f"ingested://{source}/{doc.get('id', key)}"
        file_type = "ingested"
        file_size = len(content.encode("utf-8"))

        # 创建 Document
        d = Document(
            filename=filename,
            original_filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            title=title,
            content=content,
            content_hash=key,
            status="processing",
            metadata_=json.dumps({"source": source, "id": doc.get("id"), "tags": doc.get("tags", []), "idempotency_key": key}, ensure_ascii=False),
        )
        self.db.add(d)
        self.db.commit()
        self.db.refresh(d)

        # 分块并写入 Chunk
        chunks = self.splitter.split_text(content, strategy=self.strategy, metadata={"document_id": str(d.id), "source": source})
        chunk_rows: List[Chunk] = []
        for ch in chunks:
            row = Chunk(
                document_id=d.id,
                chunk_index=ch.chunk_index,
                content=ch.content,
                content_hash=ch.content_hash,
                start_pos=ch.start_position,
                end_pos=ch.end_position,
                token_count=ch.estimated_tokens,
                char_count=ch.char_count,
                metadata_=json.dumps(ch.metadata, ensure_ascii=False),
            )
            chunk_rows.append(row)

        if chunk_rows:
            self.db.add_all(chunk_rows)
        d.total_chunks = len(chunk_rows)
        d.total_tokens = sum(c.token_count for c in chunk_rows)
        d.status = "completed"
        self.db.commit()


class BatchResumableLoader:
    """支持断点续传的批量加载器，具备幂等保障。"""

    def __init__(self, ckpt: CheckpointManager, sink: Sink, stable_fields: List[str]):
        self.ckpt = ckpt
        self.sink = sink
        self.stable_fields = stable_fields

    async def run(self, pipeline_id: str, docs: List[Dict[str, Any]], chunk_size: int = 1000):
        start_offset = (self.ckpt.load(pipeline_id) or Checkpoint(pipeline_id, 0, "pending", time.time())).offset
        n = len(docs)
        for i in range(start_offset, n, chunk_size):
            batch = docs[i : min(i + chunk_size, n)]
            self.ckpt.mark(pipeline_id, "processing", i)
            for d in batch:
                key = idempotency_key(d, self.stable_fields)
                if await self.sink.exists(key):
                    continue
                await self.sink.write(key, d)
            self.ckpt.mark(pipeline_id, "completed", i + len(batch))


def make_structured_docs(raws: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """容错的结构化文档规范化（Lesson 17 对齐的最小实现）。"""
    out: List[Dict[str, Any]] = []
    for r in raws:
        doc = {
            "id": r.get("id") or r.get("doc_id") or r.get("uuid"),
            "title": (r.get("title") or "").strip(),
            "content": (r.get("content") or "").strip(),
            "tags": r.get("tags") or [],
            "source": r.get("source") or "unknown",
        }
        out.append(doc)
    return out