"""
关键词检索服务（TF-IDF/分词近似）
实现基于字符 n-gram 的 TF-IDF 关键词检索，作为 Lesson 7 的基线。
不依赖外部搜索引擎，使用 scikit-learn 构建索引。
"""

import logging
from typing import List, Dict, Optional, Tuple

from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.database import get_db
from src.models.chunk import Chunk
from src.models.document import Document

logger = logging.getLogger(__name__)


class KeywordSearchService:
    """基于 TF-IDF 的关键词检索服务"""

    def __init__(self):
        # 使用字符 n-gram 以兼容中文；后续可替换为分词器（jieba/hanlp）
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.chunk_texts: List[str] = []
        self.chunk_ids: List[int] = []
        self.document_ids: List[int] = []
        self._ready: bool = False

    def initialize(self, db: Session = None) -> bool:
        """从数据库构建 TF-IDF 索引"""
        if db is None:
            db = next(get_db())

        try:
            chunks: List[Chunk] = db.query(Chunk).order_by(Chunk.id.asc()).all()
            if not chunks:
                logger.warning("没有可用于关键词索引的分块")
                self._ready = False
                return False

            self.chunk_texts = [c.content or "" for c in chunks]
            self.chunk_ids = [int(c.id) for c in chunks]
            self.document_ids = [int(c.document_id) for c in chunks]

            # 字符级 n-gram 以提升中文召回；可在实验中对比不同参数
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                min_df=1,
                max_df=0.95,
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
            self._ready = True
            logger.info(f"关键词索引构建完成：共 {len(self.chunk_ids)} 个分块")
            return True
        except Exception as e:
            logger.error(f"构建关键词索引失败: {e}")
            self._ready = False
            return False

    def refresh(self, db: Session = None) -> bool:
        """重建索引"""
        self._ready = False
        return self.initialize(db)

    def search(self, query: str,
               limit: int = 10,
               document_ids: Optional[List[int]] = None,
               db: Session = None) -> List[Dict]:
        """执行关键词检索，返回结构与向量检索一致的结果字典列表"""
        if not query or not query.strip():
            return []
        if db is None:
            db = next(get_db())

        # 初始化索引
        if not self._ready:
            ok = self.initialize(db)
            if not ok:
                return []

        try:
            # 计算查询向量
            query_vec = self.vectorizer.transform([query.strip()])
            # 计算余弦相似度
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            # 可选的文档过滤（先在候选阶段进行）
            mask = [True] * len(self.chunk_ids)
            if document_ids:
                allowed = set(document_ids)
                for i, doc_id in enumerate(self.document_ids):
                    if doc_id not in allowed:
                        mask[i] = False

            # 排序并选 Top-K
            indexed_scores: List[Tuple[int, float]] = [
                (i, float(sims[i])) for i in range(len(sims)) if mask[i]
            ]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top = indexed_scores[:limit]

            # 丰富结果：查询数据库拿 chunk 与 document 信息
            results: List[Dict] = []
            chunk_ids_top = [self.chunk_ids[i] for i, _ in top]
            if not chunk_ids_top:
                return []

            chunks_map: Dict[int, Chunk] = {
                int(c.id): c for c in db.query(Chunk).filter(Chunk.id.in_(chunk_ids_top)).all()
            }
            # 一次性取文档信息
            doc_ids = list({int(chunks_map[cid].document_id) for cid in chunk_ids_top if cid in chunks_map})
            documents_map: Dict[int, Document] = {
                int(d.id): d for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()
            }

            for (i, score) in top:
                cid = int(self.chunk_ids[i])
                chunk = chunks_map.get(cid)
                if not chunk:
                    continue
                doc = documents_map.get(int(chunk.document_id))
                payload = {
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "score": score,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "token_count": chunk.token_count,
                    "char_count": chunk.char_count,
                    "document_filename": getattr(doc, "filename", "") if doc else "",
                    "document_title": getattr(doc, "title", "") if doc else "",
                    "metadata": chunk.metadata_ or {} if isinstance(chunk.metadata_, dict) else {},
                }
                results.append(payload)

            logger.info(f"关键词检索完成: 查询='{query[:50]}...', 找到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []