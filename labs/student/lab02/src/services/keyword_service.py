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
        """
        执行关键词检索，返回结构与向量检索一致的结果字典列表。
        # TODO(lab02-L7): 在此实现基于字符 n-gram 的 TF-IDF 检索与结果组装。
        """
        raise NotImplementedError("TODO(lab02-L7): 实现关键词检索")