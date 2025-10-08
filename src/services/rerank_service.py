"""
重排序服务（Cross-Encoder）
覆盖 Lesson 10：对融合后的候选进行语义重排序，提高最终命中质量。
"""

import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _ensure_model(self):
        if self._model is None:
            logger.info(f"加载 CrossEncoder 模型: {self.model_name}")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: List[Dict], top_m: int = 10) -> List[Dict]:
        if not candidates or not query:
            return candidates
        self._ensure_model()

        # 取前 top_m 进行重排，避免过长队列影响性能
        pre = candidates[:max(1, top_m)]
        pairs = [(query, str(r.get("content") or "")) for r in pre]
        scores = list(self._model.predict(pairs))

        # 将 scores 写回并排序
        for r, s in zip(pre, scores):
            r["rerank_score"] = float(s)

        pre.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        # 拼接尾部未参与重排的部分，保持稳定性
        tail = candidates[len(pre):]
        return pre + tail