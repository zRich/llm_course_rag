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
        # TODO(lab02-L10): 在此惰性加载 CrossEncoder 模型；失败时合理回退
        raise NotImplementedError("TODO(lab02-L10): 加载 CrossEncoder 模型")

    def rerank(self, query: str, candidates: List[Dict], top_m: int = 10) -> List[Dict]:
        """
        # TODO(lab02-L10): 实现基于 CrossEncoder 的 Top‑M 重排并写回 rerank_score。
        当 top_m>0 时抛未实现；为 0 时直接透传以保持最小可运行。
        """
        if not candidates or not query:
            return candidates
        if top_m and top_m > 0:
            raise NotImplementedError("TODO(lab02-L10): 实现 rerank")
        return candidates