"""基于 BGE-reranker-v2-m3 的本地重排序器。

使用 FlagEmbedding 的 FlagReranker，模型首次运行时自动从 HuggingFace 下载。
接口与 ZhipuReranker 保持一致（rerank / arerank → list[RerankResult]）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.reranker.zhipu_reranker import RerankResult

logger = logging.getLogger(__name__)

# 默认模型名称
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

# 本地缓存目录（避免重复下载）
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class LocalRreranker:
    """本地 BGE reranker，免费、不依赖 API。"""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self._model_name = model
        self._device = device
        self._reranker = None  # 延迟加载

    def _load(self):
        """延迟加载模型（首次调用时才下载/加载到内存）。"""
        if self._reranker is not None:
            return
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError("请先安装依赖: pip install FlagEmbedding sentence-transformers torch")

        logger.info("正在加载本地 Reranker 模型: %s ...", self._model_name)
        self._reranker = FlagReranker(
            self._model_name,
            use_fp16=True,
            device=self._device or "cpu",
            cache_dir=str(_CACHE_DIR),
        )
        logger.info("本地 Reranker 模型加载完成")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """同步重排序。"""
        self._load()

        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self._reranker.compute_score(pairs, normalize=True)

        # compute_score 返回 float 或 list[float]
        if isinstance(scores, float):
            scores = [scores]

        # 按分数降序排列
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[RerankResult] = []
        for rank, (idx, score) in enumerate(indexed[:top_n]):
            results.append(
                RerankResult(
                    index=idx,
                    relevance_score=float(score),
                    content=documents[idx],
                )
            )

        logger.info("本地 Rerank 完成：输入 %d 文档，返回 %d 条", len(documents), len(results))
        return results

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """异步重排序（本地模型无真正异步，直接调用同步方法）。"""
        return self.rerank(query, documents, top_n)
