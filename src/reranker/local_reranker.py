"""基于 BGE-reranker-v2-m3 的本地重排序器。

使用 sentence_transformers.CrossEncoder 加载模型，首次运行自动下载。
接口与 ZhipuReranker 一致（rerank / arerank → list[RerankResult]）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.reranker.zhipu_reranker import RerankResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class LocalRreranker:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self._model_name = model
        self._device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import os

            # 全部离线，阻止任何网络请求
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("请先安装依赖: pip install sentence-transformers torch")

        # 优先使用本地快照路径，彻底绕过 HuggingFace Hub 网络请求
        model_source = self._resolve_local_path() or self._model_name

        logger.info("正在加载本地 Reranker 模型: %s ...", model_source)
        self._model = CrossEncoder(
            model_source,
            device=self._device or "cpu",
        )
        logger.info("本地 Reranker 模型加载完成")

    def _resolve_local_path(self) -> str | None:
        """从 HuggingFace Hub 缓存目录中查找已下载的模型快照路径。"""
        # HuggingFace Hub 缓存结构: <cache>/models--<org>--<model>/snapshots/<hash>/
        parts = self._model_name.split("/")
        if len(parts) != 2:
            return None
        org, name = parts
        cache_model_dir = _CACHE_DIR / f"models--{org}--{name}"
        snapshots_dir = cache_model_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None
        # 取最新的快照目录（按修改时间排序）
        snapshot_dirs = sorted(
            snapshots_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshot_dirs:
            if snapshot.is_dir() and (snapshot / "config.json").is_file():
                return str(snapshot)
        return None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        self._load()

        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)

        # 归一化到 0-1
        import numpy as np

        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            scores = (scores - min_s) / (max_s - min_s)
        else:
            scores = np.ones_like(scores)

        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[RerankResult] = []
        for idx, score in indexed[:top_n]:
            results.append(RerankResult(index=idx, relevance_score=float(score), content=documents[idx]))

        logger.info("本地 Rerank 完成：输入 %d 文档，返回 %d 条", len(documents), len(results))
        return results

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        return self.rerank(query, documents, top_n)
