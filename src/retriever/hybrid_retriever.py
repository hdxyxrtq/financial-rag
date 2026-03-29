from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.retriever.retriever import RetrievalResult

if TYPE_CHECKING:
    from src.config import HybridConfig
    from src.retriever.bm25_retriever import BM25Retriever
    from src.retriever.retriever import Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(
        self,
        retriever: Retriever,
        bm25_retriever: BM25Retriever,
        config: HybridConfig,
        score_threshold: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._bm25 = bm25_retriever
        self._config = config
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        strategy = self._config.strategy
        k = top_k if top_k is not None else 5

        if strategy == "vector":
            results = self._retriever.retrieve(query, top_k=k, where=where)
            if self._score_threshold > 0:
                results = [r for r in results if r.score >= self._score_threshold]
            return results

        if strategy == "bm25":
            return self._bm25_to_results(query, top_k=k)

        return self._hybrid_search(query, top_k=k, where=where)

    def _hybrid_search(
        self, query: str, top_k: int, where: dict | None = None,
    ) -> list[RetrievalResult]:
        cfg = self._config
        fetch_k = cfg.vector_fetch_k

        # 并行执行向量检索和 BM25 检索
        vector_results: list[RetrievalResult] | None = None
        bm25_results_raw: list[tuple[str, float]] | None = None
        vector_error: Exception | None = None
        bm25_error: Exception | None = None

        def _do_vector():
            nonlocal vector_results, vector_error
            try:
                vector_results = self._retriever.retrieve(query, top_k=fetch_k, where=where)
            except Exception as e:
                vector_error = e

        def _do_bm25():
            nonlocal bm25_results_raw, bm25_error
            try:
                bm25_results_raw = self._bm25.retrieve(query, top_k=cfg.bm25_fetch_k)
            except Exception as e:
                bm25_error = e

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_vec = executor.submit(_do_vector)
                f_bm25 = executor.submit(_do_bm25)
                # 等待两个任务都完成（无论成功或失败）
                f_vec.result()
                f_bm25.result()
        except Exception as e:
            logger.warning("并行检索框架异常: %s", e)

        # 降级逻辑：一方失败时使用另一方
        if vector_error is not None:
            logger.warning("向量检索失败，降级为纯 BM25: %s", vector_error)
            if bm25_results_raw is not None:
                ids = [doc_id for doc_id, _ in bm25_results_raw[:top_k]]
                return self._fetch_results_by_ids(ids, bm25_results_raw)
            return self._bm25_to_results(query, top_k=top_k)

        if bm25_error is not None:
            logger.warning("BM25 检索失败，降级为纯向量: %s", bm25_error)
            if vector_results is None:
                return []
            return vector_results[:top_k]

        if vector_results is None:
            return []
        if bm25_results_raw is None:
            return []

        fused = _rrf_fuse(
            [(r.doc_id, r.score) for r in vector_results],
            bm25_results_raw,
            k=cfg.rrf_k,
            vector_weight=cfg.vector_weight,
            bm25_weight=cfg.bm25_weight,
        )

        fused_ids = [doc_id for doc_id, _ in fused[:top_k]]
        return self._fetch_results_by_ids(fused_ids, fused)

    def _bm25_to_results(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            bm25_results = self._bm25.retrieve(query, top_k=top_k)
        except Exception as e:
            logger.error("BM25 检索失败: %s", e)
            return []

        ids = [doc_id for doc_id, _ in bm25_results]
        return self._fetch_results_by_ids(ids, bm25_results)

    def _fetch_results_by_ids(
        self, ids: list[str], scored_ids: list[tuple[str, float]],
    ) -> list[RetrievalResult]:
        if not ids:
            return []

        score_map = {doc_id: score for doc_id, score in scored_ids}

        try:
            doc_map = self._bm25.get_documents_by_ids(ids)
        except Exception as e:
            logger.error("从 ChromaDB 批量获取文档失败: %s", e)
            return []

        results: list[RetrievalResult] = []
        for doc_id in ids:
            doc_info = doc_map.get(doc_id)
            if not doc_info:
                continue
            content = doc_info["content"]
            if not content:
                continue
            results.append(RetrievalResult(
                content=content,
                score=score_map.get(doc_id, 0.0),
                metadata=doc_info.get("metadata") or {},
                doc_id=doc_id,
            ))

        if self._score_threshold > 0:
            results = [r for r in results if r.score >= self._score_threshold]

        logger.info("混合检索完成：返回 %d 条结果", len(results))
        return results


def _rrf_fuse(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int,
    vector_weight: float,
    bm25_weight: float,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank)
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank)
    if not scores:
        return []
    # 归一化到 0-1 范围，兼容 score_threshold
    max_score = max(scores.values())
    min_score = min(scores.values())
    score_range = max_score - min_score
    if score_range > 0:
        scores = {did: (s - min_score) / score_range for did, s in scores.items()}
    else:
        scores = {did: 1.0 for did in scores}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
