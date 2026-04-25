from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.correction.types import RetrievalQuality

if TYPE_CHECKING:
    from src.retriever.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalGate:
    """Layer 0: Assess retrieval quality based on rerank/retrieval scores."""

    def __init__(
        self,
        rerank_threshold_good: float = 0.7,
        rerank_threshold_weak: float = 0.3,
    ) -> None:
        self._threshold_good = rerank_threshold_good
        self._threshold_weak = rerank_threshold_weak

    def assess(
        self,
        results: list[RetrievalResult],
        reranked_results: list[RetrievalResult] | None = None,
    ) -> RetrievalQuality:
        """Assess retrieval quality based on rerank/retrieval scores.

        Uses the top-1 rerank score when available, otherwise falls back to
        the raw retrieval score.  Returns a quality level used to gate
        downstream generation:
            GOOD     – top-1 ≥ threshold_good  (default 0.7)
            MARGINAL – between threshold_weak and threshold_good
            WEAK     – top-1 < threshold_weak  (default 0.3) or no results
        """
        num_sources = len(results)

        if num_sources == 0:
            return RetrievalQuality(
                level="WEAK",
                top_score=0.0,
                avg_score=0.0,
                num_sources=0,
            )

        if reranked_results:
            scores = [r.score for r in reranked_results]
        else:
            scores = [r.score for r in results]

        top_score = scores[0] if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if top_score >= self._threshold_good:
            level = "GOOD"
        elif top_score < self._threshold_weak:
            level = "WEAK"
        else:
            level = "MARGINAL"

        logger.info(
            "Retrieval quality: level=%s, top=%.3f, avg=%.3f, sources=%d",
            level, top_score, avg_score, num_sources,
        )

        return RetrievalQuality(
            level=level,
            top_score=top_score,
            avg_score=avg_score,
            num_sources=num_sources,
        )
