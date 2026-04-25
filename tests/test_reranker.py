from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.reranker.local_reranker import LocalRreranker


class TestLocalRreranker:
    @patch.object(LocalRreranker, "_load")
    def test_rerank_returns_sorted_results(self, mock_load) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7, 0.95, 0.5]

        reranker = LocalRreranker()
        reranker._model = mock_model

        results = reranker.rerank("query", ["doc_a", "doc_b", "doc_c"], top_n=3)

        assert len(results) == 3
        assert results[0].content == "doc_b"
        assert results[0].relevance_score > results[1].relevance_score

    @patch.object(LocalRreranker, "_load")
    def test_rerank_top_n_limits_results(self, mock_load) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.3]

        reranker = LocalRreranker()
        reranker._model = mock_model

        results = reranker.rerank("query", ["doc_a", "doc_b", "doc_c"], top_n=1)
        assert len(results) == 1

    @patch.object(LocalRreranker, "_load")
    def test_rerank_empty_documents(self, mock_load) -> None:
        reranker = LocalRreranker()
        results = reranker.rerank("query", [], top_n=5)
        assert results == []

    @patch.object(LocalRreranker, "_load")
    def test_rerank_normalization(self, mock_load) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.5, 0.9]

        reranker = LocalRreranker()
        reranker._model = mock_model

        results = reranker.rerank("query", ["a", "b", "c"], top_n=3)

        scores = [r.relevance_score for r in results]
        assert scores[0] == pytest.approx(1.0)
        assert scores[-1] == pytest.approx(0.0)
