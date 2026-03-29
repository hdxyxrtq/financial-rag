from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import HybridConfig
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever, _rrf_fuse
from src.retriever.retriever import RetrievalResult, Retriever


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock()
    vs.get_all_documents.return_value = {
        "doc_1": "沪深300指数由沪深市场中规模大的A股组成",
        "doc_2": "市盈率是股票价格除以每股收益的比率",
        "doc_3": "资产负债率是企业总负债与总资产的比率",
        "doc_4": "ROE净资产收益率反映股东回报效率",
    }
    return vs


@pytest.fixture
def mock_retriever():
    r = MagicMock(spec=Retriever)
    r.retrieve.return_value = [
        RetrievalResult(content="doc_a", score=0.9, metadata={"source": "s1"}, doc_id="doc_1"),
        RetrievalResult(content="doc_b", score=0.8, metadata={"source": "s2"}, doc_id="doc_2"),
        RetrievalResult(content="doc_c", score=0.7, metadata={"source": "s3"}, doc_id="doc_3"),
    ]
    return r


@pytest.fixture
def hybrid_config():
    return HybridConfig(
        enabled=True,
        strategy="hybrid",
        rrf_k=60,
        vector_weight=0.6,
        bm25_weight=0.4,
        bm25_fetch_k=30,
        vector_fetch_k=30,
    )


class TestBM25Retriever:
    def test_retrieve_returns_results(self, mock_vectorstore):
        retriever = BM25Retriever(mock_vectorstore)
        results = retriever.retrieve("沪深300指数", top_k=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    def test_dirty_flag_rebuilds_on_second_call(self, mock_vectorstore):
        retriever = BM25Retriever(mock_vectorstore)
        retriever.retrieve("test")
        assert mock_vectorstore.get_all_documents.call_count == 1

        retriever.mark_dirty()
        retriever.retrieve("test2")
        assert mock_vectorstore.get_all_documents.call_count == 2

    def test_empty_vectorstore(self):
        vs = MagicMock()
        vs.get_all_documents.return_value = {}
        retriever = BM25Retriever(vs)
        results = retriever.retrieve("test")
        assert results == []


class TestRRFFusion:
    def test_fusion_combines_two_rankings(self):
        vector = [("doc_1", 0.9), ("doc_2", 0.8), ("doc_3", 0.7)]
        bm25 = [("doc_3", 5.0), ("doc_1", 4.0), ("doc_2", 2.0)]
        fused = _rrf_fuse(vector, bm25, k=60, vector_weight=0.6, bm25_weight=0.4)
        assert len(fused) == 3
        assert fused[0][0] == "doc_1"

    def test_fusion_gives_boost_to_shared_docs(self):
        vector = [("doc_1", 0.9), ("doc_2", 0.8), ("doc_5", 0.7)]
        bm25 = [("doc_1", 5.0), ("doc_3", 4.0), ("doc_4", 2.0)]
        fused = _rrf_fuse(vector, bm25, k=60, vector_weight=0.6, bm25_weight=0.4)
        fused_ids = [doc_id for doc_id, _ in fused]
        doc_1_idx = fused_ids.index("doc_1")
        doc_5_idx = fused_ids.index("doc_5")
        assert doc_1_idx < doc_5_idx


class TestHybridRetriever:
    def test_strategy_vector_uses_base_retriever(self, mock_retriever, hybrid_config):
        bm25 = MagicMock(spec=BM25Retriever)
        config = HybridConfig(strategy="vector", enabled=True)
        retriever = HybridRetriever(mock_retriever, bm25, config, score_threshold=0.0)
        results = retriever.retrieve("test", top_k=5)
        mock_retriever.retrieve.assert_called_once()
        assert len(results) == 3

    def test_hybrid_strategy_calls_both(self, mock_retriever, mock_vectorstore, hybrid_config):
        bm25 = BM25Retriever(mock_vectorstore)
        retriever = HybridRetriever(mock_retriever, bm25, hybrid_config, score_threshold=0.0)
        results = retriever.retrieve("test", top_k=5)
        mock_retriever.retrieve.assert_called_once()
        assert isinstance(results, list)

    def test_bm25_strategy(self, mock_retriever, mock_vectorstore, hybrid_config):
        bm25 = BM25Retriever(mock_vectorstore)
        config = HybridConfig(strategy="bm25", enabled=True)
        retriever = HybridRetriever(mock_retriever, bm25, config, score_threshold=0.0)
        results = retriever.retrieve("test", top_k=5)
        mock_retriever.retrieve.assert_not_called()
        assert isinstance(results, list)
