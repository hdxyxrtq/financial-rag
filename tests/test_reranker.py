from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.reranker.zhipu_reranker import ZhipuReranker


class TestZhipuReranker:
    @patch("src.reranker.zhipu_reranker.requests.post")
    def test_rerank_returns_sorted_results(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95, "document": {"text": "doc_c"}},
                {"index": 0, "relevance_score": 0.85, "document": {"text": "doc_a"}},
                {"index": 1, "relevance_score": 0.70, "document": {"text": "doc_b"}},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        reranker = ZhipuReranker(api_key="test_key")
        results = reranker.rerank("query", ["doc_a", "doc_b", "doc_c"], top_n=3)

        assert len(results) == 3
        assert results[0].content == "doc_c"
        assert results[0].relevance_score == 0.95
        assert results[0].index == 2

    @patch("src.reranker.zhipu_reranker.requests.post")
    def test_rerank_top_n_limits_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "doc_a"}},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        reranker = ZhipuReranker(api_key="test_key")
        results = reranker.rerank("query", ["doc_a", "doc_b", "doc_c"], top_n=1)
        assert len(results) == 1
        payload = mock_post.call_args.kwargs["json"]
        assert payload["top_n"] == 1

    @patch("src.reranker.zhipu_reranker.requests.post")
    def test_rerank_empty_documents(self, mock_post):
        reranker = ZhipuReranker(api_key="test_key")
        results = reranker.rerank("query", [], top_n=5)
        assert results == []
        mock_post.assert_not_called()

    @patch("src.reranker.zhipu_reranker.requests.post")
    def test_rerank_failure_graceful_degradation(self, mock_post):
        mock_post.side_effect = Exception("API error")
        reranker = ZhipuReranker(api_key="test_key")
        results = reranker.rerank("query", ["doc_a", "doc_b"], top_n=2)
        assert len(results) == 2
        assert results[0].relevance_score == 1.0
