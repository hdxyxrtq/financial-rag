from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder


def _mock_response(embeddings: list[list[float]]) -> dict:
    """构造 mock SiliconFlow Embedding API 响应。"""
    return {
        "data": [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)],
    }


class TestSiliconFlowEmbedder:

    def test_embed_texts_success(self) -> None:
        """正常批量嵌入应返回正确向量。"""
        expected = [[0.1, 0.2], [0.3, 0.4]]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_response(expected)

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp):
            embedder = SiliconFlowEmbedder(api_key="test-key")
            result = embedder.embed_texts(["hello", "world"])

        assert result == expected

    def test_embed_texts_empty_list(self) -> None:
        """空输入应返回空列表，不调用 API。"""
        embedder = SiliconFlowEmbedder(api_key="test-key")
        result = embedder.embed_texts([])

        assert result == []

    def test_embed_texts_batch_splitting(self) -> None:
        """超出 batch_size 的输入应分批调用 API。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_response([[1.0] for _ in range(10)])

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp) as mock_post:
            embedder = SiliconFlowEmbedder(api_key="test-key", batch_size=10)
            texts = [f"text-{i}" for i in range(25)]
            embedder.embed_texts(texts)

        assert mock_post.call_count == 3

    def test_embed_texts_response_order(self) -> None:
        """API 乱序返回时应按 index 排序。"""
        mock_data = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_data

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp):
            embedder = SiliconFlowEmbedder(api_key="test-key")
            result = embedder.embed_texts(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query(self) -> None:
        """embed_query 应调用 embed_texts 并返回第一个结果。"""
        expected = [0.5, 0.6]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_response([expected])

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp):
            embedder = SiliconFlowEmbedder(api_key="test-key")
            result = embedder.embed_query("query text")

        assert result == expected

    def test_embed_query_empty_result_raises(self) -> None:
        """embed_query 空结果应抛出 RuntimeError。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_response([])

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp):
            embedder = SiliconFlowEmbedder(api_key="test-key")
            with pytest.raises(RuntimeError, match="返回空结果"):
                embedder.embed_query("query text")

    def test_embed_texts_custom_model(self) -> None:
        """自定义模型名称应正确传递给 API。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _mock_response([[1.0]])

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp) as mock_post:
            embedder = SiliconFlowEmbedder(api_key="test-key", model="custom-model")
            embedder.embed_texts(["test"])

        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "custom-model"

    def test_embed_texts_api_error_raises(self) -> None:
        """API 调用失败应抛出异常。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("API error")

        with patch("src.embeddings.siliconflow_embedder.httpx.post", return_value=mock_resp):
            embedder = SiliconFlowEmbedder(api_key="test-key")
            with pytest.raises(Exception, match="API error"):
                embedder.embed_texts(["test"])
