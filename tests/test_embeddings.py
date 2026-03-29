import time
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.zhipu_embedder import EmbeddingError, ZhipuEmbedder


class TestZhipuEmbedder:
    """ZhipuEmbedder 单元测试。"""

    def setup_method(self) -> None:
        self.mock_client = MagicMock()
        self.patcher = patch(
            "src.embeddings.zhipu_embedder.ZhipuAI",
            return_value=self.mock_client,
        )
        self.patcher.start()
        self.embedder = ZhipuEmbedder(api_key="test-key")

    def teardown_method(self) -> None:
        self.patcher.stop()

    def _mock_response(self, embeddings: list[list[float]]) -> MagicMock:
        """构造 mock API 响应。"""
        data = [
            MagicMock(index=i, embedding=emb)
            for i, emb in enumerate(embeddings)
        ]
        return MagicMock(data=data)

    def test_embed_texts_success(self) -> None:
        """正常批量嵌入应返回正确向量。"""
        expected = [[0.1, 0.2], [0.3, 0.4]]
        self.mock_client.embeddings.create.return_value = self._mock_response(expected)

        result = self.embedder.embed_texts(["hello", "world"])

        assert result == expected
        self.mock_client.embeddings.create.assert_called_once_with(
            model="embedding-3", input=["hello", "world"],
        )

    def test_embed_texts_empty_list(self) -> None:
        """空输入应返回空列表，不调用 API。"""
        result = self.embedder.embed_texts([])

        assert result == []
        self.mock_client.embeddings.create.assert_not_called()

    def test_embed_texts_batch_splitting(self) -> None:
        """超出 batch_size 的输入应分批调用 API。"""
        # batch_size=10, 25 条文本 → 3 次调用 (10+10+5)
        embedder = ZhipuEmbedder(api_key="test-key", batch_size=10)
        self.mock_client.embeddings.create.return_value = self._mock_response(
            [[1.0] for _ in range(10)]
        )

        texts = [f"text-{i}" for i in range(25)]
        embedder.embed_texts(texts)

        assert self.mock_client.embeddings.create.call_count == 3

    def test_embed_texts_response_order(self) -> None:
        """API 乱序返回时应按 index 排序。"""
        # 返回顺序：index=1 在前, index=0 在后
        mock_data = [
            MagicMock(index=1, embedding=[0.3, 0.4]),
            MagicMock(index=0, embedding=[0.1, 0.2]),
        ]
        self.mock_client.embeddings.create.return_value = MagicMock(data=mock_data)

        result = self.embedder.embed_texts(["a", "b"])

        # 应按 index 排序：[0.1,0.2] 在前
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query(self) -> None:
        """embed_query 应调用 embed_texts 并返回第一个结果。"""
        expected = [0.5, 0.6]
        self.mock_client.embeddings.create.return_value = self._mock_response([expected])

        result = self.embedder.embed_query("query text")

        assert result == expected

    def test_embed_texts_retry_on_error(self) -> None:
        """API 失败后应指数退避重试，最终成功。"""
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("API unreachable")
            return self._mock_response([[1.0, 2.0]])

        self.mock_client.embeddings.create.side_effect = side_effect

        # mock sleep 避免实际等待
        with patch("src.utils.time.sleep"):
            result = self.embedder.embed_texts(["test"])

        assert result == [[1.0, 2.0]]
        assert call_count == 3

    def test_embed_texts_max_retries_exceeded(self) -> None:
        """超过最大重试次数应抛出 EmbeddingError。"""
        self.mock_client.embeddings.create.side_effect = ConnectionError("always fail")

        with patch("src.utils.time.sleep"):
            with pytest.raises(EmbeddingError, match="调用失败"):
                self.embedder.embed_texts(["test"])

    def test_embed_texts_custom_model(self) -> None:
        """自定义模型名称应正确传递给 API。"""
        embedder = ZhipuEmbedder(api_key="test-key", model="embedding-2")
        self.mock_client.embeddings.create.return_value = self._mock_response([[1.0]])

        embedder.embed_texts(["test"])

        self.mock_client.embeddings.create.assert_called_once_with(
            model="embedding-2", input=["test"],
        )
