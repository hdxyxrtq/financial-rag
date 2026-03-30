import logging

from zhipuai import ZhipuAI

from src.utils import _AUTH_KEYWORDS, _QUOTA_KEYWORDS, _TIMEOUT_KEYWORDS, async_call_with_retry, call_with_retry

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class EmbeddingAuthError(EmbeddingError):
    pass


class EmbeddingQuotaError(EmbeddingError):
    pass


class EmbeddingTimeoutError(EmbeddingError):
    pass


def _classify_embedding_error(error: Exception) -> EmbeddingError:
    msg = str(error).lower()
    if any(kw in msg for kw in _AUTH_KEYWORDS):
        return EmbeddingAuthError("Embedding API Key 无效，请检查您的 API Key")
    if any(kw in msg for kw in _QUOTA_KEYWORDS):
        return EmbeddingQuotaError("Embedding API 额度已用完")
    if any(kw in msg for kw in _TIMEOUT_KEYWORDS):
        return EmbeddingTimeoutError("Embedding API 请求超时，请检查网络连接后重试")
    return EmbeddingError(f"Embedding API 调用失败: {error}")


class ZhipuEmbedder:
    """智谱 Embedding-3 向量化封装。

    支持批量文本向量化，自动分批处理与指数退避重试。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        batch_size: int = 20,
    ) -> None:
        self._client = ZhipuAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(
                "Embedding batch %d/%d (%d texts)",
                i // self._batch_size + 1,
                (len(texts) + self._batch_size - 1) // self._batch_size,
                len(batch),
            )

            def do_call(b=batch):
                response = self._client.embeddings.create(
                    model=self._model,
                    input=b,
                )
                sorted_data = sorted(response.data, key=lambda x: x.index or 0)
                return [item.embedding for item in sorted_data]

            embeddings = call_with_retry(
                do_call,
                _classify_embedding_error,
                non_retriable_types=(EmbeddingAuthError, EmbeddingQuotaError),
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_texts([text])
        if not result:
            raise EmbeddingError("API returned no embeddings for the query")
        return result[0]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.debug(
                "Async embedding batch %d/%d (%d texts)",
                i // self._batch_size + 1,
                (len(texts) + self._batch_size - 1) // self._batch_size,
                len(batch),
            )

            def do_call(b=batch):
                response = self._client.embeddings.create(
                    model=self._model,
                    input=b,
                )
                sorted_data = sorted(response.data, key=lambda x: x.index or 0)
                return [item.embedding for item in sorted_data]

            embeddings = await async_call_with_retry(
                do_call,
                _classify_embedding_error,
                non_retriable_types=(EmbeddingAuthError, EmbeddingQuotaError),
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def aembed_query(self, text: str) -> list[float]:
        result = await self.aembed_batch([text])
        if not result:
            raise EmbeddingError("API returned no embeddings for the query")
        return result[0]
