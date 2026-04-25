import logging

import httpx

logger = logging.getLogger(__name__)


class SiliconFlowEmbedder:
    """SiliconFlow Embedding（OpenAI 兼容接口）。"""

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-large-zh-v1.5",
        base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 20,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = httpx.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": batch},
                timeout=60.0,
            )
            resp.raise_for_status()
            body = resp.json()
            sorted_data = sorted(body["data"], key=lambda x: x.get("index", 0))
            all_embeddings.extend([item["embedding"] for item in sorted_data])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.embed_texts([text])
        if not result:
            raise RuntimeError("SiliconFlow Embedding 返回空结果")
        return result[0]

    async def aembed_query(self, text: str) -> list[float]:
        result = await self.aembed_texts([text])
        if not result:
            raise RuntimeError("SiliconFlow Embedding 返回空结果")
        return result[0]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=60.0) as client:
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                resp = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                body = resp.json()
                sorted_data = sorted(body["data"], key=lambda x: x.get("index", 0))
                all_embeddings.extend([item["embedding"] for item in sorted_data])
            return all_embeddings
