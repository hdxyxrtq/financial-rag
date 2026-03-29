from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from src.utils import _AUTH_KEYWORDS, _QUOTA_KEYWORDS, call_with_retry

logger = logging.getLogger(__name__)


class RerankError(Exception):
    pass


class RerankAuthError(RerankError):
    pass


class RerankQuotaError(RerankError):
    pass


@dataclass
class RerankResult:
    index: int
    relevance_score: float
    content: str


def _classify_rerank_error(error: Exception) -> RerankError:
    msg = str(error).lower()
    if any(kw in msg for kw in _AUTH_KEYWORDS):
        return RerankAuthError(f"Rerank API Key 无效: {error}")
    if any(kw in msg for kw in _QUOTA_KEYWORDS):
        return RerankQuotaError(f"Rerank API 额度不足: {error}")
    return RerankError(f"Rerank API 调用失败: {error}")


class ZhipuReranker:
    def __init__(self, api_key: str, model: str = "rerank") -> None:
        self._api_key = api_key
        self._model = model
        self._url = "https://open.bigmodel.cn/api/paas/v4/rerank"

    def rerank(
        self, query: str, documents: list[str], top_n: int = 5,
    ) -> list[RerankResult]:
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents[:128],
            "top_n": min(top_n, len(documents), 128),
            "return_documents": True,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        def do_call():
            resp = requests.post(
                self._url, json=payload, headers=headers, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            data = call_with_retry(do_call, _classify_rerank_error, non_retriable_types=(RerankAuthError, RerankQuotaError))
        except RerankAuthError:
            raise
        except RerankQuotaError:
            raise
        except RerankError as e:
            logger.warning("Rerank 失败，降级为原始排序: %s", e)
            return [
                RerankResult(index=i, relevance_score=1.0, content=doc)
                for i, doc in enumerate(documents[:top_n])
            ]

        results: list[RerankResult] = []
        for item in data.get("results", []):
            doc_info = item.get("document", {})
            content = doc_info.get("text", "") if isinstance(doc_info, dict) else str(doc_info)
            results.append(RerankResult(
                index=item["index"],
                relevance_score=item["relevance_score"],
                content=content,
            ))

        logger.info("Rerank 完成：输入 %d 文档，返回 %d 条", len(documents), len(results))
        return results
