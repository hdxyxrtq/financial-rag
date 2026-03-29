from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.config import RetrieverConfig
    from src.embeddings.zhipu_embedder import ZhipuEmbedder
    from src.vectorstore.chroma_store import ChromaStore


class RetrievalError(Exception):
    pass


@dataclass
class RetrievalResult:
    """单条检索结果。"""

    content: str
    score: float
    metadata: dict[str, str]
    doc_id: str


class Retriever:
    """检索器：Embedding 查询 → ChromaDB 语义检索 → 结果过滤与排序。"""

    def __init__(self, embedder: ZhipuEmbedder, vectorstore: ChromaStore, config: RetrieverConfig) -> None:
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """语义检索，返回按相似度降序排列的结果。

        Args:
            query: 用户查询文本。
            top_k: 返回结果数量，默认取配置值。
            where: 元数据过滤条件（ChromaDB where 语法）。

        Returns:
            过滤后的检索结果列表。

        Raises:
            RetrievalError: 向量化或检索过程发生错误。
        """
        try:
            query_embedding = self._embedder.embed_query(query)
        except Exception as e:
            raise RetrievalError(f"查询向量化失败: {e}") from e

        k = top_k if top_k is not None else self._config.top_k

        logger.debug("检索 query=%r, top_k=%d, where=%s", query, k, where)

        raw_results = self._vectorstore.search(
            query_embedding=query_embedding,
            top_k=k,
            where=where,
        )

        # 转换并过滤低分结果
        threshold = self._config.score_threshold
        results: list[RetrievalResult] = []
        for item in raw_results:
            if item["score"] < threshold:
                continue
            results.append(RetrievalResult(
                content=item["content"],
                score=item["score"],
                metadata=item.get("metadata") or {},
                doc_id=item.get("id", ""),
            ))

        # 按相似度降序排列
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "检索完成：返回 %d/%d 条结果（阈值 %.2f）",
            len(results), len(raw_results), threshold,
        )
        return results
