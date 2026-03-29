from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jieba
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_FINANCIAL_TERMS = [
    "资产负债率", "净资产收益率", "市盈率", "市净率", "毛利率", "净利率",
    "营业收入", "净利润", "总资产", "流动比率", "速动比率", "资产回报率",
    "股东权益", "息税前利润", "经营活动现金流", "投资活动现金流",
    "筹资活动现金流", "应收账款周转率", "存货周转率", "沪深300",
    "中证500", "上证综指", "深证成指", "创业板指", "科创板",
    "融券", "融资融券", "涨跌停", "IPO", "市盈率TTM",
    "每股收益", "股息率", "ROE", "ROA", "EBITDA",
]


def _init_financial_dict() -> None:
    for term in _FINANCIAL_TERMS:
        jieba.add_word(term)


_init_financial_dict()


class BM25Retriever:
    def __init__(self, vectorstore: ChromaStore) -> None:
        self._vectorstore = vectorstore
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._dirty = True

    def mark_dirty(self) -> None:
        """标记 BM25 索引需要重建。
        
        Note: 当前通过 _init_bm25_retriever.clear() (st.cache_resource)
        销毁缓存实例来实现索引刷新，此方法保留供未来使用。
        """
        self._dirty = True
        logger.debug("BM25 索引标记为 dirty")

    def get_documents_by_ids(self, ids: list[str]) -> dict[str, dict]:
        return self._vectorstore.get_documents_by_ids(ids)

    def retrieve(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        if self._dirty:
            self._rebuild_index()

        if self._bm25 is None or not self._doc_ids:
            return []

        tokens = jieba.lcut_for_search(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.info("BM25 检索完成：query=%r, 返回 %d 条", query, min(top_k, len(ranked)))
        return ranked[:top_k]

    def _rebuild_index(self) -> None:
        logger.info("正在重建 BM25 索引...")
        try:
            all_docs = self._vectorstore.get_all_documents()
        except Exception as e:
            logger.error("从 ChromaDB 读取文档失败: %s", e)
            # 保留 dirty 状态以便下次调用时重新重建索引
            self._bm25 = None
            self._doc_ids = []
            # Keep dirty=True so next retrieve() will retry rebuild
            return

        docs = list(all_docs.values())
        ids = list(all_docs.keys())

        if not docs:
            self._bm25 = None
            self._doc_ids = []
            self._dirty = False
            logger.info("ChromaDB 无文档，BM25 索引为空")
            return

        tokenized = [jieba.lcut_for_search(doc) for doc in docs if doc]
        ids = [doc_id for doc, doc_id in zip(docs, ids) if doc]
        self._bm25 = BM25Okapi(tokenized)
        self._doc_ids = ids
        self._dirty = False
        logger.info("BM25 索引重建完成，共 %d 条文档", len(ids))
