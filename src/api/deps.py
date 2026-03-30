"""FastAPI 依赖注入 — Pipeline 和 Store 实例的构建。"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import Config, RetrieverConfig
from src.embeddings.zhipu_embedder import ZhipuEmbedder
from src.generator.query_rewriter import QueryRewriter
from src.generator.zhipu_llm import ZhipuLLM
from src.rag_pipeline import RAGPipeline
from src.reranker.zhipu_reranker import ZhipuReranker
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_config() -> Config:
    return Config()


@lru_cache(maxsize=1)
def _get_store() -> ChromaStore:
    config = _get_config()
    return ChromaStore(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
    )


@lru_cache(maxsize=1)
def _get_embedder() -> ZhipuEmbedder:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("ZHIPU_API_KEY 未配置，请设置环境变量或 .env 文件")
    return ZhipuEmbedder(
        api_key=api_key,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
    )


@lru_cache(maxsize=1)
def _get_llm() -> ZhipuLLM:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("ZHIPU_API_KEY 未配置，请设置环境变量或 .env 文件")
    return ZhipuLLM(
        api_key=api_key,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def get_store() -> ChromaStore:
    """FastAPI 依赖：获取 ChromaStore 实例。"""
    return _get_store()


def get_pipeline() -> RAGPipeline:
    """FastAPI 依赖：构建 RAGPipeline 实例。

    默认使用 hybrid 策略，不带 Reranker。
    """
    config = _get_config()
    embedder = _get_embedder()
    store = _get_store()
    llm = _get_llm()

    # 构建 Retriever（默认 hybrid）
    base_retriever = Retriever(
        embedder,
        store,
        RetrieverConfig(
            top_k=config.hybrid.vector_fetch_k,
            score_threshold=0.0,
        ),
    )
    bm25_retriever = BM25Retriever(store)
    retriever = HybridRetriever(
        retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        config=config.hybrid,
        score_threshold=config.retriever.score_threshold,
    )

    # Reranker（默认关闭）
    reranker = ZhipuReranker(api_key=config.api_key or "") if config.reranker.enabled else None
    reranker_config = config.reranker if config.reranker.enabled and config.api_key else None

    # Query Rewriter（默认关闭）
    query_rewriter = QueryRewriter(llm=llm) if config.rag.query_rewrite else None

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        config=config.rag,
        reranker=reranker,
        reranker_config=reranker_config,
        query_rewriter=query_rewriter,
    )
