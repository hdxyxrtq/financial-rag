"""FastAPI 依赖注入 — Pipeline 和 Store 实例的构建。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import cast

from src.config import Config, RetrieverConfig
from src.embeddings.siliconflow_embedder import SiliconFlowEmbedder
from src.generator.query_rewriter import QueryRewriter
from src.generator.siliconflow_llm import SiliconFlowLLM
from src.rag_pipeline import RAGPipeline
from src.reranker.local_reranker import LocalRreranker
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
def _get_embedder() -> SiliconFlowEmbedder:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY 未配置，请设置环境变量或 .env 文件")
    return SiliconFlowEmbedder(
        api_key=api_key,
        model=config.embedding.model,
    )


@lru_cache(maxsize=1)
def _get_llm() -> SiliconFlowLLM:
    config = _get_config()
    api_key = config.api_key
    if not api_key:
        raise ValueError("SILICONFLOW_API_KEY 未配置，请设置环境变量或 .env 文件")
    return SiliconFlowLLM(
        api_key=api_key,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def get_store() -> ChromaStore:
    return _get_store()


def get_pipeline() -> RAGPipeline:
    config = _get_config()
    embedder = _get_embedder()
    store = _get_store()
    llm = _get_llm()

    base_retriever = Retriever(
        cast(object, embedder),
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

    reranker = LocalRreranker() if config.reranker.enabled else None
    reranker_config = config.reranker if config.reranker.enabled else None

    query_rewriter = QueryRewriter(llm=llm) if config.rag.query_rewrite else None

    return RAGPipeline(
        retriever=retriever,
        llm=cast(object, llm),
        config=config.rag,
        reranker=cast(object, reranker) if reranker else None,
        reranker_config=reranker_config,
        query_rewriter=query_rewriter,
    )
