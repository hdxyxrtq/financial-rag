"""异步 Pipeline 测试：验证 aquery、aretrieve、arerank、arewrite 等异步方法。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import HybridConfig, RAGConfig, RerankerConfig
from src.rag_pipeline import RAGPipeline
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import RetrievalResult, Retriever


def _make_result(
    content: str = "测试内容",
    score: float = 0.85,
    doc_id: str = "chunk-0",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        metadata={"title": "测试文档", "source": "test.txt"},
        doc_id=doc_id,
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.achat = AsyncMock(return_value="这是一个测试回答")
    llm.astream_chat = AsyncMock()
    return llm


@pytest.fixture
def mock_retriever() -> AsyncMock:
    r = AsyncMock(spec=Retriever)
    r.aretrieve = AsyncMock(return_value=[
        _make_result("文档A", 0.9, "doc_1"),
        _make_result("文档B", 0.7, "doc_2"),
    ])
    return r


@pytest.fixture
def mock_reranker() -> AsyncMock:
    reranker = AsyncMock()
    reranker.arerank = AsyncMock(return_value=[])
    return reranker


@pytest.fixture
def mock_config() -> RAGConfig:
    return RAGConfig(max_context_tokens=4000)


class TestAquery:
    @pytest.mark.asyncio
    async def test_aquery_normal_flow(self, mock_retriever, mock_llm, mock_config):
        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        result = await pipeline.aquery("什么是A股")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "这是一个测试回答"
        assert len(result["sources"]) == 2
        mock_retriever.aretrieve.assert_called_once()
        mock_llm.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_aquery_empty_question(self, mock_retriever, mock_llm, mock_config):
        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        result = await pipeline.aquery("")

        assert "请输入有效的问题" in result["answer"]
        assert result["sources"] == []
        mock_retriever.aretrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_aquery_no_results(self, mock_retriever, mock_llm, mock_config):
        mock_retriever.aretrieve = AsyncMock(return_value=[])
        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        result = await pipeline.aquery("不相关的问题")

        assert "未找到" in result["answer"]
        mock_llm.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_aquery_retrieve_exception_degrades(self, mock_retriever, mock_llm, mock_config):
        mock_retriever.aretrieve = AsyncMock(side_effect=Exception("检索异常"))
        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        result = await pipeline.aquery("测试")

        assert "未找到" in result["answer"]

    @pytest.mark.asyncio
    async def test_aquery_with_reranker(self, mock_retriever, mock_llm, mock_config, mock_reranker):
        from src.reranker.zhipu_reranker import RerankResult

        mock_reranker.arerank = AsyncMock(return_value=[
            RerankResult(index=0, relevance_score=0.95, content="文档A"),
        ])
        reranker_config = RerankerConfig(top_n=5, min_score=0.0)

        pipeline = RAGPipeline(
            mock_retriever, mock_llm, mock_config,
            reranker=mock_reranker, reranker_config=reranker_config,
        )
        result = await pipeline.aquery("测试")

        mock_reranker.arerank.assert_called_once()
        assert len(result["sources"]) == 1


class TestAstreamQuery:
    @pytest.mark.asyncio
    async def test_astream_query_yields_sources_and_answer(self, mock_retriever, mock_llm, mock_config):
        async def fake_stream(*args, **kwargs):
            yield "你"
            yield "好"

        mock_llm.astream_chat = fake_stream

        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        events = []
        async for event in pipeline.astream_query("测试"):
            events.append(event)

        source_events = [e for e in events if e["type"] == "sources"]
        answer_events = [e for e in events if e["type"] == "answer"]
        assert len(source_events) == 1
        assert len(answer_events) == 2
        assert answer_events[0]["content"] == "你"
        assert answer_events[1]["content"] == "好"

    @pytest.mark.asyncio
    async def test_astream_query_empty_question(self, mock_retriever, mock_llm, mock_config):
        pipeline = RAGPipeline(mock_retriever, mock_llm, mock_config)
        events = []
        async for event in pipeline.astream_query(""):
            events.append(event)

        assert len(events) == 1
        assert events[0]["type"] == "answer"
        assert "请输入" in events[0]["content"]


class TestHybridAretrieve:
    @pytest.mark.asyncio
    async def test_async_gather_parallel(self):
        mock_base = AsyncMock(spec=Retriever)
        mock_base.aretrieve = AsyncMock(return_value=[
            _make_result("向量结果", 0.9, "doc_1"),
        ])

        mock_bm25 = AsyncMock()
        mock_bm25.aretrieve = AsyncMock(return_value=[
            ("doc_1", 5.0),
        ])
        mock_bm25.get_documents_by_ids = MagicMock(return_value={
            "doc_1": {"content": "BM25内容", "metadata": {"source": "bm25"}},
        })

        config = HybridConfig(
            strategy="hybrid", enabled=True,
            rrf_k=60, vector_weight=0.6, bm25_weight=0.4,
            bm25_fetch_k=30, vector_fetch_k=30,
        )
        hybrid = HybridRetriever(mock_base, mock_bm25, config, score_threshold=0.0)
        results = await hybrid.aretrieve("测试", top_k=5)

        mock_base.aretrieve.assert_called_once()
        mock_bm25.aretrieve.assert_called_once()
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_vector_fallback_on_bm25_failure(self):
        mock_base = AsyncMock(spec=Retriever)
        mock_base.aretrieve = AsyncMock(return_value=[
            _make_result("向量结果", 0.9, "doc_1"),
        ])

        mock_bm25 = AsyncMock()
        mock_bm25.aretrieve = AsyncMock(side_effect=Exception("BM25 异常"))

        config = HybridConfig(
            strategy="hybrid", enabled=True,
            rrf_k=60, vector_weight=0.6, bm25_weight=0.4,
            bm25_fetch_k=30, vector_fetch_k=30,
        )
        hybrid = HybridRetriever(mock_base, mock_bm25, config, score_threshold=0.0)
        results = await hybrid.aretrieve("测试", top_k=5)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_bm25_fallback_on_vector_failure(self):
        mock_base = AsyncMock(spec=Retriever)
        mock_base.aretrieve = AsyncMock(side_effect=Exception("向量检索异常"))

        mock_bm25 = AsyncMock()
        mock_bm25.aretrieve = AsyncMock(return_value=[
            ("doc_1", 5.0),
        ])
        mock_bm25.get_documents_by_ids = MagicMock(return_value={
            "doc_1": {"content": "BM25降级结果", "metadata": {}},
        })

        config = HybridConfig(
            strategy="hybrid", enabled=True,
            rrf_k=60, vector_weight=0.6, bm25_weight=0.4,
            bm25_fetch_k=30, vector_fetch_k=30,
        )
        hybrid = HybridRetriever(mock_base, mock_bm25, config, score_threshold=0.0)
        results = await hybrid.aretrieve("测试", top_k=5)

        assert len(results) >= 1


class TestArewrite:
    @pytest.mark.asyncio
    async def test_arewrite_with_history(self):
        mock_llm = AsyncMock()
        mock_llm.achat = AsyncMock(return_value="沪深300指数的市盈率是多少？")

        from src.generator.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(mock_llm)
        rewritten = await rewriter.arewrite(
            "那它的市盈率呢？",
            [
                {"role": "user", "content": "沪深300的成分股有哪些？"},
                {"role": "assistant", "content": "沪深300指数成分股..."},
            ],
        )

        mock_llm.achat.assert_called_once()
        assert "沪深300" in rewritten

    @pytest.mark.asyncio
    async def test_arewrite_empty_history_returns_original(self):
        mock_llm = AsyncMock()
        from src.generator.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.arewrite("什么是GDP", [])
        assert result == "什么是GDP"
        mock_llm.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_arewrite_failure_returns_original(self):
        mock_llm = AsyncMock()
        mock_llm.achat = AsyncMock(side_effect=Exception("LLM 异常"))

        from src.generator.query_rewriter import QueryRewriter
        rewriter = QueryRewriter(mock_llm)

        result = await rewriter.arewrite(
            "那它的市盈率呢？",
            [{"role": "user", "content": "沪深300是什么？"}],
        )
        assert result == "那它的市盈率呢？"
