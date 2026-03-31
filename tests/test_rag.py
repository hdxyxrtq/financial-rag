from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import RAGConfig, RetrieverConfig
from src.rag_pipeline import RAGPipeline, RAGPipelineError
from src.retriever.retriever import RetrievalError, RetrievalResult, Retriever

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_retrieval_result(
    content: str = "测试内容",
    score: float = 0.85,
    metadata: dict | None = None,
    doc_id: str = "chunk-0",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        metadata=metadata or {"title": "测试文档", "source": "test.txt"},
        doc_id=doc_id,
    )


@pytest.fixture
def retriever_config() -> RetrieverConfig:
    return RetrieverConfig(top_k=5, score_threshold=0.5)


@pytest.fixture
def mock_embedder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_vectorstore() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_config(retriever_config) -> RAGConfig:
    return RAGConfig(max_context_tokens=4000)


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


class TestRetriever:
    """Retriever 单元测试。"""

    def test_retrieve_success(self, mock_embedder, mock_vectorstore, retriever_config):
        """正常检索应返回排序后的 RetrievalResult 列表。"""
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vectorstore.search.return_value = [
            {"content": "文档A", "score": 0.9, "metadata": {"title": "A"}, "id": "a"},
            {"content": "文档B", "score": 0.6, "metadata": {"title": "B"}, "id": "b"},
            {"content": "文档C", "score": 0.3, "metadata": {"title": "C"}, "id": "c"},
        ]

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)
        results = retriever.retrieve("什么是A股")

        # 0.3 < threshold(0.5) 应被过滤
        assert len(results) == 2
        assert results[0].content == "文档A"
        assert results[1].content == "文档B"
        # 降序排列
        assert results[0].score >= results[1].score

    def test_retrieve_score_filtering(self, mock_embedder, mock_vectorstore, retriever_config):
        """低于 score_threshold 的结果应被过滤。"""
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_vectorstore.search.return_value = [
            {"content": "低分文档", "score": 0.2, "metadata": {}, "id": "x"},
        ]

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)
        results = retriever.retrieve("测试")

        assert results == []

    def test_retrieve_custom_top_k(self, mock_embedder, mock_vectorstore, retriever_config):
        """自定义 top_k 应传递给 vectorstore.search。"""
        mock_embedder.embed_query.return_value = [0.1]
        mock_vectorstore.search.return_value = []

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)
        retriever.retrieve("测试", top_k=3)

        mock_vectorstore.search.assert_called_once()
        call_kwargs = mock_vectorstore.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 3 or call_kwargs[1].get("top_k") == 3

    def test_retrieve_with_metadata_filter(self, mock_embedder, mock_vectorstore, retriever_config):
        """where 参数应传递给 vectorstore.search。"""
        mock_embedder.embed_query.return_value = [0.1]
        mock_vectorstore.search.return_value = []

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)
        retriever.retrieve("测试", where={"doc_type": "news"})

        mock_vectorstore.search.assert_called_once()
        call_kwargs = mock_vectorstore.search.call_args
        assert call_kwargs.kwargs.get("where") == {"doc_type": "news"} or call_kwargs[1].get("where") == {
            "doc_type": "news"
        }

    def test_retrieve_embedder_failure(self, mock_embedder, mock_vectorstore, retriever_config):
        """Embedding 失败应抛出 RetrievalError。"""
        mock_embedder.embed_query.side_effect = RuntimeError("API down")

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)

        with pytest.raises(RetrievalError, match="查询向量化失败"):
            retriever.retrieve("测试")

    def test_retrieve_empty_results(self, mock_embedder, mock_vectorstore, retriever_config):
        """无检索结果应返回空列表。"""
        mock_embedder.embed_query.return_value = [0.1]
        mock_vectorstore.search.return_value = []

        retriever = Retriever(mock_embedder, mock_vectorstore, retriever_config)
        results = retriever.retrieve("测试")

        assert results == []


# ---------------------------------------------------------------------------
# RAG Pipeline tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """RAGPipeline 单元测试。"""

    def _make_pipeline(self, mock_llm, mock_retriever, mock_config) -> RAGPipeline:
        return RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)

    def test_query_returns_answer_and_sources(self, mock_llm, mock_config):
        """正常查询应返回 answer + sources。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_retrieval_result(content="A股市场今日大涨", score=0.9, metadata={"title": "财经新闻"}),
            _make_retrieval_result(content="债券收益率下降", score=0.7, metadata={"title": "研报"}),
        ]
        mock_llm.chat.return_value = "根据参考资料，A股市场今日大涨。"

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        result = pipeline.query("A股市场怎么样？")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "根据参考资料，A股市场今日大涨。"
        assert len(result["sources"]) == 2

    def test_query_with_chat_history(self, mock_llm, mock_config):
        """带 chat_history 的查询应正确传递历史消息。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_retrieval_result(content="GDP增长5%", score=0.8),
        ]
        mock_llm.chat.return_value = "GDP增长5%"

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        history = [
            {"role": "user", "content": "中国经济怎么样？"},
            {"role": "assistant", "content": "请问具体想了解哪方面？"},
        ]
        pipeline.query("GDP增速如何？", chat_history=history)

        # 验证 LLM 收到的 messages 包含历史 + 当前问题
        call_args = mock_llm.chat.call_args
        messages = call_args[0][1] if call_args[0] else call_args.kwargs.get("messages", [])
        assert len(messages) == 3  # 2 历史 + 1 当前
        assert messages[-1]["content"] == "GDP增速如何？"

    def test_query_no_results_fallback(self, mock_llm, mock_config):
        """无检索结果时应优雅降级。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        result = pipeline.query("未知问题")

        assert "未找到" in result["answer"]
        assert result["sources"] == []
        # LLM 不应被调用
        mock_llm.chat.assert_not_called()

    def test_query_retrieval_error_fallback(self, mock_llm, mock_config):
        """检索异常时应优雅降级。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = Exception("网络错误")

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        result = pipeline.query("测试问题")

        assert "未找到" in result["answer"]
        mock_llm.chat.assert_not_called()

    def test_query_llm_error_raises(self, mock_llm, mock_config):
        """LLM 调用失败应抛出 RAGPipelineError。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_retrieval_result(content="金融数据", score=0.9),
        ]
        mock_llm.chat.side_effect = RuntimeError("GLM API error")

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)

        with pytest.raises(RAGPipelineError, match="LLM 生成失败"):
            pipeline.query("问题")

    def test_stream_query_yields_sources_then_answer(self, mock_llm, mock_config):
        """流式查询应先产出 sources，再逐 chunk 产出 answer。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            _make_retrieval_result(content="市场分析", score=0.8),
        ]
        mock_llm.stream_chat.return_value = iter(["你", "好", "世界"])

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        chunks = list(pipeline.stream_query("你好"))

        # 第一个 chunk 应为 sources
        assert chunks[0]["type"] == "sources"
        assert len(chunks[0]["sources"]) == 1
        # 后续 chunks 为 answer
        assert all(c["type"] == "answer" for c in chunks[1:])
        assert "".join(c["content"] for c in chunks[1:]) == "你好世界"

    def test_stream_query_no_results_fallback(self, mock_llm, mock_config):
        """流式查询无结果时应降级。"""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        pipeline = self._make_pipeline(mock_llm, mock_retriever, mock_config)
        chunks = list(pipeline.stream_query("未知"))

        assert len(chunks) == 1
        assert chunks[0]["type"] == "answer"
        assert "未找到" in chunks[0]["content"]


# ---------------------------------------------------------------------------
# RAG Pipeline context trimming tests
# ---------------------------------------------------------------------------


class TestContextTrimming:
    """上下文裁剪逻辑测试。"""

    @pytest.fixture
    def pipeline(self) -> RAGPipeline:
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        mock_config = MagicMock()
        return RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)

    def test_short_context_not_trimmed(self, pipeline):
        """短上下文不应被裁剪。"""
        short_text = "这是一段很短的文本。"
        result, sources = pipeline._trim_context(short_text, max_tokens=100)
        assert result == short_text
        assert sources == []

    def test_empty_context(self, pipeline):
        """空上下文应原样返回。"""
        result, sources = pipeline._trim_context("", 100)
        assert result == ""
        assert sources == []

    def test_long_context_trimmed(self, pipeline):
        """超长上下文应从末尾裁剪文档块。"""
        # 构造多个来源块（每块足够长以确保超限）
        blocks = []
        for i in range(5):
            blocks.append(f"[来源 {i + 1}] 来源: 文档{i}\n" + "这是一段很长的金融分析文本。" * 100)
        long_text = "\n\n".join(blocks)

        # 设置一个极小的 max_tokens 迫使裁剪
        result, sources = pipeline._trim_context(long_text, max_tokens=50)

        # 裁剪后应至少保留第一个块
        assert "[来源 1]" in result
        # 后面的块应被移除
        assert "[来源 5]" not in result

    def test_trim_preserves_at_least_one_block(self, pipeline):
        """即使单个块也超限，仍应保留至少一个块。"""
        single_block = "[来源 1] 来源: 文档A\n" + "超长文本" * 1000
        result, sources = pipeline._trim_context(single_block, max_tokens=1)

        # 至少保留一个块（第一个）
        assert "[来源 1]" in result


# ---------------------------------------------------------------------------
# Format context tests
# ---------------------------------------------------------------------------


class TestFormatContext:
    """_format_context 格式化测试。"""

    def test_format_single_result(self):
        """单条结果应正确格式化。"""
        results = [
            _make_retrieval_result(content="GDP数据", score=0.9, metadata={"title": "经济报告"}),
        ]
        formatted, sources = RAGPipeline._format_context(results)

        assert "[来源 1]" in formatted
        assert "GDP数据" in formatted
        assert "经济报告" in formatted
        assert len(sources) == 1
        assert sources[0]["score"] == 0.9

    def test_format_multiple_results(self):
        """多条结果应有正确编号。"""
        results = [
            _make_retrieval_result(content="内容A", score=0.9, metadata={"title": "文档A"}),
            _make_retrieval_result(content="内容B", score=0.7, metadata={"title": "文档B"}),
        ]
        formatted, sources = RAGPipeline._format_context(results)

        assert "[来源 1]" in formatted
        assert "[来源 2]" in formatted
        assert len(sources) == 2

    def test_format_empty_results(self):
        """空结果应返回空字符串和空列表。"""
        formatted, sources = RAGPipeline._format_context([])
        assert formatted == ""
        assert sources == []


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """输入校验测试。"""

    def test_query_empty_question(self, mock_llm, mock_config):
        """空问题应返回友好提示。"""
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)
        result = pipeline.query("")
        assert "请输入有效的问题" in result["answer"]
        assert result["sources"] == []
        mock_llm.chat.assert_not_called()

    def test_query_whitespace_question(self, mock_llm, mock_config):
        """纯空白问题应返回友好提示。"""
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)
        result = pipeline.query("   \n\t  ")
        assert "请输入有效的问题" in result["answer"]
        mock_llm.chat.assert_not_called()

    def test_query_too_long_question(self, mock_llm, mock_config):
        """超长问题应返回友好提示。"""
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)
        long_question = "测试" * 2049  # 4098 chars > 4096
        result = pipeline.query(long_question)
        assert "问题过长" in result["answer"]
        assert result["sources"] == []
        mock_llm.chat.assert_not_called()

    def test_stream_query_empty_question(self, mock_llm, mock_config):
        """流式查询空问题应返回友好提示。"""
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)
        chunks = list(pipeline.stream_query(""))
        assert len(chunks) == 1
        assert "请输入有效的问题" in chunks[0]["content"]
        mock_llm.stream_chat.assert_not_called()

    def test_stream_query_too_long_question(self, mock_llm, mock_config):
        """流式查询超长问题应返回友好提示。"""
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm, config=mock_config)
        long_question = "测试" * 2049
        chunks = list(pipeline.stream_query(long_question))
        assert len(chunks) == 1
        assert "问题过长" in chunks[0]["content"]
        mock_llm.stream_chat.assert_not_called()
