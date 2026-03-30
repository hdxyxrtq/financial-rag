from __future__ import annotations

from unittest.mock import MagicMock

from src.generator.query_rewriter import QueryRewriter


def _make_mock_llm(return_value: str = "沪深300指数的市盈率是多少？") -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = return_value
    return llm


class TestQueryRewriter:
    def test_rewrite_with_history(self):
        llm = _make_mock_llm()
        rewriter = QueryRewriter(llm)

        history = [
            {"role": "user", "content": "沪深300的成分股有哪些？"},
            {"role": "assistant", "content": "沪深300指数成分股覆盖了300只股票..."},
        ]
        result = rewriter.rewrite("那它的市盈率呢？", history)

        assert result == "沪深300指数的市盈率是多少？"
        llm.chat.assert_called_once()
        call_args = llm.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0
        assert call_args.kwargs["max_tokens"] == 256

    def test_no_history_returns_original(self):
        llm = _make_mock_llm()
        rewriter = QueryRewriter(llm)

        result = rewriter.rewrite("什么是量化交易？", [])

        assert result == "什么是量化交易？"
        llm.chat.assert_not_called()

    def test_none_history_returns_original(self):
        llm = _make_mock_llm()
        rewriter = QueryRewriter(llm)

        result = rewriter.rewrite("什么是量化交易？", None)

        assert result == "什么是量化交易？"
        llm.chat.assert_not_called()

    def test_empty_query_returns_empty(self):
        llm = _make_mock_llm()
        rewriter = QueryRewriter(llm)

        result = rewriter.rewrite("   ", [{"role": "user", "content": "hello"}])

        assert result == "   "
        llm.chat.assert_not_called()

    def test_rewrite_failure_returns_original(self):
        llm = MagicMock()
        llm.chat.side_effect = Exception("API timeout")
        rewriter = QueryRewriter(llm)

        history = [
            {"role": "user", "content": "沪深300是什么？"},
            {"role": "assistant", "content": "沪深300是..."},
        ]
        original = "那它的市盈率呢？"
        result = rewriter.rewrite(original, history)

        assert result == original

    def test_rewrite_empty_response_returns_original(self):
        llm = _make_mock_llm(return_value="   ")
        rewriter = QueryRewriter(llm)

        history = [{"role": "user", "content": "上证指数怎么样？"}]
        original = "它最近涨了吗？"
        result = rewriter.rewrite(original, history)

        assert result == original

    def test_rewrite_truncates_long_history(self):
        llm = _make_mock_llm()
        rewriter = QueryRewriter(llm)

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"message_{i:03d}"}
            for i in range(12)
        ]

        rewriter.rewrite("下一个问题", long_history)

        call_args = llm.chat.call_args
        user_prompt = call_args.kwargs["messages"][0]["content"]
        assert "message_000" not in user_prompt
        assert "message_005" not in user_prompt
        assert "message_006" in user_prompt
        assert "message_011" in user_prompt
