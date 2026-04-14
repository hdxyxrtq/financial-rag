from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Generator
from string import Template
from time import perf_counter
from typing import TYPE_CHECKING, TypeAlias

from src.cache import QueryCache
from src.metrics.collector import MetricsCollector, QueryMetrics
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:
    import tiktoken

    from src.config import RAGConfig, RerankerConfig
    from src.generator.query_rewriter import QueryRewriter
    from src.generator.zhipu_llm import ZhipuLLM
    from src.reranker.zhipu_reranker import ZhipuReranker

RetrieverType: TypeAlias = Retriever | HybridRetriever

logger = logging.getLogger(__name__)

_cached_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding | None:
    global _cached_tokenizer
    if _cached_tokenizer is None:
        try:
            import tiktoken

            _cached_tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass
    return _cached_tokenizer


# system prompt 模板
_SYSTEM_PROMPT_TEMPLATE = """\
你是一位严谨的金融分析师。请严格遵守以下规则回答问题：

## 回答规则
1. **只根据参考资料**：所有事实必须来自下方参考资料，禁止编造、推测或使用外部知识
2. **先分析再回答**：先判断问题类型（定义/数值/对比/时间），再从资料中定位相关信息
3. **标注来源**：每个事实后标注 [来源 N]，N 对应参考资料的编号
4. **资料不足时**：明确说明"根据现有参考资料无法完整回答"，指出缺少什么信息
5. **简洁直接**：只回答被问到的问题，不展开无关内容

## 参考资料
$retrieved_documents

## 问题
$user_question"""


class RAGPipelineError(Exception):
    pass


class RAGPipeline:
    """RAG 主流程编排：检索 → 构造 Prompt → LLM 生成。"""

    def __init__(
        self,
        retriever: Retriever | HybridRetriever,
        llm: ZhipuLLM,
        config: RAGConfig,
        reranker: ZhipuReranker | None = None,
        reranker_config: RerankerConfig | None = None,
        query_rewriter: QueryRewriter | None = None,
        cache: QueryCache | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_tokens = config.max_context_tokens
        self._reranker = reranker
        self._reranker_config = reranker_config
        self._query_rewriter = query_rewriter
        self._cache = cache

    def query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> dict:
        """同步查询：检索 → 构造 Prompt → 生成回答。

        Args:
            question: 用户问题。
            chat_history: 历史对话消息列表，每项为 {"role": ..., "content": ...}。
            **retrieve_kwargs: 传递给 retriever.retrieve() 的额外参数（top_k, where 等）。

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        if not question or not question.strip():
            return {"answer": "请输入有效的问题。", "sources": []}
        if len(question) > 4096:
            return {"answer": "问题过长，请精简后重试。", "sources": []}

        collector = MetricsCollector()
        from_cache = False

        if self._cache is not None:
            cached = self._run_async(self._cache.get(question))
            if cached is not None:
                logger.info("Cache hit for sync query: %s", question)
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        total_ms=0.0,
                        cache_hit=True,
                    )
                )
                return cached

        if chat_history and self._query_rewriter:
            question = self._query_rewriter.rewrite(question, chat_history)

        t_retrieve = perf_counter()
        results = self._retrieve_or_fallback(question, **retrieve_kwargs)
        retrieve_ms = (perf_counter() - t_retrieve) * 1000
        if results is None:
            collector.record(
                QueryMetrics(
                    question=question[:50],
                    retrieve_ms=retrieve_ms,
                    total_ms=retrieve_ms,
                    num_sources=0,
                )
            )
            return {"answer": "抱歉，在知识库中未找到与您问题相关的信息。", "sources": []}

        rerank_ms = 0.0
        if self._reranker and results:
            t_rerank = perf_counter()
            results = self._rerank_results(question, results)
            rerank_ms = (perf_counter() - t_rerank) * 1000

        formatted_context, sources = self._format_context(results)
        trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )

        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        t_generate = perf_counter()
        try:
            answer = self._llm.chat(system_prompt, messages)
        except Exception as e:
            raise RAGPipelineError(f"LLM 生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000

        total_ms = retrieve_ms + rerank_ms + generate_ms
        collector.record(
            QueryMetrics(
                question=question[:50],
                retrieve_ms=round(retrieve_ms, 2),
                rerank_ms=round(rerank_ms, 2),
                generate_ms=round(generate_ms, 2),
                total_ms=round(total_ms, 2),
                cache_hit=from_cache,
                strategy=self._get_strategy_name(),
                reranker_enabled=self._reranker is not None,
                num_sources=len(sources),
            )
        )

        result = {"answer": answer, "sources": sources}
        if self._cache is not None:
            self._run_async(self._cache.put(question, result))
        return result

    def stream_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> Generator[dict, None, None]:
        """流式查询：检索 → 构造 Prompt → 逐 token 生成回答。

        Yields:
            首先产出 {"type": "sources", "sources": list[dict]}，
            之后逐 token 产出 {"type": "answer", "content": str}。
        """
        if not question or not question.strip():
            yield {"type": "answer", "content": "请输入有效的问题。"}
            return
        if len(question) > 4096:
            yield {"type": "answer", "content": "问题过长，请精简后重试。"}
            return

        if self._cache is not None:
            cached = self._run_async(self._cache.get(question))
            if cached is not None:
                logger.info("Cache hit for stream_query: %s", question)
                yield {"type": "sources", "sources": cached.get("sources", [])}
                yield {"type": "answer", "content": cached.get("answer", "")}
                return

        if chat_history and self._query_rewriter:
            question = self._query_rewriter.rewrite(question, chat_history)

        results = self._retrieve_or_fallback(question, **retrieve_kwargs)
        if results is None:
            yield {"type": "answer", "content": "抱歉，在知识库中未找到与您问题相关的信息。"}
            return

        if self._reranker and results:
            results = self._rerank_results(question, results)

        formatted_context, sources = self._format_context(results)
        trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        yield {"type": "sources", "sources": sources}

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )
        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        full_answer_parts: list[str] = []
        try:
            for chunk in self._llm.stream_chat(system_prompt, messages):
                full_answer_parts.append(chunk)
                yield {"type": "answer", "content": chunk}
        except Exception as e:
            raise RAGPipelineError(f"LLM 流式生成失败: {e}") from e

        if self._cache is not None:
            self._run_async(self._cache.put(question, {"answer": "".join(full_answer_parts), "sources": sources}))

    async def aquery(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> dict:
        """异步查询：检索 → 构造 Prompt → 生成回答。"""
        if not question or not question.strip():
            return {"answer": "请输入有效的问题。", "sources": []}
        if len(question) > 4096:
            return {"answer": "问题过长，请精简后重试。", "sources": []}

        collector = MetricsCollector()
        from_cache = False

        # 1) 通过语义缓存快速命中（阶段 2，若启用）
        if self._cache is not None:
            cached = await self._cache.get(question)
            if cached is not None:
                logger.info("Cache hit for async query: %s", question)
                collector.record(
                    QueryMetrics(
                        question=question[:50],
                        total_ms=0.0,
                        cache_hit=True,
                    )
                )
                return cached

        if chat_history and self._query_rewriter:
            question = await self._query_rewriter.arewrite(question, chat_history)

        t_retrieve = perf_counter()
        results = await self._aretrieve_or_fallback(question, **retrieve_kwargs)
        retrieve_ms = (perf_counter() - t_retrieve) * 1000
        if results is None:
            collector.record(
                QueryMetrics(
                    question=question[:50],
                    retrieve_ms=round(retrieve_ms, 2),
                    total_ms=round(retrieve_ms, 2),
                    num_sources=0,
                )
            )
            return {"answer": "抱歉，在知识库中未找到与您问题相关的信息。", "sources": []}

        rerank_ms = 0.0
        if self._reranker and results:
            t_rerank = perf_counter()
            results = await self._arerank_results(question, results)
            rerank_ms = (perf_counter() - t_rerank) * 1000

        formatted_context, sources = self._format_context(results)
        trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )

        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        t_generate = perf_counter()
        try:
            answer = await self._llm.achat(system_prompt, messages)
        except Exception as e:
            raise RAGPipelineError(f"LLM 异步生成失败: {e}") from e
        generate_ms = (perf_counter() - t_generate) * 1000

        total_ms = retrieve_ms + rerank_ms + generate_ms
        collector.record(
            QueryMetrics(
                question=question[:50],
                retrieve_ms=round(retrieve_ms, 2),
                rerank_ms=round(rerank_ms, 2),
                generate_ms=round(generate_ms, 2),
                total_ms=round(total_ms, 2),
                cache_hit=from_cache,
                strategy=self._get_strategy_name(),
                reranker_enabled=self._reranker is not None,
                num_sources=len(sources),
            )
        )

        # 2) 缓存写入
        if self._cache is not None:
            await self._cache.put(question, {"answer": answer, "sources": sources})
        return {"answer": answer, "sources": sources}

    async def astream_query(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        **retrieve_kwargs,
    ) -> AsyncGenerator[dict, None]:
        """异步流式查询：检索 → 构造 Prompt → 逐 token 生成回答。"""
        if not question or not question.strip():
            yield {"type": "answer", "content": "请输入有效的问题。"}
            return
        if len(question) > 4096:
            yield {"type": "answer", "content": "问题过长，请精简后重试。"}
            return

        if chat_history and self._query_rewriter:
            question = await self._query_rewriter.arewrite(question, chat_history)

        results = await self._aretrieve_or_fallback(question, **retrieve_kwargs)
        if results is None:
            yield {"type": "answer", "content": "抱歉，在知识库中未找到与您问题相关的信息。"}
            return

        if self._reranker and results:
            results = await self._arerank_results(question, results)

        formatted_context, sources = self._format_context(results)
        trimmed_context, sources = self._trim_context(formatted_context, self._max_context_tokens, sources)

        yield {"type": "sources", "sources": sources}

        system_prompt = Template(_SYSTEM_PROMPT_TEMPLATE).safe_substitute(
            retrieved_documents=trimmed_context,
            user_question=question,
        )
        messages = self._build_messages(
            chat_history,
            question,
            max_history_tokens=max(200, self._max_context_tokens * 2 // 5),
        )

        try:
            async for chunk in self._llm.astream_chat(system_prompt, messages):
                yield {"type": "answer", "content": chunk}
        except Exception as e:
            raise RAGPipelineError(f"LLM 异步流式生成失败: {e}") from e

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_strategy_name(self) -> str:
        if isinstance(self._retriever, HybridRetriever):
            return self._retriever._config.strategy  # noqa: SLF001
        return "vector"

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    async def _aretrieve_or_fallback(self, question: str, **retrieve_kwargs) -> list[RetrievalResult] | None:
        try:
            results = await self._retriever.aretrieve(question, **retrieve_kwargs)
        except Exception as e:
            logger.warning("异步检索异常，降级为无上下文回答: %s", e)
            return None

        if not results:
            logger.info("异步检索无结果，降级为无上下文回答")
            return None
        return results

    async def _arerank_results(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        cfg = self._reranker_config
        if cfg is None:
            return results

        contents = [r.content for r in results]
        try:
            reranker = self._reranker
            if reranker is None:
                return results
            rerank_results = await reranker.arerank(
                question,
                contents,
                top_n=cfg.top_n,
            )
        except Exception as e:
            logger.warning("异步 Rerank 失败，降级为原始排序: %s", e)
            return results

        reranked: list[RetrievalResult] = []
        for rr in rerank_results:
            if rr.relevance_score < cfg.min_score:
                continue
            if rr.index < 0 or rr.index >= len(results):
                logger.warning("Reranker returned out-of-range index %d, skipping", rr.index)
                continue
            original = results[rr.index]
            reranked.append(
                RetrievalResult(
                    content=original.content,
                    score=rr.relevance_score,
                    metadata=original.metadata,
                    doc_id=original.doc_id,
                )
            )

        logger.info("异步 Rerank 完成：输入 %d 条，输出 %d 条", len(results), len(reranked))
        return reranked

    def _retrieve_or_fallback(self, question: str, **retrieve_kwargs) -> list[RetrievalResult] | None:
        try:
            results = self._retriever.retrieve(question, **retrieve_kwargs)
        except Exception as e:
            logger.warning("检索异常，降级为无上下文回答: %s", e)
            return None

        if not results:
            logger.info("检索无结果，降级为无上下文回答")
            return None
        return results

    def _rerank_results(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        cfg = self._reranker_config
        if cfg is None:
            return results

        contents = [r.content for r in results]
        try:
            reranker = self._reranker
            if reranker is None:
                return results
            rerank_results = reranker.rerank(
                question,
                contents,
                top_n=cfg.top_n,
            )
        except Exception as e:
            logger.warning("Rerank 失败，降级为原始排序: %s", e)
            return results

        reranked: list[RetrievalResult] = []
        for rr in rerank_results:
            if rr.relevance_score < cfg.min_score:
                continue
            if rr.index < 0 or rr.index >= len(results):
                logger.warning("Reranker returned out-of-range index %d, skipping", rr.index)
                continue
            original = results[rr.index]
            reranked.append(
                RetrievalResult(
                    content=original.content,
                    score=rr.relevance_score,
                    metadata=original.metadata,
                    doc_id=original.doc_id,
                )
            )

        logger.info("Rerank 完成：输入 %d 条，输出 %d 条", len(results), len(reranked))
        return reranked

    @staticmethod
    def _build_messages(
        chat_history: list[dict[str, str]] | None,
        question: str,
        max_history_tokens: int = 2400,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if chat_history:
            trimmed = list(chat_history)
            try:
                encoder = _get_tokenizer()
                if encoder is not None:
                    history_text = "\n".join(m.get("content", "") for m in trimmed)
                    while len(trimmed) > 2 and len(encoder.encode(history_text)) > max_history_tokens:
                        trimmed = trimmed[2:]
                        history_text = "\n".join(m.get("content", "") for m in trimmed)
                else:
                    trimmed = chat_history[-10:]
            except Exception as e:
                logger.warning("Token trimming failed, using last 10 messages: %s", e)
                trimmed = chat_history[-10:]
            messages.extend(trimmed)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _format_context(
        results: list[RetrievalResult],
    ) -> tuple[str, list[dict]]:
        """将检索结果格式化为带编号的参考文本。

        Returns:
            (formatted_context, sources_list)
        """
        parts: list[str] = []
        sources: list[dict] = []

        for idx, result in enumerate(results, start=1):
            title = result.metadata.get("title", result.metadata.get("source", "未知来源"))
            parts.append(f"[来源 {idx}] 来源: {title}\n{result.content}")
            sources.append(
                {
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                }
            )

        formatted = "\n\n".join(parts)
        return formatted, sources

    def _count_tokens(self, text: str) -> int:
        encoder = _get_tokenizer()
        if encoder is not None:
            return len(encoder.encode(text))

        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_count = len(text) - chinese_count
        return int(chinese_count * 1.5 + other_count * 0.25)

    def _trim_context(
        self, formatted_docs: str, max_tokens: int, sources: list[dict] | None = None
    ) -> tuple[str, list[dict]]:
        """Trim context to fit within token limit, always returning a tuple."""
        if not formatted_docs:
            return "", sources or []

        if self._count_tokens(formatted_docs) <= max_tokens:
            return formatted_docs, sources or []

        blocks = re.split(r"(?=^\[来源 \d+\])", formatted_docs, flags=re.MULTILINE)

        blocks = [b.strip() for b in blocks if b.strip()]

        if sources:
            while len(blocks) > 1:
                total = "\n\n".join(blocks)
                if self._count_tokens(total) <= max_tokens:
                    return total, sources[: len(blocks)]
                blocks.pop()
            return (blocks[0] if blocks else formatted_docs), sources[:1]
        else:
            while len(blocks) > 1:
                total = "\n\n".join(blocks)
                if self._count_tokens(total) <= max_tokens:
                    return total, []
                blocks.pop()
            return (blocks[0] if blocks else formatted_docs), []
