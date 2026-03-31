from __future__ import annotations

import logging

import streamlit as st

from src.generator.zhipu_llm import (
    LLMAuthError,
    LLMError,
    LLMQuotaError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.rag_pipeline import RAGPipelineError
from src.ui.constants import _extract_keywords, _highlight_keywords
from src.ui.services import _build_rag_pipeline

logger = logging.getLogger(__name__)


def _get_messages() -> list[dict]:
    cid = st.session_state.current_conversation_id
    conversations: dict = st.session_state.conversations
    return conversations.setdefault(cid, [])


def render_chat_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        "<h1>Financial RAG — Query Terminal</h1>"
        '<div class="subtitle">Retrieval-Augmented Generation for Financial Intelligence</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    messages = _get_messages()

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("References", expanded=False):
                    for src in msg["sources"]:
                        title = src.get("metadata", {}).get("title", src.get("metadata", {}).get("source", "未知来源"))
                        score = src.get("score", 0)
                        st.markdown(f"**📄 {title}** （相似度: {score:.2f}）")
                        content = src.get("content", "")
                        if msg.get("query"):
                            keywords = _extract_keywords(msg["query"])
                            content = _highlight_keywords(content, keywords)
                        st.markdown(content)
                        st.divider()

    if st.session_state.retrieval_results:
        with st.expander("Retrieval Details", expanded=False):
            query = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    query = m["content"]
                    break
            keywords = _extract_keywords(query) if query else []
            for i, result in enumerate(st.session_state.retrieval_results):
                meta = result.get("metadata", {})
                title = meta.get("title", meta.get("source", f"文档 {i + 1}"))
                score = result.get("score", 0)
                st.markdown(f"**[{i + 1}] 📄 {title}** — 相似度: `{score:.4f}`")
                content = result.get("content", "")
                if keywords:
                    content = _highlight_keywords(content, keywords)
                st.markdown(content)
                st.divider()

    if prompt := st.chat_input("请输入您的问题..."):
        api_key = st.session_state.api_key
        if not api_key:
            st.error("请先在侧边栏输入 API Key")
            return

        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Retrieving relevant documents...")
            try:
                pipeline = _build_rag_pipeline(api_key=api_key)

                sources = []
                answer_parts = []

                status.update(label="Generating response...")
                answer_placeholder = st.empty()

                for chunk in pipeline.stream_query(
                    question=prompt,
                    chat_history=[m for m in messages[:-1]],
                ):
                    if chunk["type"] == "sources":
                        sources = chunk["sources"]
                        st.session_state.retrieval_results = sources
                    elif chunk["type"] == "answer":
                        answer_parts.append(chunk["content"])

                        answer_placeholder.markdown("".join(answer_parts) + "▌")
                answer = "".join(answer_parts)
                answer_placeholder.markdown(answer)
                status.update(label="Response complete", state="complete")

                messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "query": prompt,
                    }
                )

            except LLMAuthError as e:
                logger.error("LLM Auth 错误: %s", e, exc_info=True)
                st.error("API Key 无效，请在侧边栏重新输入")
                status.update(label="Auth failed", state="error")
                msg = {"role": "assistant", "content": "抱歉，API Key 无效，请在侧边栏重新输入。", "sources": []}
                messages.append(msg)
            except LLMQuotaError as e:
                logger.error("LLM 额度耗尽: %s", e, exc_info=True)
                st.error("API 额度已用完，请到 [智谱开放平台](https://open.bigmodel.cn/) 充值或更换 Key")
                status.update(label="Quota exceeded", state="error")
                msg = {"role": "assistant", "content": "抱歉，API 额度已用完，请充值或更换 Key。", "sources": []}
                messages.append(msg)
            except LLMTimeoutError as e:
                logger.error("LLM 超时: %s", e, exc_info=True)
                st.error("请求超时，请检查网络连接后重试")
                status.update(label="Timeout", state="error")
                msg = {"role": "assistant", "content": "抱歉，请求超时，请检查网络连接后重试。", "sources": []}
                messages.append(msg)
            except LLMRateLimitError as e:
                logger.error("LLM 频率限制: %s", e, exc_info=True)
                st.error("请求过于频繁，请等待几秒后重试")
                status.update(label="Rate limited", state="error")
                messages.append({"role": "assistant", "content": "抱歉，请求过于频繁，请稍后再试。", "sources": []})
            except LLMError as e:
                logger.error("LLM 错误: %s", e, exc_info=True)
                st.error(f"LLM 错误: {e}")
                status.update(label="LLM error", state="error")
                messages.append({"role": "assistant", "content": f"抱歉，LLM 错误: {e}", "sources": []})
            except RAGPipelineError as e:
                logger.error("RAG 流程错误: %s", e, exc_info=True)
                st.error(f"RAG 流程错误: {e}")
                status.update(label="Pipeline error", state="error")
                messages.append({"role": "assistant", "content": f"抱歉，RAG 流程错误: {e}", "sources": []})
            except Exception as e:
                logger.error("未知错误: %s", e, exc_info=True)
                st.error("发生未知错误，请刷新页面重试")
                status.update(label="Unknown error", state="error")
                messages.append({"role": "assistant", "content": "抱歉，发生未知错误，请刷新页面重试。", "sources": []})
