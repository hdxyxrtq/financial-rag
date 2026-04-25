from __future__ import annotations

import io
import json
import logging
import time

import pandas as pd
import streamlit as st

from src.metrics.collector import MetricsCollector
from src.ui.constants import _PROJECT_ROOT
from src.ui.services import _build_rag_pipeline

logger = logging.getLogger(__name__)


def render_eval_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        "<h1>RAG Evaluation</h1>"
        '<div class="subtitle">Automated quality assessment using RAGAS framework</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    api_key = st.session_state.api_key
    if not api_key:
        st.warning("Please configure API Key in sidebar before evaluation")
        return

    tab_eval, tab_metrics = st.tabs(["RAGAS Evaluation", "Real-time Metrics"])

    with tab_metrics:
        _render_metrics_tab()

    with tab_eval:
        _render_eval_content(api_key)


def _render_metrics_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        "<h1>System Metrics</h1>"
        '<div class="subtitle">Real-time query performance monitoring</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    collector = MetricsCollector()
    summary = collector.summary()

    if summary["total_queries"] == 0:
        st.info("暂无查询数据。请先进行查询，Metrics 会自动记录。")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", summary["total_queries"])
    col2.metric("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")
    col3.metric("Cache Hit Rate", f"{summary['cache_hit_rate'] * 100:.1f}%")
    col4.metric("Total Tokens", summary["total_input_tokens"] + summary["total_output_tokens"])

    col5, col6, col7 = st.columns(3)
    col5.metric("P50 Latency", f"{summary['p50_latency_ms']:.0f}ms")
    col6.metric("P95 Latency", f"{summary['p95_latency_ms']:.0f}ms")
    col7.metric("Avg Sources", f"{_avg_sources(summary):.1f}")

    st.divider()
    st.subheader("Recent Query Latency")
    recent = summary.get("recent_queries", [])
    if recent:
        latency_df = pd.DataFrame(
            [
                {
                    "Query": r["question"][:30],
                    "Total (ms)": r["total_ms"],
                    "Retrieve": r["retrieve_ms"],
                    "Generate": r["generate_ms"],
                }
                for r in recent
            ]
        )
        st.bar_chart(latency_df.set_index("Query")[["Total (ms)"]], use_container_width=True)

        st.subheader("Latency Breakdown")
        st.dataframe(latency_df, use_container_width=True, hide_index=True)

    strategy_dist = summary.get("strategy_distribution", {})
    if strategy_dist:
        st.subheader("Strategy Distribution")
        st.bar_chart(
            pd.DataFrame([{"Strategy": k, "Count": v} for k, v in strategy_dist.items()]).set_index("Strategy"),
            use_container_width=True,
        )

    if st.button("Clear Metrics", type="secondary"):
        collector.clear()
        st.success("Metrics cleared")
        st.rerun()


def _avg_sources(summary: dict) -> float:
    recent: list = summary.get("recent_queries", [])
    if not recent:
        return 0.0
    return float(sum(r.get("num_sources", 0) for r in recent) / len(recent))


def _render_eval_content(api_key: str) -> None:

    eval_source = st.radio(
        "评估数据来源",
        ["内置数据集", "上传 JSON 文件"],
        horizontal=True,
    )

    eval_samples = None
    eval_path = _PROJECT_ROOT / "data" / "eval" / "financial_qa_eval.json"

    if eval_source == "内置数据集":
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                eval_samples = json.load(f)
            st.success(f"已加载内置数据集：{len(eval_samples)} 条问答对")
        else:
            st.error("内置评估数据集不存在")
            return
    else:
        uploaded = st.file_uploader("上传 JSON 评估文件", type=["json"])
        if uploaded:
            try:
                content = uploaded.read().decode("utf-8")
                eval_samples = json.loads(content)
                st.success(f"已加载上传数据集：{len(eval_samples)} 条问答对")
            except Exception as e:
                st.error(f"解析 JSON 失败: {e}")
                return

    if eval_samples is None:
        return

    with st.expander("预览数据集", expanded=False):
        preview = []
        for s in eval_samples[:10]:
            preview.append(
                {
                    "Question": s.get("question", "")[:50],
                    "Reference": (s.get("reference", "") or "")[:50],
                }
            )
        st.dataframe(preview, use_container_width=True, hide_index=True)
        if len(eval_samples) > 10:
            st.caption(f"... 共 {len(eval_samples)} 条")

    if st.button("开始评估", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing evaluator...")
        status_text = st.empty()

        try:
            from src.evaluation.ragas_eval import RAGEvaluator

            evaluator = RAGEvaluator(api_key=api_key)
            pipeline = _build_rag_pipeline(api_key=api_key)

            progress_bar.progress(0.2, text="Running pipeline queries...")

            questions = [s["question"] for s in eval_samples]
            references = [s.get("reference") for s in eval_samples if s.get("reference")]

            responses = []
            all_contexts = []
            for i, sample in enumerate(eval_samples):
                progress = 0.2 + 0.5 * (i / len(eval_samples))
                progress_bar.progress(progress, text=f"Querying {i + 1}/{len(eval_samples)}...")
                try:
                    result = pipeline.query(sample["question"])
                    answer = result.get("answer", "")
                    responses.append(str(answer) if answer else "")
                    raw_sources: list[dict] = result.get("sources", [])  # type: ignore[assignment]
                    all_contexts.append([str(src["content"]) for src in raw_sources])
                except Exception as e:
                    status_text.warning(f"[WARN] Q{i + 1} failed: {e}")
                    responses.append("")
                    all_contexts.append([])

            progress_bar.progress(0.7, text="Running RAGAS evaluation...")
            status_text.info("Running RAGAS metrics...")

            ref_list = [s.get("reference") or "" for s in eval_samples] if references else None
            scores = evaluator.evaluate(
                questions=questions,
                responses=responses,
                contexts=all_contexts,
                references=ref_list,
            )

            progress_bar.progress(1.0, text="Evaluation complete")
            time.sleep(0.5)
            progress_bar.empty()

            st.divider()
            st.subheader("评估结果")

            metrics_df = pd.DataFrame([{"Metric": k.replace("_", " ").title(), "Score": v} for k, v in scores.items()])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            avg = sum(scores.values()) / len(scores) if scores else 0
            st.metric("Average Score", f"{avg:.4f}")

            csv_buffer = io.StringIO()
            rows = [
                {"question": q, "response": r, **{k: "" for k in scores}}
                for q, r in zip(questions, responses, strict=False)
            ]
            pd.DataFrame(rows).to_csv(csv_buffer, index=False)
            st.download_button(
                "Export Results (CSV)",
                csv_buffer.getvalue(),
                "rag_eval_results.csv",
                "text/csv",
            )

        except ImportError as e:
            progress_bar.empty()
            st.error(f"缺少评估依赖: {e}。请运行 pip install ragas datasets langchain-openai")
        except Exception as e:
            progress_bar.empty()
            logger.error("评估失败: %s", e, exc_info=True)
            st.error(f"评估失败: {e}")
