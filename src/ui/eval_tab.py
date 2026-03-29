from __future__ import annotations

import io
import json
import logging
import time

import pandas as pd
import streamlit as st

from src.ui.services import _build_rag_pipeline
from src.ui.constants import _PROJECT_ROOT

logger = logging.getLogger(__name__)


def render_eval_tab() -> None:
    st.markdown(
        '<div class="main-header">'
        '<h1>RAG Evaluation</h1>'
        '<div class="subtitle">Automated quality assessment using RAGAS framework</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(201,168,76,0.2);margin:0.4rem 0 1rem 0;">', unsafe_allow_html=True)

    api_key = st.session_state.api_key
    if not api_key:
        st.warning("Please configure API Key in sidebar before evaluation")
        return

    eval_source = st.radio(
        "评估数据来源",
        ["内置数据集", "上传 JSON 文件"],
        horizontal=True,
    )

    eval_samples = None
    eval_path = _PROJECT_ROOT / "data" / "eval" / "financial_qa_eval.json"

    if eval_source == "内置数据集":
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
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
            preview.append({
                "Question": s.get("question", "")[:50],
                "Reference": (s.get("reference", "") or "")[:50],
            })
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
                progress_bar.progress(progress, text=f"Querying {i+1}/{len(eval_samples)}...")
                try:
                    result = pipeline.query(sample["question"])
                    responses.append(result["answer"])
                    all_contexts.append([src["content"] for src in result["sources"]])
                except Exception as e:
                    status_text.warning(f"[WARN] Q{i+1} failed: {e}")
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

            metrics_df = pd.DataFrame([
                {"Metric": k.replace("_", " ").title(), "Score": v}
                for k, v in scores.items()
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            avg = sum(scores.values()) / len(scores) if scores else 0
            st.metric("Average Score", f"{avg:.4f}")

            csv_buffer = io.StringIO()
            pd.DataFrame([
                {"question": q, "response": r, **{k: "" for k in scores}}
                for q, r in zip(questions, responses)
            ]).to_csv(csv_buffer, index=False)
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
