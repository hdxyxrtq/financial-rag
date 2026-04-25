"""查询 API 路由 — 同步查询、SSE 流式查询、健康检查、Metrics。"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.api.deps import get_pipeline, get_store
from src.api.schemas import (
    EvalRequest,
    EvalResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import Config
from src.correction.pipeline import SelfCorrectingPipeline
from src.correction.types import CorrectionResult
from src.metrics.collector import MetricsCollector
from src.rag_pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """同步查询：检索 + 重排序（可选）+ LLM 生成。"""
    t0 = time.perf_counter()
    try:
        effective_pipeline: RAGPipeline | SelfCorrectingPipeline = pipeline
        if body.self_correction:
            config = Config()
            effective_pipeline = SelfCorrectingPipeline(
                pipeline=pipeline,
                config=config.self_correction,
                api_key=config.siliconflow_api_key,
                base_url=config.self_correction.siliconflow_base_url,
                model=config.self_correction.siliconflow_model,
            )

        kwargs: dict = {}
        if body.top_k is not None:
            kwargs["top_k"] = body.top_k

        result = await effective_pipeline.aquery(
            question=body.question,
            chat_history=body.chat_history,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        correction_data = None
        raw_correction = result.get("correction")
        if raw_correction is not None and isinstance(raw_correction, CorrectionResult):
            c = raw_correction
            correction_data = {
                "passed": c.passed,
                "flagged_claims": c.flagged_claims,
                "confidence": c.confidence,
                "layer_results": {
                    k: str(v) for k, v in (c.layer_results or {}).items()
                },
            }

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            cached=False,
            latency_ms=round(latency_ms, 2),
            correction=correction_data,
        )
    except Exception as e:
        logger.error("查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {e}") from e


@router.post("/query/stream")
async def query_stream(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EventSourceResponse:
    """SSE 流式查询：先推送 sources，再逐 token 推送 answer。"""

    async def _generate() -> AsyncGenerator[dict, None]:
        kwargs: dict = {}
        if body.top_k is not None:
            kwargs["top_k"] = body.top_k

        try:
            async for chunk in pipeline.astream_query(
                question=body.question,
                chat_history=body.chat_history,
                **kwargs,
            ):
                if chunk.get("type") == "sources":
                    import json

                    yield {"event": "sources", "data": json.dumps(chunk["sources"], ensure_ascii=False)}
                elif chunk.get("type") == "answer":
                    yield {"event": "answer", "data": chunk.get("content", "")}
        except Exception as e:
            logger.error("流式查询失败: %s", e, exc_info=True)
            import json

            yield {"event": "error", "data": json.dumps({"detail": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(_generate())


@router.get("/health", response_model=HealthResponse)
async def health(
    store: ChromaStore = Depends(get_store),
) -> HealthResponse:
    """健康检查：返回系统状态和已索引文档块数。"""
    stats = store.get_stats()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        indexed_chunks=stats.get("document_count", 0),
    )


@router.post("/eval", response_model=EvalResponse)
async def run_eval(
    body: EvalRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvalResponse:
    """运行 RAGAS 评估（需要 API Key 和充足额度）。"""
    try:
        from pathlib import Path

        from src.evaluation.ragas_eval import RAGEvaluator

        config = pipeline._llm  # noqa: SLF001 — 获取配置信息
        eval_path = body.eval_path or "data/eval/financial_qa_eval.json"
        eval_file = Path(eval_path)
        if not eval_file.exists():
            raise HTTPException(status_code=404, detail=f"评估数据集不存在: {eval_path}")

        import json

        with open(eval_file, encoding="utf-8") as f:
            eval_data = json.load(f)

        questions = [d["question"] for d in eval_data]
        # 简化：使用 pipeline 生成回答后评估
        responses = []
        contexts_list = []
        for q in questions:
            result = await pipeline.aquery(q)
            responses.append(result.get("answer", ""))
            sources = result.get("sources", [])
            contexts_list.append([s.get("content", "") for s in sources])

        references = [d.get("reference", "") for d in eval_data]

        t0 = time.perf_counter()
        evaluator = RAGEvaluator(
            api_key=config._api_key,  # noqa: SLF001
            model=config._model,  # noqa: SLF001
        )
        scores = evaluator.evaluate(
            questions=questions,
            responses=responses,
            contexts=contexts_list,
            references=references if any(references) else None,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return EvalResponse(
            scores=scores,
            sample_count=len(questions),
            latency_ms=round(latency_ms, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("评估失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"评估失败: {e}") from e


@router.get("/metrics")
async def get_metrics() -> dict:
    """返回 MetricsCollector 的汇总统计数据。"""
    collector = MetricsCollector()
    return collector.summary()


@router.delete("/metrics")
async def clear_metrics() -> dict:
    """清空所有 Metrics 记录。"""
    collector = MetricsCollector()
    collector.clear()
    return {"status": "ok", "message": "Metrics cleared"}
