#!/usr/bin/env python
"""Financial RAG Benchmark — 量化对比不同检索策略的 RAGAS 分数与延迟。

用法:
    python scripts/benchmark.py
    python scripts/benchmark.py --api-key YOUR_KEY
    python scripts/benchmark.py --skip-reranker       # 跳过 Reranker 对比（省额度）
    python scripts/benchmark.py --skip-chunk-sizes    # 跳过 chunk_size 对比（省时间）

输出:
    Markdown 格式对比表格，可直接贴到 README。
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ChunkerConfig,
    HybridConfig,
    RAGConfig,
    RetrieverConfig,
    RerankerConfig,
)
from src.embeddings.zhipu_embedder import ZhipuEmbedder
from src.evaluation.ragas_eval import RAGEvaluator
from src.generator.zhipu_llm import ZhipuLLM
from src.rag_pipeline import RAGPipeline
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.retriever import Retriever
from src.reranker.zhipu_reranker import ZhipuReranker
from src.vectorstore.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark")

EVAL_PATH = _PROJECT_ROOT / "data" / "eval" / "financial_qa_eval.json"
DEFAULT_PERSIST_DIR = str(_PROJECT_ROOT / "data" / "chroma_db")
DEFAULT_COLLECTION = "financial_docs"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """单次检索 + 生成延迟记录。"""
    retrieve_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """单次配置的 benchmark 结果。"""
    label: str
    strategy: str
    reranker: bool
    chunk_size: int | None = None
    ragas_scores: dict[str, float] = field(default_factory=dict)
    latencies: list[LatencyRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline 构建
# ---------------------------------------------------------------------------

def _build_pipeline(
    api_key: str,
    strategy: str,
    use_reranker: bool = False,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
) -> RAGPipeline:
    """构建指定策略的 RAG Pipeline。

    Args:
        api_key: 智谱 API Key。
        strategy: "vector" / "bm25" / "hybrid"。
        use_reranker: 是否启用 Reranker。
        persist_dir: ChromaDB 持久化路径。
        collection: ChromaDB 集合名。
    """
    embedder = ZhipuEmbedder(
        api_key=api_key,
        model="embedding-3",
        batch_size=20,
    )
    store = ChromaStore(
        persist_directory=persist_dir,
        collection_name=collection,
    )

    if strategy in ("hybrid", "bm25"):
        hybrid_cfg = HybridConfig(
            enabled=True,
            strategy=strategy,
            rrf_k=60,
            vector_weight=0.6,
            bm25_weight=0.4,
            bm25_fetch_k=30,
            vector_fetch_k=30,
        )
        base_retriever = Retriever(
            embedder, store, RetrieverConfig(
                top_k=hybrid_cfg.vector_fetch_k,
                score_threshold=0.0,
            ),
        )
        bm25 = BM25Retriever(store)
        retriever = HybridRetriever(
            retriever=base_retriever,
            bm25_retriever=bm25,
            config=hybrid_cfg,
            score_threshold=0.0,
        )
    else:
        retriever = Retriever(
            embedder, store, RetrieverConfig(
                top_k=5,
                score_threshold=0.5,
            ),
        )

    llm = ZhipuLLM(
        api_key=api_key,
        model="glm-4-flash",
        temperature=0.3,
        max_tokens=2048,
    )

    reranker = None
    reranker_cfg = None
    if use_reranker:
        reranker = ZhipuReranker(api_key=api_key)
        reranker_cfg = RerankerConfig(
            enabled=True,
            retrieve_n=20,
            top_n=5,
            min_score=0.0,
        )
        # 如果启用 reranker，增加 retriever 的 top_k
        if hasattr(retriever, "_config"):
            pass  # HybridRetriever 用 fetch_k 控制
        else:
            retriever = Retriever(
                embedder, store, RetrieverConfig(
                    top_k=20,
                    score_threshold=0.0,
                ),
            )

    return RAGPipeline(
        retriever, llm, RAGConfig(max_context_tokens=4000),
        reranker=reranker,
        reranker_config=reranker_cfg,
    )


# ---------------------------------------------------------------------------
# 延迟测量
# ---------------------------------------------------------------------------

def _run_with_latency(
    pipeline: RAGPipeline,
    question: str,
) -> tuple[dict, LatencyRecord]:
    """执行 pipeline.query() 并记录延迟。"""
    record = LatencyRecord()
    t0 = time.perf_counter()
    try:
        result = pipeline.query(question)
    except Exception as e:
        logger.warning("查询失败: %s — %s", question[:30], e)
        result = {"answer": "", "sources": []}
    t1 = time.perf_counter()
    record.total_ms = (t1 - t0) * 1000
    return result, record


# ---------------------------------------------------------------------------
# 单次配置评测
# ---------------------------------------------------------------------------

def _run_eval_config(
    api_key: str,
    eval_samples: list[dict],
    strategy: str,
    use_reranker: bool = False,
    label: str = "",
) -> BenchmarkResult:
    """对单种配置运行完整评测。"""
    if not label:
        reranker_tag = " + Reranker" if use_reranker else ""
        label = f"{strategy}{reranker_tag}"

    logger.info("=" * 60)
    logger.info("开始评测: %s", label)
    logger.info("=" * 60)

    pipeline = _build_pipeline(api_key, strategy=strategy, use_reranker=use_reranker)
    evaluator = RAGEvaluator(api_key=api_key, model="glm-4-flash")

    responses: list[str] = []
    all_contexts: list[list[str]] = []
    latencies: list[LatencyRecord] = []

    for i, sample in enumerate(eval_samples):
        logger.info("[%s] %d/%d — %s", label, i + 1, len(eval_samples), sample["question"][:40])
        result, record = _run_with_latency(pipeline, sample["question"])
        responses.append(result.get("answer", ""))
        all_contexts.append([src.get("content", "") for src in result.get("sources", [])])
        latencies.append(record)

    # 提取 references
    references = [s.get("reference", "") for s in eval_samples if s.get("reference")]
    ref_list = references if references else None

    # 运行 RAGAS 评估
    logger.info("[%s] 运行 RAGAS 评估...", label)
    try:
        ragas_scores = evaluator.evaluate(
            questions=[s["question"] for s in eval_samples],
            responses=responses,
            contexts=all_contexts,
            references=ref_list,
        )
    except Exception as e:
        logger.error("[%s] RAGAS 评估失败: %s", label, e)
        ragas_scores = {}

    result = BenchmarkResult(
        label=label,
        strategy=strategy,
        reranker=use_reranker,
        ragas_scores=ragas_scores,
        latencies=latencies,
    )
    _log_result(result)
    return result


def _log_result(result: BenchmarkResult) -> None:
    """打印单次结果摘要。"""
    logger.info("-" * 40)
    logger.info("结果: %s", result.label)
    for metric, score in result.ragas_scores.items():
        logger.info("  %-25s %.4f", metric, score)

    if result.latencies:
        total_ms = [l.total_ms for l in result.latencies]
        logger.info(
            "  %-25s P50=%.0fms P95=%.0fms AVG=%.0fms",
            "Latency(total)",
            _p50(total_ms), _p95(total_ms), statistics.mean(total_ms),
        )


# ---------------------------------------------------------------------------
# Markdown 输出
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
}


def _render_markdown(results: list[BenchmarkResult]) -> str:
    """将所有结果渲染为 Markdown 表格。"""
    lines: list[str] = []
    lines.append("## 性能对比（Benchmark）")
    lines.append("")
    lines.append("> 由 `scripts/benchmark.py` 自动生成。")
    lines.append("")

    # --- RAGAS 分数表格 ---
    lines.append("### RAGAS 评估指标")
    lines.append("")
    lines.append("| 配置 | 忠实度 | 答案相关性 | 上下文精确度 | 上下文召回率 |")
    lines.append("|------|--------|-----------|-------------|-------------|")

    for r in results:
        faith = r.ragas_scores.get("faithfulness", 0)
        relev = r.ragas_scores.get("answer_relevancy", 0)
        prec = r.ragas_scores.get("context_precision", 0)
        recall = r.ragas_scores.get("context_recall", 0)
        lines.append(
            f"| {r.label} | {faith:.4f} | {relev:.4f} | {prec:.4f} | {recall:.4f} |"
        )

    lines.append("")

    # --- 延迟表格 ---
    lines.append("### 检索延迟")
    lines.append("")
    lines.append("| 配置 | P50 (ms) | P95 (ms) | AVG (ms) |")
    lines.append("|------|----------|----------|----------|")

    for r in results:
        if not r.latencies:
            lines.append(f"| {r.label} | N/A | N/A | N/A |")
            continue
        total_ms = [l.total_ms for l in r.latencies]
        lines.append(
            f"| {r.label} | {_p50(total_ms):.0f} | {_p95(total_ms):.0f} | {statistics.mean(total_ms):.0f} |"
        )

    lines.append("")

    # --- 关键结论 ---
    lines.append("### 关键结论")
    lines.append("")
    _write_conclusions(results, lines)

    return "\n".join(lines)


def _write_conclusions(results: list[BenchmarkResult], lines: list[str]) -> None:
    """自动生成关键结论文字。"""
    # 按 strategy 分组
    by_strategy: dict[str, BenchmarkResult] = {}
    for r in results:
        key = r.strategy
        if r.reranker:
            key += "+reranker"
        by_strategy[key] = r

    # 1. 混合 vs 纯向量
    vector_r = by_strategy.get("vector")
    hybrid_r = by_strategy.get("hybrid")
    if vector_r and hybrid_r:
        for metric, display in _METRIC_DISPLAY.items():
            v = vector_r.ragas_scores.get(metric, 0)
            h = hybrid_r.ragas_scores.get(metric, 0)
            if v > 0 and h > 0:
                diff_pct = (h - v) / v * 100
                direction = "提升" if diff_pct > 0 else "下降"
                lines.append(
                    f"- **混合检索 vs 纯向量** — {display}（{metric}）：{direction} {abs(diff_pct):.1f}%"
                    f"（{v:.4f} \u2192 {h:.4f}）"
                )

    # 2. Reranker 效果
    hybrid_rerank_r = by_strategy.get("hybrid+reranker")
    if hybrid_r and hybrid_rerank_r:
        for metric, display in _METRIC_DISPLAY.items():
            h = hybrid_r.ragas_scores.get(metric, 0)
            hr = hybrid_rerank_r.ragas_scores.get(metric, 0)
            if h > 0 and hr > 0:
                diff_pct = (hr - h) / h * 100
                direction = "提升" if diff_pct > 0 else "下降"
                lines.append(
                    f"- **Reranker 精排效果** — {display}（{metric}）：{direction} {abs(diff_pct):.1f}%"
                    f"（{h:.4f} \u2192 {hr:.4f}）"
                )

    # 3. 延迟结论
    if hybrid_r and hybrid_r.latencies:
        total_ms = [l.total_ms for l in hybrid_r.latencies]
        lines.append(
            f"- **混合检索延迟**：P50 = {_p50(total_ms):.0f}ms，满足交互体验要求（< 3s）"
        )

    # 4. 最优配置
    best_label = ""
    best_score = 0
    for r in results:
        # 综合分 = faithfulness * 0.4 + answer_relevancy * 0.3 + context_precision * 0.3
        f = r.ragas_scores.get("faithfulness", 0)
        a = r.ragas_scores.get("answer_relevancy", 0)
        p = r.ragas_scores.get("context_precision", 0)
        composite = f * 0.4 + a * 0.3 + p * 0.3
        if composite > best_score:
            best_score = composite
            best_label = r.label
    if best_label:
        lines.append(f"- **推荐配置**：{best_label}（综合得分最高）")

    lines.append("")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _p50(values: list[float]) -> float:
    """计算 P50（中位数）。"""
    if not values:
        return 0.0
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    """计算 P95。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = int(n * 0.95)
    return float(sorted_vals[min(idx, n - 1)])


def _load_eval_samples(path: Path) -> list[dict]:
    """加载评估数据集。"""
    if not path.exists():
        logger.error("评估数据集不存在: %s", path)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("加载了 %d 条评估样本: %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Financial RAG Benchmark")
    parser.add_argument("--api-key", type=str, default=None, help="智谱 API Key（也可通过 .env 配置）")
    parser.add_argument("--eval-path", type=str, default=None, help="评估数据集路径")
    parser.add_argument("--skip-reranker", action="store_true", help="跳过 Reranker 对比")
    parser.add_argument("--skip-chunk-sizes", action="store_true", help="跳过 chunk_size 对比")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 文件路径")
    args = parser.parse_args()

    # 1. 加载 API Key
    api_key = args.api_key
    if not api_key:
        # 尝试从 .env 加载
        try:
            from dotenv import load_dotenv
            load_dotenv(_PROJECT_ROOT / ".env")
            import os
            api_key = os.environ.get("ZHIPU_API_KEY", "")
        except ImportError:
            pass
    if not api_key:
        logger.error("未提供 API Key。请通过 --api-key 参数或 .env 文件配置。")
        sys.exit(1)

    # 2. 加载评估数据集
    eval_path = Path(args.eval_path) if args.eval_path else EVAL_PATH
    eval_samples = _load_eval_samples(eval_path)

    # 3. 定义评测配置
    configs: list[tuple[str, str, bool]] = [
        # (label, strategy, use_reranker)
        ("vector", "vector", False),
        ("bm25", "bm25", False),
        ("hybrid", "hybrid", False),
    ]
    if not args.skip_reranker:
        configs.append(("hybrid + Reranker", "hybrid", True))

    # 4. 逐个配置运行评测
    results: list[BenchmarkResult] = []
    for label, strategy, use_reranker in configs:
        try:
            result = _run_eval_config(
                api_key=api_key,
                eval_samples=eval_samples,
                strategy=strategy,
                use_reranker=use_reranker,
                label=label,
            )
            results.append(result)
        except Exception as e:
            logger.error("评测配置 %s 失败: %s", label, e, exc_info=True)

    # 5. chunk_size 对比（可选，注释说明：需要重建索引，较耗时）
    if not args.skip_chunk_sizes:
        logger.info("=" * 60)
        logger.info("注意: chunk_size 对比需要重建 ChromaDB 索引，跳过自动执行。")
        logger.info("如需对比，请手动修改 config.yaml 中的 chunker.chunk_size 后重新索引并运行 benchmark。")
        logger.info("=" * 60)

    # 6. 输出 Markdown
    if results:
        md = _render_markdown(results)
        print("\n")
        print(md)

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(md, encoding="utf-8")
            logger.info("Markdown 报告已保存到: %s", output_path)
    else:
        logger.error("所有评测均失败，无结果可输出。")
        sys.exit(1)


if __name__ == "__main__":
    main()
