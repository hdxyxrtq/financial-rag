"""Metrics 收集器 — 线程安全的单例，记录查询延迟、Token 消耗、缓存命中率等。"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """单次查询的度量数据。"""

    question: str
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    strategy: str = "vector"
    reranker_enabled: bool = False
    num_sources: int = 0


class MetricsCollector:
    """线程安全的 Metrics 收集器（单例）。

    使用方式::

        collector = MetricsCollector()
        collector.record(QueryMetrics(...))
        summary = collector.summary(last_n=20)
    """

    _instance: MetricsCollector | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricsCollector:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._queries: list[QueryMetrics] = []
        self._max_records = 1000
        self._data_lock = threading.Lock()
        self._initialized = True

    def record(self, metrics: QueryMetrics) -> None:
        """记录一次查询的度量数据。"""
        with self._data_lock:
            self._queries.append(metrics)
            if len(self._queries) > self._max_records:
                self._queries = self._queries[-self._max_records :]

    def summary(self, last_n: int | None = None) -> dict:
        """返回汇总统计。

        Args:
            last_n: 只统计最近 N 条记录。None 表示全部。

        Returns:
            包含 total_queries, avg_latency_ms, p50/p95, cache_hit_rate 等字段的字典。
        """
        with self._data_lock:
            queries = list(self._queries)

        if last_n is not None and last_n > 0:
            queries = queries[-last_n:]

        if not queries:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "cache_hit_rate": 0.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "strategy_distribution": {},
                "recent_queries": [],
            }

        total = len(queries)
        latencies = sorted(q.total_ms for q in queries)
        cache_hits = sum(1 for q in queries if q.cache_hit)

        total_input = sum(q.input_tokens for q in queries)
        total_output = sum(q.output_tokens for q in queries)

        strategy_dist: dict[str, int] = defaultdict(int)
        for q in queries:
            strategy_dist[q.strategy] += 1

        recent = queries[-20:]

        def _percentile(sorted_vals: list[float], pct: float) -> float:
            if not sorted_vals:
                return 0.0
            n = len(sorted_vals)
            k = (pct / 100) * (n - 1)
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_vals[-1]
            return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

        return {
            "total_queries": total,
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p50_latency_ms": round(_percentile(latencies, 50), 2),
            "p95_latency_ms": round(_percentile(latencies, 95), 2),
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "avg_input_tokens": round(total_input / total, 1),
            "avg_output_tokens": round(total_output / total, 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "strategy_distribution": dict(strategy_dist),
            "recent_queries": [
                {
                    "question": q.question,
                    "total_ms": round(q.total_ms, 2),
                    "retrieve_ms": round(q.retrieve_ms, 2),
                    "rerank_ms": round(q.rerank_ms, 2),
                    "generate_ms": round(q.generate_ms, 2),
                    "cache_hit": q.cache_hit,
                    "strategy": q.strategy,
                    "num_sources": q.num_sources,
                }
                for q in recent
            ],
        }

    def clear(self) -> None:
        """清空所有记录。"""
        with self._data_lock:
            self._queries.clear()

    @classmethod
    def reset_singleton(cls) -> None:
        """重置单例（仅用于测试）。"""
        with cls._lock:
            cls._instance = None
