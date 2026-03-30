import threading

import pytest

from src.metrics.collector import MetricsCollector, QueryMetrics


@pytest.fixture(autouse=True)
def _reset_singleton():
    MetricsCollector.reset_singleton()
    yield
    MetricsCollector.reset_singleton()


def _make_metrics(**overrides) -> QueryMetrics:
    defaults = dict(
        question="test query",
        retrieve_ms=100.0,
        rerank_ms=50.0,
        generate_ms=200.0,
        total_ms=350.0,
        input_tokens=100,
        output_tokens=50,
        cache_hit=False,
        strategy="hybrid",
        reranker_enabled=True,
        num_sources=5,
    )
    defaults.update(overrides)
    return QueryMetrics(**defaults)


class TestMetricsCollectorSingleton:
    def test_singleton_returns_same_instance(self):
        a = MetricsCollector()
        b = MetricsCollector()
        assert a is b

    def test_reset_singleton_creates_new_instance(self):
        a = MetricsCollector()
        MetricsCollector.reset_singleton()
        b = MetricsCollector()
        assert a is not b


class TestRecord:
    def test_record_stores_metrics(self):
        collector = MetricsCollector()
        m = _make_metrics(question="hello")
        collector.record(m)
        summary = collector.summary()
        assert summary["total_queries"] == 1
        assert summary["recent_queries"][0]["question"] == "hello"

    def test_record_multiple(self):
        collector = MetricsCollector()
        for i in range(5):
            collector.record(_make_metrics(question=f"q{i}", total_ms=100.0 + i * 10))
        summary = collector.summary()
        assert summary["total_queries"] == 5
        assert summary["avg_latency_ms"] == 120.0


class TestSummary:
    def test_empty_summary(self):
        collector = MetricsCollector()
        s = collector.summary()
        assert s["total_queries"] == 0
        assert s["avg_latency_ms"] == 0.0
        assert s["p50_latency_ms"] == 0.0
        assert s["p95_latency_ms"] == 0.0
        assert s["cache_hit_rate"] == 0.0
        assert s["strategy_distribution"] == {}
        assert s["recent_queries"] == []

    def test_summary_last_n(self):
        collector = MetricsCollector()
        for i in range(10):
            collector.record(_make_metrics(question=f"q{i}", total_ms=float(i)))
        s = collector.summary(last_n=3)
        assert s["total_queries"] == 3
        assert s["avg_latency_ms"] == 8.0

    def test_summary_percentiles(self):
        collector = MetricsCollector()
        for i in range(100):
            collector.record(_make_metrics(total_ms=float(i + 1)))
        s = collector.summary()
        assert s["p50_latency_ms"] == pytest.approx(50.5, abs=0.1)
        assert s["p95_latency_ms"] == pytest.approx(95.05, abs=0.1)

    def test_cache_hit_rate(self):
        collector = MetricsCollector()
        collector.record(_make_metrics(cache_hit=True))
        collector.record(_make_metrics(cache_hit=True))
        collector.record(_make_metrics(cache_hit=False))
        s = collector.summary()
        assert s["cache_hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_strategy_distribution(self):
        collector = MetricsCollector()
        collector.record(_make_metrics(strategy="vector"))
        collector.record(_make_metrics(strategy="hybrid"))
        collector.record(_make_metrics(strategy="hybrid"))
        s = collector.summary()
        assert s["strategy_distribution"] == {"vector": 1, "hybrid": 2}

    def test_token_totals(self):
        collector = MetricsCollector()
        collector.record(_make_metrics(input_tokens=100, output_tokens=50))
        collector.record(_make_metrics(input_tokens=200, output_tokens=100))
        s = collector.summary()
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 150
        assert s["avg_input_tokens"] == 150.0
        assert s["avg_output_tokens"] == 75.0

    def test_recent_queries_limit(self):
        collector = MetricsCollector()
        for i in range(25):
            collector.record(_make_metrics(question=f"q{i}"))
        s = collector.summary()
        assert len(s["recent_queries"]) == 20


class TestClear:
    def test_clear_empties_all(self):
        collector = MetricsCollector()
        collector.record(_make_metrics())
        collector.record(_make_metrics())
        assert collector.summary()["total_queries"] == 2
        collector.clear()
        assert collector.summary()["total_queries"] == 0


class TestMaxRecords:
    def test_max_records_eviction(self):
        collector = MetricsCollector()
        collector._max_records = 10
        for i in range(15):
            collector.record(_make_metrics(question=f"q{i}"))
        s = collector.summary()
        assert s["total_queries"] == 10
        assert s["recent_queries"][0]["question"] == "q5"


class TestThreadSafety:
    def test_concurrent_record(self):
        collector = MetricsCollector()
        errors = []

        def _worker(start: int):
            try:
                for i in range(100):
                    collector.record(_make_metrics(question=f"t{start}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        s = collector.summary()
        assert s["total_queries"] == 500
