from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch


def _install_fake_modules() -> dict[str, types.ModuleType]:
    """Inject fake modules for optional dependencies into sys.modules.

    Returns a dict of the fake modules for cleanup.
    """
    # --- fake ragas ---
    fake_ragas = types.ModuleType("ragas")
    fake_ragas_metrics = types.ModuleType("ragas.metrics")

    fake_ragas_metrics.faithfulness = MagicMock()
    fake_ragas_metrics.answer_relevancy = MagicMock()
    fake_ragas_metrics.context_precision = MagicMock()
    fake_ragas_metrics.context_recall = MagicMock()

    sys.modules["ragas"] = fake_ragas
    sys.modules["ragas.metrics"] = fake_ragas_metrics

    # --- fake datasets ---
    fake_datasets = types.ModuleType("datasets")
    fake_dataset_cls = MagicMock(name="Dataset")
    fake_datasets.Dataset = fake_dataset_cls

    sys.modules["datasets"] = fake_datasets

    # --- fake langchain_openai ---
    fake_lc = types.ModuleType("langchain_openai")
    fake_lc.ChatOpenAI = MagicMock()
    fake_lc.OpenAIEmbeddings = MagicMock()

    sys.modules["langchain_openai"] = fake_lc

    return {"ragas": fake_ragas, "ragas.metrics": fake_ragas_metrics,
            "datasets": fake_datasets, "langchain_openai": fake_lc}


def _uninstall_fake_modules(modules: dict[str, types.ModuleType]) -> None:
    """Remove fake modules from sys.modules, restoring originals if they existed."""
    for name in modules:
        sys.modules.pop(name, None)


class TestRAGEvaluator:
    def test_evaluate_returns_scores(self):
        """Mock 全部可选依赖，验证 evaluate() 正确返回分数。"""
        fakes = _install_fake_modules()

        try:
            # Patch ragas.evaluate to return known scores
            fake_evaluate_fn = MagicMock(return_value={
                "faithfulness": 0.95,
                "answer_relevancy": 0.87,
                "context_precision": 0.82,
            })
            sys.modules["ragas"].evaluate = fake_evaluate_fn

            from src.evaluation.ragas_eval import RAGEvaluator
            evaluator = RAGEvaluator(api_key="test_key")
            scores = evaluator.evaluate(
                questions=["什么是GDP？"],
                responses=["GDP是国内生产总值..."],
                contexts=[["GDP是衡量经济总量的指标"]],
            )
            assert scores["faithfulness"] == 0.95
            assert scores["answer_relevancy"] == 0.87
            assert scores["context_precision"] == 0.82
        finally:
            _uninstall_fake_modules(fakes)

    def test_evaluate_with_references_includes_recall(self):
        """有 references 时，context_recall 应回现在结果中。"""
        fakes = _install_fake_modules()

        try:
            fake_evaluate_fn = MagicMock(return_value={
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.75,
            })
            sys.modules["ragas"].evaluate = fake_evaluate_fn

            from src.evaluation.ragas_eval import RAGEvaluator
            evaluator = RAGEvaluator(api_key="test_key")
            scores = evaluator.evaluate(
                questions=["什么是GDP？"],
                responses=["GDP是国内生产总值..."],
                contexts=[["GDP是衡量经济总量的指标"]],
                references=["国内生产总值（GDP）是..."],
            )
            assert "context_recall" in scores
        finally:
            _uninstall_fake_modules(fakes)

    def test_evaluate_pipeline_calls_evaluate(self):
        """evaluate_pipeline 应委托给 evaluate() 方法。"""
        with patch("src.evaluation.ragas_eval.RAGEvaluator.evaluate") as mock_eval:
            mock_eval.return_value = {"faithfulness": 0.9}
            mock_pipeline = MagicMock()
            mock_pipeline.query.return_value = {
                "answer": "test answer",
                "sources": [{"content": "context1"}],
            }

            from src.evaluation.ragas_eval import RAGEvaluator
            evaluator = RAGEvaluator(api_key="test_key")
            scores = evaluator.evaluate_pipeline(
                mock_pipeline,
                [{"question": "什么是GDP？", "reference": "GDP是..."}],
            )
            assert scores["faithfulness"] == 0.9
            mock_pipeline.query.assert_called_once()
