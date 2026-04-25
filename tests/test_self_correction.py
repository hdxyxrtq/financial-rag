"""Tests for the self-correction module (Phase 1-3)."""

from unittest.mock import MagicMock

import pytest

from src.correction.retrieval_gate import RetrievalGate
from src.correction.rule_checker import RuleChecker
from src.correction.types import RetrievalQuality
from src.retriever.retriever import RetrievalResult


def _make_result(content: str = "test", score: float = 0.5, doc_id: str = "d1") -> RetrievalResult:
    return RetrievalResult(content=content, score=score, metadata={}, doc_id=doc_id)


class TestModuleImports:
    """Verify all correction modules can be imported without errors."""

    def test_import_types(self):
        from src.correction.types import ClaimVerdict, CorrectionResult, RetrievalQuality

        assert ClaimVerdict is not None
        assert CorrectionResult is not None
        assert RetrievalQuality is not None

    def test_import_retrieval_gate(self):
        from src.correction.retrieval_gate import RetrievalGate

        assert RetrievalGate is not None

    def test_import_rule_checker(self):
        from src.correction.rule_checker import RuleChecker

        assert RuleChecker is not None

    def test_import_nli_verifier(self):
        from src.correction.nli_verifier import NLIVerifier

        assert NLIVerifier is not None

    def test_import_external_verifier(self):
        from src.correction.external_verifier import ExternalVerifier

        assert ExternalVerifier is not None

    def test_import_pipeline(self):
        from src.correction.pipeline import SelfCorrectingPipeline

        assert SelfCorrectingPipeline is not None

    def test_import_package(self):
        from src.correction import SelfCorrectingPipeline

        assert SelfCorrectingPipeline is not None


class TestSelfCorrectionConfig:
    """Verify Config loads self_correction settings."""

    def test_config_has_self_correction(self):
        from src.config import Config

        config = Config()
        sc = config.self_correction
        assert sc is not None
        assert sc.enabled is False
        assert sc.max_retries == 2
        assert sc.rerank_threshold_good == 0.7
        assert sc.rerank_threshold_weak == 0.3

    def test_config_siliconflow_api_key(self):
        from src.config import Config

        config = Config()
        key = config.siliconflow_api_key
        assert key is None or isinstance(key, str)


class TestSelfCorrectingPipeline:
    """Verify SelfCorrectingPipeline delegates to inner pipeline."""

    def test_query_delegates(self):
        from src.config import Config, SelfCorrectionConfig
        from src.correction.pipeline import SelfCorrectingPipeline

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "test answer", "sources": []}

        config = Config()
        sc_config = config.self_correction
        wrapper = SelfCorrectingPipeline(pipeline=mock_pipeline, config=sc_config)

        result = wrapper.query("test question")
        mock_pipeline.query.assert_called_once_with("test question", chat_history=None)
        assert result["answer"] == "test answer"
        assert result["sources"] == []
        assert "correction" in result

    def test_query_with_history(self):
        from src.config import Config
        from src.correction.pipeline import SelfCorrectingPipeline

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {"answer": "test", "sources": []}

        config = Config()
        sc_config = config.self_correction
        wrapper = SelfCorrectingPipeline(pipeline=mock_pipeline, config=sc_config)

        history = [{"role": "user", "content": "hello"}]
        wrapper.query("test", chat_history=history)
        mock_pipeline.query.assert_called_once_with("test", chat_history=history)

    def test_stream_query_delegates(self):
        from src.config import Config
        from src.correction.pipeline import SelfCorrectingPipeline

        mock_pipeline = MagicMock()
        mock_pipeline.stream_query.return_value = iter([
            {"type": "sources", "sources": []},
            {"type": "answer", "content": "test"},
        ])

        config = Config()
        sc_config = config.self_correction
        wrapper = SelfCorrectingPipeline(pipeline=mock_pipeline, config=sc_config)

        results = list(wrapper.stream_query("test"))
        mock_pipeline.stream_query.assert_called_once_with("test", chat_history=None)
        assert len(results) == 2


class TestDataclassCreation:
    """Verify dataclasses can be instantiated."""

    def test_retrieval_quality_creation(self):
        from src.correction.types import RetrievalQuality

        rq = RetrievalQuality(level="GOOD", top_score=0.9, avg_score=0.7, num_sources=5)
        assert rq.level == "GOOD"
        assert rq.top_score == 0.9

    def test_claim_verdict_creation(self):
        from src.correction.types import ClaimVerdict

        cv = ClaimVerdict(claim="test", supported=True, evidence="ev", source="src")
        assert cv.supported is True

    def test_correction_result_creation(self):
        from src.correction.types import CorrectionResult

        cr = CorrectionResult(passed=True)
        assert cr.passed is True
        assert cr.flagged_claims == []
        assert cr.layer_results == {}
        assert cr.confidence == 0.0


class TestExternalVerifier:
    """Layer 4: External model verification tests (mocked httpx)."""

    def test_verify_supported(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"claim": "营收5000万元", "supported": true, "evidence_quote": "公司营收5000万元"}]'}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(["营收5000万元"], "公司营收5000万元")

        assert len(results) == 1
        assert results[0].supported is True
        assert results[0].source == "external_verifier"

    def test_verify_unsupported(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"claim": "量子计算突破", "supported": false, "evidence_quote": "UNSUPPORTED"}]'}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(["量子计算突破"], "公司营收5000万元")

        assert len(results) == 1
        assert results[0].supported is False

    def test_verify_api_failure(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        with patch("src.correction.external_verifier.httpx.post", side_effect=Exception("API error")):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(["测试断言"], "context")

        assert len(results) == 1
        assert results[0].supported is True
        assert "degraded" in results[0].evidence

    def test_verify_mixed_results(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"claim": "营收5000万元", "supported": true, "evidence_quote": "营收5000万"}, {"claim": "量子计算突破", "supported": false, "evidence_quote": "UNSUPPORTED"}]'}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(
                ["营收5000万元", "量子计算突破"],
                "公司营收5000万元",
            )

        assert len(results) == 2
        supported = [r for r in results if r.supported]
        unsupported = [r for r in results if not r.supported]
        assert len(supported) == 1
        assert len(unsupported) == 1

    def test_verify_empty_claims(self):
        from src.correction.external_verifier import ExternalVerifier

        verifier = ExternalVerifier(api_key="test-key")
        results = verifier.verify([], "context")
        assert results == []

    def test_verify_malformed_json(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is not JSON"}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(["测试断言"], "context")

        assert len(results) == 1
        assert results[0].supported is True

    def test_verify_markdown_wrapped_json(self):
        from unittest.mock import patch

        from src.correction.external_verifier import ExternalVerifier

        json_content = '[{"claim": "营收5000万元", "supported": true, "evidence_quote": "营收5000万"}]'
        wrapped = f"```json\n{json_content}\n```"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": wrapped}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            verifier = ExternalVerifier(api_key="test-key")
            results = verifier.verify(["营收5000万元"], "context")

        assert len(results) == 1
        assert results[0].supported is True


class TestClaimDecomposition:
    """Layer 3: Claim decomposition tests."""

    def test_decompose_single_claim(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        result = verifier.decompose_claims("公司2024年营收达到5000万元。")
        assert len(result) == 1
        assert "5000万元" in result[0]

    def test_decompose_multiple_claims(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        answer = "公司营收为5000万元人民币。净利润达到1000万元人民币。同比增长率达到20%以上。"
        result = verifier.decompose_claims(answer)
        assert len(result) == 3

    def test_decompose_filters_short(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        answer = "公司业绩很好。该公司2024年营收达到5000万元人民币。"
        result = verifier.decompose_claims(answer)
        assert len(result) == 1
        assert "5000万元" in result[0]

    def test_decompose_filters_boilerplate(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        answer = "公司营收为5000万元。根据现有资料整理。以上分析仅供参考。"
        result = verifier.decompose_claims(answer)
        assert len(result) == 1
        assert "5000万元" in result[0]

    def test_decompose_preserves_financial_terms(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        answer = "沪深300指数的市盈率为15.3倍。该基金的ROE达到18.5%的水平。"
        result = verifier.decompose_claims(answer)
        assert len(result) == 2
        assert any("市盈率" in c for c in result)
        assert any("ROE" in c for c in result)


class TestNLIVerification:
    """Layer 3: NLI verification tests."""

    def test_verify_supported_claim(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        context = "公司2024年营收达到5000万元，净利润为1000万元，同比增长20%。"
        claims = ["公司2024年营收达到5000万元"]
        results = verifier.verify(claims, context)
        assert len(results) == 1
        assert results[0].supported is True
        assert results[0].claim == claims[0]

    def test_verify_unsupported_claim(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        context = "公司2024年营收达到3000万元，净利润为500万元。"
        claims = ["量子计算在药物研发领域取得重大突破性进展"]
        results = verifier.verify(claims, context)
        assert len(results) == 1
        assert results[0].supported is False

    def test_verify_mixed_claims(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        context = "公司2024年营收达到5000万元，净利润为1000万元。"
        claims = [
            "公司2024年营收达到5000万元",
            "量子计算在药物研发领域取得重大突破性进展",
        ]
        results = verifier.verify(claims, context)
        assert len(results) == 2
        supported = [r for r in results if r.supported]
        unsupported = [r for r in results if not r.supported]
        assert len(supported) >= 1
        assert len(unsupported) >= 1

    def test_verify_empty_claims(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        results = verifier.verify([], "some context")
        assert results == []

    def test_verify_exact_quote(self):
        from src.correction.nli_verifier import NLIVerifier

        verifier = NLIVerifier()
        context = "该公司2024年第三季度实现营业收入50亿元人民币"
        claims = ["该公司2024年第三季度实现营业收入50亿元人民币"]
        results = verifier.verify(claims, context)
        assert len(results) == 1
        assert results[0].supported is True


class TestRetrievalGate:
    """Layer 0: Retrieval quality gate."""

    def test_gate_good_retrieval(self):
        gate = RetrievalGate()
        results = [_make_result(score=0.85)]
        quality = gate.assess(results)
        assert quality.level == "GOOD"
        assert quality.top_score == 0.85

    def test_gate_weak_retrieval(self):
        gate = RetrievalGate()
        results = [_make_result(score=0.15)]
        quality = gate.assess(results)
        assert quality.level == "WEAK"
        assert quality.top_score == 0.15

    def test_gate_marginal_retrieval(self):
        gate = RetrievalGate()
        results = [_make_result(score=0.45)]
        quality = gate.assess(results)
        assert quality.level == "MARGINAL"
        assert quality.top_score == 0.45

    def test_gate_no_rerank_fallback(self):
        gate = RetrievalGate()
        results = [
            _make_result(score=0.75, doc_id="d1"),
            _make_result(score=0.60, doc_id="d2"),
        ]
        reranked = [_make_result(score=0.90, doc_id="d2")]
        quality = gate.assess(results, reranked_results=reranked)
        assert quality.level == "GOOD"
        assert quality.top_score == 0.90
        assert quality.avg_score == 0.90

    def test_gate_no_rerank_uses_retrieval_score(self):
        gate = RetrievalGate()
        results = [_make_result(score=0.80)]
        quality = gate.assess(results, reranked_results=None)
        assert quality.level == "GOOD"
        assert quality.top_score == 0.80

    def test_gate_empty_results(self):
        gate = RetrievalGate()
        quality = gate.assess([])
        assert quality.level == "WEAK"
        assert quality.num_sources == 0

    def test_gate_avg_score_multiple_results(self):
        gate = RetrievalGate()
        results = [
            _make_result(score=0.80, doc_id="d1"),
            _make_result(score=0.60, doc_id="d2"),
        ]
        quality = gate.assess(results)
        assert quality.avg_score == pytest.approx(0.70)
        assert quality.num_sources == 2

    def test_gate_custom_thresholds(self):
        gate = RetrievalGate(rerank_threshold_good=0.9, rerank_threshold_weak=0.5)
        results = [_make_result(score=0.85)]
        quality = gate.assess(results)
        assert quality.level == "MARGINAL"


class TestRuleChecker:
    """Layer 2: Rule-based pre-check."""

    def test_number_in_source(self):
        checker = RuleChecker()
        source = "公司2024年营收达到5000万元"
        answer = "营收为5000万元"
        assert checker.check(answer, [source]) == []

    def test_number_not_in_source(self):
        checker = RuleChecker()
        source = "公司营收为3000万元"
        answer = "营收达到5000万元"
        issues = checker.check(answer, [source])
        number_issues = [i for i in issues if i["type"] == "number"]
        assert len(number_issues) >= 1
        assert number_issues[0]["severity"] == "HIGH"
        assert "5000" in str(number_issues[0]["value"])

    def test_entity_in_source(self):
        checker = RuleChecker(financial_terms=["市盈率", "ROE"])
        source = "该公司的市盈率为15倍，ROE为18%"
        answer = "市盈率是15倍，ROE为18%"
        entity_issues = [i for i in checker.check(answer, [source]) if i["type"] == "entity"]
        assert entity_issues == []

    def test_entity_not_in_source(self):
        checker = RuleChecker(financial_terms=["市盈率", "ROE"])
        source = "公司经营稳健"
        answer = "该公司的市盈率为15倍"
        issues = checker.check(answer, [source])
        entity_issues = [i for i in issues if i["type"] == "entity"]
        assert len(entity_issues) == 1
        assert entity_issues[0]["value"] == "市盈率"

    def test_date_in_source(self):
        checker = RuleChecker()
        source = "2024年3月15日发布财报"
        answer = "2024年3月15日发布了财报"
        date_issues = [i for i in checker.check(answer, [source]) if i["type"] == "date"]
        assert date_issues == []

    def test_date_not_in_source(self):
        checker = RuleChecker()
        source = "公司于2023年发布了年报"
        answer = "2025年公司业绩大幅增长"
        issues = checker.check(answer, [source])
        date_issues = [i for i in issues if i["type"] == "date"]
        assert len(date_issues) >= 1
        assert date_issues[0]["severity"] == "HIGH"

    def test_low_overlap(self):
        checker = RuleChecker()
        source = "公司经营稳健，营业收入稳步增长"
        answer = "量子计算在药物研发领域取得重大突破XYZ"
        issues = checker.check(answer, [source])
        overlap_issues = [i for i in issues if i["type"] == "overlap"]
        assert len(overlap_issues) == 1
        assert overlap_issues[0]["severity"] == "MEDIUM"

    def test_clean_answer_no_issues(self):
        checker = RuleChecker(financial_terms=["营收"])
        source = "公司2024年营收为5000万元，同比增长20%"
        answer = "2024年营收为5000万元，增长20%"
        assert checker.check(answer, [source]) == []

    def test_multiple_sources_combined(self):
        checker = RuleChecker()
        sources = ["营收3000万元", "利润1000万元"]
        answer = "营收为3000万元，利润为2000万元"
        issues = checker.check(answer, sources)
        number_issues = [i for i in issues if i["type"] == "number"]
        assert any("2000" in str(i["value"]) for i in number_issues)

    def test_default_financial_terms_loaded(self):
        checker = RuleChecker()
        assert len(checker._financial_terms) > 0


def _make_pipeline_result(answer: str, sources: list[dict] | None = None) -> dict:
    """Build a mock RAGPipeline.query() return value."""
    if sources is None:
        sources = [{"content": "公司2024年营收5000万元，净利润1000万元。", "score": 0.8, "metadata": {}}]
    return {"answer": answer, "sources": sources}


class TestSelfCorrectingPipelineOrchestration:
    """SelfCorrectingPipeline full orchestration tests."""

    def _make_wrapper(self, mock_pipeline, api_key: str | None = None):
        from src.config import SelfCorrectionConfig
        from src.correction.pipeline import SelfCorrectingPipeline

        config = SelfCorrectionConfig(
            enabled=True, max_retries=2,
            rerank_threshold_good=0.7, rerank_threshold_weak=0.3,
        )
        return SelfCorrectingPipeline(
            pipeline=mock_pipeline, config=config, api_key=api_key,
        )

    def test_query_passes_through(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_pipeline_result(
            "公司2024年营收为5000万元，净利润达到1000万元。",
        )

        wrapper = self._make_wrapper(mock_pipeline)
        result = wrapper.query("公司业绩如何")

        assert result["answer"] == "公司2024年营收为5000万元，净利润达到1000万元。"
        assert "correction" in result
        assert result["correction"].passed is True

    def test_query_weak_retrieval(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_pipeline_result(
            "回答",
            [{"content": "weak", "score": 0.1, "metadata": {}}],
        )

        wrapper = self._make_wrapper(mock_pipeline)
        result = wrapper.query("问题")

        assert "correction" in result
        assert result["correction"].confidence == pytest.approx(0.2)

    def test_query_rule_check_catches_issues(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_pipeline_result(
            "量子计算在药物研发领域取得重大突破性进展XYZ",
        )

        wrapper = self._make_wrapper(mock_pipeline)
        result = wrapper.query("量子计算")

        assert "correction" in result

    def test_query_correction_retry(self):
        from unittest.mock import patch

        mock_pipeline = MagicMock()
        first_result = _make_pipeline_result(
            "量子计算在药物研发领域取得重大突破XYZ进展",
        )
        fixed_result = _make_pipeline_result(
            "公司2024年营收为5000万元，净利润达到1000万元。",
        )
        mock_pipeline.query.side_effect = [first_result, fixed_result]

        wrapper = self._make_wrapper(mock_pipeline)
        result = wrapper.query("公司业绩如何")

        assert mock_pipeline.query.call_count == 2
        assert "correction" in result

    def test_query_max_retries_exhausted(self):
        mock_pipeline = MagicMock()
        bad_result = _make_pipeline_result(
            "量子计算在药物研发领域取得重大突破XYZ进展",
        )
        mock_pipeline.query.return_value = bad_result

        wrapper = self._make_wrapper(mock_pipeline)
        result = wrapper.query("量子计算")

        assert mock_pipeline.query.call_count == 3
        assert "correction" in result
        assert result["correction"].passed is False

    def test_stream_query_delegates(self):
        mock_pipeline = MagicMock()
        mock_pipeline.stream_query.return_value = iter([
            {"type": "sources", "sources": []},
            {"type": "answer", "content": "test"},
        ])

        wrapper = self._make_wrapper(mock_pipeline)
        results = list(wrapper.stream_query("test"))

        mock_pipeline.stream_query.assert_called_once_with("test", chat_history=None)
        assert len(results) == 2

    def test_query_with_external_verifier(self):
        from unittest.mock import patch

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_pipeline_result(
            "量子计算在药物研发领域取得重大突破XYZ进展",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '[{"claim": "量子计算突破", "supported": true, "evidence_quote": "已核实"}]'}}],
        }

        with patch("src.correction.external_verifier.httpx.post", return_value=mock_response):
            wrapper = self._make_wrapper(mock_pipeline, api_key="test-key")
            result = wrapper.query("量子计算")

        assert "correction" in result
