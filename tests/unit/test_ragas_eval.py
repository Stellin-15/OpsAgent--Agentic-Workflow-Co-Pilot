"""Unit tests for RAGAS evaluation (heuristic fallback path)."""

import pytest

from opsagent.ai.evaluation.ragas_eval import RagasEvaluator, RagasScores


class TestRagasScores:
    def test_confidence_is_mean(self) -> None:
        scores = RagasScores(
            context_precision=0.8,
            faithfulness=0.6,
            answer_relevance=0.7,
        )
        assert abs(scores.confidence - 0.7) < 0.001

    def test_low_confidence_flag(self) -> None:
        low = RagasScores(0.3, 0.4, 0.5)
        assert low.is_low_confidence is True

    def test_high_confidence_flag(self) -> None:
        high = RagasScores(0.8, 0.9, 0.7)
        assert high.is_low_confidence is False

    def test_to_dict_contains_required_keys(self) -> None:
        scores = RagasScores(0.7, 0.8, 0.6)
        d = scores.to_dict()
        assert "context_precision" in d
        assert "faithfulness" in d
        assert "answer_relevance" in d
        assert "confidence" in d
        assert "low_confidence" in d


class TestRagasEvaluatorHeuristic:
    @pytest.mark.asyncio
    async def test_heuristic_returns_scores(self) -> None:
        evaluator = RagasEvaluator()
        # Force heuristic path
        evaluator._ragas_available = False

        scores = await evaluator.evaluate(
            question="How do I fix high CPU?",
            answer="1. Check the top process\n2. Kill the offender\n3. Restart service",
            contexts=["High CPU usage runbook: check top, kill process, restart service"],
        )

        assert 0.0 <= scores.confidence <= 1.0
        assert 0.0 <= scores.context_precision <= 1.0
        assert 0.0 <= scores.faithfulness <= 1.0
        assert 0.0 <= scores.answer_relevance <= 1.0

    @pytest.mark.asyncio
    async def test_empty_context_does_not_crash(self) -> None:
        evaluator = RagasEvaluator()
        evaluator._ragas_available = False

        scores = await evaluator.evaluate(
            question="What is wrong?",
            answer="I don't know",
            contexts=[],
        )
        assert scores is not None
        assert isinstance(scores.confidence, float)
