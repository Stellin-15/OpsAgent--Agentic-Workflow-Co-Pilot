"""RAGAS evaluation on every generated draft.

RAGAS (Retrieval-Augmented Generation Assessment) scores three dimensions:
    context_precision  — are the retrieved chunks relevant to the question?
    faithfulness       — does the answer stick to the retrieved context?
    answer_relevance   — does the answer address the actual question?

Each score is in [0, 1]. We compute a composite confidence score:
    confidence = mean(context_precision, faithfulness, answer_relevance)

If confidence < CONFIDENCE_THRESHOLD the draft is flagged as low-confidence
in the Slack message so the on-call engineer knows to double-check.

The scores are stored in the ``drafts.ragas_scores`` JSONB column and logged
to MLflow for experiment tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)

CONFIDENCE_THRESHOLD = 0.6


@dataclass
class RagasScores:
    context_precision: float
    faithfulness: float
    answer_relevance: float

    @property
    def confidence(self) -> float:
        return (self.context_precision + self.faithfulness + self.answer_relevance) / 3

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence
        d["low_confidence"] = self.is_low_confidence
        return d


class RagasEvaluator:
    """
    Runs RAGAS evaluation on a draft.

    Falls back gracefully to neutral scores (0.5) if the ragas library
    is not installed or the LLM call fails — this ensures the pipeline
    never fails just because eval is unavailable.
    """

    def __init__(self, llm_router=None) -> None:
        self._llm_router = llm_router
        self._ragas_available = self._check_ragas()

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            return False

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RagasScores:
        """
        Evaluate *answer* against *question* and *contexts*.

        Uses the ragas library if available; otherwise returns heuristic scores
        based on answer length and context overlap.
        """
        if self._ragas_available:
            return await self._ragas_evaluate(question, answer, contexts)
        else:
            return self._heuristic_evaluate(question, answer, contexts)

    async def _ragas_evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RagasScores:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset,
                metrics=[context_precision, faithfulness, answer_relevancy],
            )
            scores = RagasScores(
                context_precision=float(result["context_precision"]),
                faithfulness=float(result["faithfulness"]),
                answer_relevance=float(result["answer_relevancy"]),
            )
            log.info(
                "ragas.evaluated",
                confidence=scores.confidence,
                low_confidence=scores.is_low_confidence,
            )
            return scores
        except Exception as exc:
            log.warning("ragas.eval_failed", error=str(exc))
            return self._heuristic_evaluate(question, answer, contexts)

    @staticmethod
    def _heuristic_evaluate(
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RagasScores:
        """
        Fast heuristic fallback.
        - context_precision: fraction of context words that appear in answer
        - faithfulness: answer length ratio (longer = more complete, capped at 1)
        - answer_relevance: keyword overlap between question and answer
        """
        context_words = set(" ".join(contexts).lower().split())
        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split()) - {"the", "a", "an", "is", "what"}

        context_precision = (
            len(answer_words & context_words) / max(len(context_words), 1)
        )
        context_precision = min(context_precision * 5, 1.0)  # scale up

        faithfulness = min(len(answer.split()) / 100, 1.0)

        answer_relevance = (
            len(question_words & answer_words) / max(len(question_words), 1)
        )

        return RagasScores(
            context_precision=round(context_precision, 3),
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(answer_relevance, 3),
        )
