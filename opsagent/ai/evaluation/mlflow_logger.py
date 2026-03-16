"""MLflow experiment tracker for RAG pipeline runs.

Logs each draft generation as an MLflow run with:
    Parameters:  model used, k retrieved, chunking strategy
    Metrics:     RAGAS scores, token counts, latency_ms
    Tags:        incident_id, alert_name, provider

Falls back silently if MLflow is not configured (no MLFLOW_TRACKING_URI).
This lets the pipeline run in development without MLflow installed.

Usage:
    logger = MlflowLogger()
    with logger.start_run(experiment_name="opsagent_rag") as run_id:
        logger.log_params({"model": "gemini-1.5-flash", "k": 5})
        logger.log_metrics({"faithfulness": 0.87, "latency_ms": 320})
        logger.log_tags({"incident_id": "01ABCDEF..."})
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import structlog

log = structlog.get_logger(__name__)


class MlflowLogger:
    """Thin wrapper around MLflow that degrades gracefully when unavailable."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri
        self._available = self._check_mlflow()

    @staticmethod
    def _check_mlflow() -> bool:
        try:
            import mlflow  # noqa: F401
            return True
        except ImportError:
            return False

    @contextmanager
    def start_run(
        self,
        experiment_name: str = "opsagent_rag",
        run_name: str | None = None,
    ) -> Generator[str | None, None, None]:
        """Context manager that starts an MLflow run and returns run_id."""
        if not self._available:
            yield None
            return

        try:
            import mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                yield run.info.run_id
        except Exception as exc:
            log.warning("mlflow.run_failed", error=str(exc))
            yield None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            import mlflow
            mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception as exc:
            log.warning("mlflow.log_params_failed", error=str(exc))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self._available:
            return
        try:
            import mlflow
            mlflow.log_metrics(metrics)
        except Exception as exc:
            log.warning("mlflow.log_metrics_failed", error=str(exc))

    def log_tags(self, tags: dict[str, str]) -> None:
        if not self._available:
            return
        try:
            import mlflow
            mlflow.set_tags(tags)
        except Exception as exc:
            log.warning("mlflow.log_tags_failed", error=str(exc))

    def log_draft(
        self,
        *,
        incident_id: str,
        alert_name: str,
        provider: str,
        model: str,
        k: int,
        ragas_scores: dict[str, float],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Convenience method — log a complete RAG pipeline run."""
        with self.start_run(run_name=f"draft_{incident_id[:8]}"):
            self.log_params(
                {
                    "provider": provider,
                    "model": model,
                    "retrieval_k": k,
                }
            )
            self.log_metrics(
                {
                    "context_precision": ragas_scores.get("context_precision", 0),
                    "faithfulness": ragas_scores.get("faithfulness", 0),
                    "answer_relevance": ragas_scores.get("answer_relevance", 0),
                    "confidence": ragas_scores.get("confidence", 0),
                    "input_tokens": float(input_tokens),
                    "output_tokens": float(output_tokens),
                    "latency_ms": latency_ms,
                }
            )
            self.log_tags(
                {
                    "incident_id": incident_id,
                    "alert_name": alert_name,
                }
            )
