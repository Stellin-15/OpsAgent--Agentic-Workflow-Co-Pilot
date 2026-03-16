"""Production RAG pipeline (Phase 4).

Replaces Phase 1-3's LangChain-based RagService with a purpose-built
pipeline that gives us full control over:
    - embedding caching
    - hybrid retrieval (dense + BM25)
    - multi-model LLM routing with A/B testing
    - RAGAS evaluation on every draft
    - MLflow experiment tracking
    - SSE token streaming

The pipeline is used both from the Celery worker (background processing)
and from the streaming endpoint (real-time token delivery to the browser).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import structlog

from opsagent.ai.evaluation.mlflow_logger import MlflowLogger
from opsagent.ai.evaluation.ragas_eval import RagasEvaluator
from opsagent.ai.llm.base import LLMMessage
from opsagent.ai.retrieval.chunking import MarkdownChunker

if TYPE_CHECKING:
    from opsagent.ai.llm.router import LLMRouter
    from opsagent.ai.retrieval.hybrid_search import HybridRetriever
    from opsagent.config import Settings

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are OpsAgent, an expert SRE assistant.
Your job is to produce a clear, numbered remediation plan for the given alert.
Base your response ONLY on the provided runbook context.
If the context doesn't cover the issue, say so clearly rather than guessing.
Format: markdown with numbered steps, commands in code blocks, warnings clearly marked."""


class RagPipeline:
    """
    End-to-end RAG pipeline.

    Must be initialised before use:
        pipeline = await RagPipeline.create(settings)
    """

    def __init__(
        self,
        retriever: "HybridRetriever",
        llm_router: "LLMRouter",
        evaluator: RagasEvaluator,
        mlflow_logger: MlflowLogger,
        k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._router = llm_router
        self._evaluator = evaluator
        self._mlflow = mlflow_logger
        self._k = k
        self._ready = False

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    async def create(cls, settings: "Settings") -> "RagPipeline":
        """Build and initialise the full pipeline from settings."""
        from opsagent.ai.embeddings.google import GoogleEmbeddingProvider
        from opsagent.ai.llm.router import LLMRouter
        from opsagent.ai.retrieval.hybrid_search import HybridRetriever
        from opsagent.ai.retrieval.pgvector_store import PgVectorStore
        from opsagent.cache.embedding_cache import EmbeddingCache
        from opsagent.database import get_engine

        # Embedding provider with Redis cache
        cache = EmbeddingCache(redis_url=settings.redis_url)
        embedder = GoogleEmbeddingProvider(
            api_key=settings.google_api_key,
            cache=cache,
        )

        # pgvector store
        engine = get_engine()
        vector_store = PgVectorStore(engine, embedder)

        # Hybrid retriever
        retriever = HybridRetriever(vector_store, k=settings.rag_retrieval_k)

        # Index runbooks into pgvector (and BM25)
        runbooks_path = Path(settings.runbooks_path)
        if runbooks_path.exists():
            chunker = MarkdownChunker(
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
            chunks = chunker.chunk_directory(runbooks_path)
            if chunks:
                await vector_store.upsert_chunks(chunks)
                retriever.build_bm25_index(
                    [(c.content, c.source) for c in chunks]
                )
                log.info(
                    "rag_pipeline.indexed",
                    chunk_count=len(chunks),
                    runbooks_path=str(runbooks_path),
                )
        else:
            log.warning("rag_pipeline.runbooks_not_found", path=str(runbooks_path))

        # LLM router
        llm_router = LLMRouter.from_settings(settings)

        # Evaluator and MLflow
        evaluator = RagasEvaluator()
        mlflow_uri = getattr(settings, "mlflow_tracking_uri", None)
        mlflow_logger = MlflowLogger(tracking_uri=mlflow_uri)

        pipeline = cls(
            retriever=retriever,
            llm_router=llm_router,
            evaluator=evaluator,
            mlflow_logger=mlflow_logger,
            k=settings.rag_retrieval_k,
        )
        pipeline._ready = True
        return pipeline

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self._ready

    async def generate_draft(
        self,
        incident_id: str,
        alert_name: str,
        labels: dict,
        description: str | None,
    ) -> tuple[str, dict]:
        """
        Generate a draft remediation plan.

        Returns:
            (draft_content, ragas_scores_dict)
        """
        t0 = time.monotonic()

        # Retrieve context
        query = self._build_query(alert_name, labels, description)
        docs = await self._retriever.search(query)
        contexts = [d["content"] for d in docs]
        retrieval_chunks = [
            {"content": d["content"][:300], "source": d["source"], "score": d["score"]}
            for d in docs
        ]

        # Build messages
        messages = self._build_messages(query, contexts)

        # LLM call
        response = await self._router.complete(
            messages,
            incident_id=incident_id,
        )

        latency_ms = (time.monotonic() - t0) * 1000

        # RAGAS evaluation
        ragas_scores = await self._evaluator.evaluate(
            question=query,
            answer=response.content,
            contexts=contexts,
        )

        # MLflow tracking
        self._mlflow.log_draft(
            incident_id=incident_id,
            alert_name=alert_name,
            provider=response.provider,
            model=response.model,
            k=self._k,
            ragas_scores=ragas_scores.to_dict(),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency_ms,
        )

        log.info(
            "rag_pipeline.draft_generated",
            incident_id=incident_id,
            model=response.model,
            confidence=ragas_scores.confidence,
            low_confidence=ragas_scores.is_low_confidence,
            latency_ms=round(latency_ms),
        )

        return response.content, ragas_scores.to_dict()

    async def stream_draft(
        self,
        incident_id: str,
        alert_name: str,
        labels: dict,
        description: str | None,
    ) -> AsyncIterator[str]:
        """
        Stream draft tokens for real-time SSE delivery.

        Yields individual token strings. Retrieval and RAGAS eval happen
        before the first token is yielded.
        """
        query = self._build_query(alert_name, labels, description)
        docs = await self._retriever.search(query)
        contexts = [d["content"] for d in docs]
        messages = self._build_messages(query, contexts)

        async for token in self._router.stream(
            messages,
            incident_id=incident_id,
        ):
            yield token

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_query(
        alert_name: str,
        labels: dict,
        description: str | None,
    ) -> str:
        parts = [f"Alert: {alert_name}"]
        if labels:
            labels_str = ", ".join(f"{k}={v}" for k, v in labels.items())
            parts.append(f"Labels: {labels_str}")
        if description:
            parts.append(f"Description: {description}")
        return "\n".join(parts)

    @staticmethod
    def _build_messages(query: str, contexts: list[str]) -> list[LLMMessage]:
        context_block = "\n\n---\n\n".join(contexts) if contexts else "(no runbook context found)"
        user_content = f"""RUNBOOK CONTEXT:
{context_block}

ALERT:
{query}

Please provide a step-by-step remediation plan based on the runbook context above."""
        return [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_content),
        ]
