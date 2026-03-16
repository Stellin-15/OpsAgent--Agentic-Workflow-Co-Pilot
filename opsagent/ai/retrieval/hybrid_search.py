"""Hybrid retrieval: dense vector search + BM25 keyword search fused via RRF.

Reciprocal Rank Fusion (RRF) combines rankings from two independent retrieval
systems.  It is parameter-free and consistently outperforms either system alone
on out-of-distribution queries (important for incident-specific jargon).

Formula:  score(d) = Σ_r  1 / (k + rank_r(d))   where k=60 (empirical default)

References:
    Cormack, Clarke, Buettcher — "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods" (SIGIR 2009)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from opsagent.ai.retrieval.pgvector_store import PgVectorStore

log = structlog.get_logger(__name__)

_RRF_K = 60


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """Combines pgvector dense search with in-memory BM25 keyword search."""

    def __init__(
        self,
        vector_store: "PgVectorStore",
        k: int = 5,
    ) -> None:
        self._store = vector_store
        self._k = k
        # BM25 corpus — built when chunks are indexed
        self._bm25 = None
        self._corpus_docs: list[tuple[str, str]] = []  # (content, source)

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def build_bm25_index(self, docs: list[tuple[str, str]]) -> None:
        """
        Build the in-memory BM25 index.

        *docs* is a list of (content, source) pairs — the same chunks stored
        in pgvector.  Called after every runbook re-index.
        """
        try:
            from rank_bm25 import BM25Okapi
            self._corpus_docs = docs
            tokenised = [d[0].lower().split() for d in docs]
            self._bm25 = BM25Okapi(tokenised)
            log.info("bm25.indexed", doc_count=len(docs))
        except ImportError:
            log.warning("bm25.rank_bm25_not_installed — falling back to dense only")
            self._bm25 = None

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    async def search(self, query: str) -> list[dict]:
        """
        Run dense + (optionally) BM25 search, fuse with RRF, return top-k.

        Returns a list of dicts:
            { "content": str, "source": str, "score": float }
        """
        dense_results = await self._store.similarity_search(query, k=self._k * 2)

        rrf_scores: dict[str, float] = defaultdict(float)
        doc_lookup: dict[str, tuple[str, str]] = {}

        # Dense ranks
        for rank, (content, source, _dist) in enumerate(dense_results):
            key = f"{source}::{content[:80]}"
            rrf_scores[key] += _rrf_score(rank)
            doc_lookup[key] = (content, source)

        # BM25 ranks (keyword)
        if self._bm25 is not None:
            tokens = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[: self._k * 2]
            for rank, idx in enumerate(top_indices):
                content, source = self._corpus_docs[idx]
                key = f"{source}::{content[:80]}"
                rrf_scores[key] += _rrf_score(rank)
                doc_lookup[key] = (content, source)

        # Sort by RRF score, take top-k
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        results = []
        for key in sorted_keys[: self._k]:
            content, source = doc_lookup[key]
            results.append(
                {"content": content, "source": source, "score": rrf_scores[key]}
            )

        return results
