"""pgvector-backed runbook chunk store.

Replaces the FAISS in-memory index from Phase 1-3.  Chunks and their
embeddings are persisted in PostgreSQL via the pgvector extension so they
survive restarts and are searchable with SQL.

The model for the chunks lives in ``opsagent.models.runbook_chunk``.

Usage
-----
    store = PgVectorStore(engine, embedding_provider)
    await store.upsert_chunks(chunks)                # index runbooks
    results = await store.similarity_search(query, k=3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from opsagent.ai.retrieval.chunking import Chunk

if TYPE_CHECKING:
    from opsagent.ai.embeddings.base import EmbeddingProvider

log = structlog.get_logger(__name__)


class PgVectorStore:
    """CRUD + similarity search over the ``runbook_chunks`` table."""

    def __init__(
        self,
        engine: AsyncEngine,
        embedding_provider: "EmbeddingProvider",
    ) -> None:
        self._engine = engine
        self._embedder = embedding_provider
        self._factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed each chunk and upsert into the DB.
        Returns the number of rows written.
        """
        from opsagent.models.runbook_chunk import RunbookChunk

        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        vectors = await self._embedder.embed_batch(texts)

        async with self._factory() as session:
            async with session.begin():
                # Delete existing chunks for the same source files to avoid
                # duplicates on re-indexing.
                sources = list({c.source for c in chunks})
                await session.execute(
                    delete(RunbookChunk).where(RunbookChunk.source.in_(sources))
                )

                for chunk, vector in zip(chunks, vectors):
                    row = RunbookChunk(
                        content=chunk.content,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        embedding=vector,
                    )
                    session.add(row)

        log.info("pgvector.upsert", count=len(chunks))
        return len(chunks)

    async def clear(self) -> None:
        """Delete all rows — used during tests."""
        from opsagent.models.runbook_chunk import RunbookChunk
        async with self._factory() as session:
            async with session.begin():
                await session.execute(delete(RunbookChunk))

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """
        Embed *query* then retrieve the *k* nearest chunks.

        Returns a list of (content, source, cosine_distance) tuples.
        Lower distance = more similar.
        """
        from opsagent.models.runbook_chunk import RunbookChunk

        query_vector = await self._embedder.embed(query)

        async with self._factory() as session:
            # pgvector cosine distance operator: <=>
            result = await session.execute(
                select(
                    RunbookChunk.content,
                    RunbookChunk.source,
                    RunbookChunk.embedding.cosine_distance(query_vector).label("dist"),
                )
                .order_by(text("dist"))
                .limit(k)
            )
            rows = result.all()

        return [(row.content, row.source, float(row.dist)) for row in rows]
