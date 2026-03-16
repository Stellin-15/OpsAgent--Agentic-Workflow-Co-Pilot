#!/usr/bin/env python
"""
CLI script to manually rebuild the pgvector runbook index.

Usage:
    python scripts/index_runbooks.py
    # or:
    make index-runbooks

Requires DATABASE_URL and GOOGLE_API_KEY to be set in environment.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from opsagent.config import get_settings
    from opsagent.database import init_db, get_engine
    from opsagent.ai.embeddings.google import GoogleEmbeddingProvider
    from opsagent.ai.retrieval.chunking import MarkdownChunker
    from opsagent.ai.retrieval.pgvector_store import PgVectorStore
    from opsagent.cache.embedding_cache import EmbeddingCache

    settings = get_settings()

    print(f"Connecting to database: {settings.database_url[:40]}...")
    init_db(settings.database_url)
    engine = get_engine()

    cache = EmbeddingCache(redis_url=settings.redis_url)
    embedder = GoogleEmbeddingProvider(api_key=settings.google_api_key, cache=cache)
    store = PgVectorStore(engine, embedder)

    runbooks_path = Path(settings.runbooks_path)
    if not runbooks_path.exists():
        print(f"ERROR: Runbooks path not found: {runbooks_path}")
        sys.exit(1)

    print(f"Chunking runbooks from: {runbooks_path}")
    chunker = MarkdownChunker(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    chunks = chunker.chunk_directory(runbooks_path)
    print(f"Found {len(chunks)} chunks across {len({c.source for c in chunks})} files")

    if not chunks:
        print("No markdown files found — nothing to index.")
        sys.exit(0)

    print("Embedding and upserting to pgvector...")
    count = await store.upsert_chunks(chunks)
    print(f"Done. Indexed {count} chunks.")


if __name__ == "__main__":
    asyncio.run(main())
