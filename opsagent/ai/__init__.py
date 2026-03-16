"""AI pipeline package — Phase 4.

Sub-packages:
    embeddings   — embedding provider abstraction (Google, OpenAI, local)
    llm          — LLM provider abstraction + multi-model router
    retrieval    — pgvector store, chunking, hybrid retrieval (dense + BM25)
    pipeline     — end-to-end RAG pipeline + SSE streaming
    evaluation   — RAGAS eval + MLflow experiment tracking
"""
