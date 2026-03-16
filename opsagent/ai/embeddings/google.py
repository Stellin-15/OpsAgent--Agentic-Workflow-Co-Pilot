"""Google text-embedding-004 provider with Redis cache."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from opsagent.cache.embedding_cache import EmbeddingCache

log = structlog.get_logger(__name__)

_MODEL_ID = "models/text-embedding-004"
_DIMENSIONS = 768


class GoogleEmbeddingProvider:
    """
    Wraps google.generativeai.embed_content with an optional Redis cache.

    The cache key is sha256(model_id + text) so embeddings are portable
    across restarts and Redis instances.
    """

    def __init__(
        self,
        api_key: str,
        cache: "EmbeddingCache | None" = None,
        model: str = _MODEL_ID,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._configured = False

    def _ensure_configured(self) -> None:
        if not self._configured:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._configured = True

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._model}:{text}".encode()).hexdigest()
        return f"emb:{digest}"

    async def embed(self, text: str) -> list[float]:
        # Cache hit?
        if self._cache:
            cached = await self._cache.get(self._cache_key(text))
            if cached is not None:
                return cached

        self._ensure_configured()
        import google.generativeai as genai

        result = await asyncio.to_thread(
            genai.embed_content,
            model=self._model,
            content=text,
            task_type="retrieval_document",
        )
        vector: list[float] = result["embedding"]

        if self._cache:
            await self._cache.set(self._cache_key(text), vector)

        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Run embeds concurrently — each still checks cache individually
        return list(await asyncio.gather(*[self.embed(t) for t in texts]))
