"""
Redis-backed embedding cache.

Why this matters:
  Embedding API calls (Google, OpenAI) cost ~$0.0001 per 1k tokens and add
  ~100-300ms latency. The same SOP chunks are re-embedded on every startup.
  This cache stores vectors in Redis so we only pay the cost once per 24 hours.

Cache key design:
  sha256(model_name + ":" + text_content)
  Using SHA-256 ensures the key is always 64 hex chars regardless of input length,
  and collisions are practically impossible for this use case.

Usage:
  cache = EmbeddingCache(redis_url="redis://localhost:6379/0")
  vec = await cache.get("models/embedding-001", chunk_text)
  if vec is None:
      vec = await api_call(chunk_text)
      await cache.set("models/embedding-001", chunk_text, vec)
"""

import hashlib
import json

import structlog

log = structlog.get_logger(__name__)


class EmbeddingCache:
    """Async Redis cache for embedding vectors. Gracefully degrades if Redis is unavailable."""

    KEY_PREFIX = "emb:"
    DEFAULT_TTL = 86_400  # 24 hours in seconds

    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client = None

    async def _get_client(self):
        """Lazy-initialise the Redis client on first use."""
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _cache_key(self, model: str, text: str) -> str:
        digest = hashlib.sha256(f"{model}:{text}".encode("utf-8")).hexdigest()
        return f"{self.KEY_PREFIX}{digest}"

    async def get(self, model: str, text: str) -> list[float] | None:
        """Return cached embedding vector, or None if not cached."""
        try:
            client = await self._get_client()
            raw = await client.get(self._cache_key(model, text))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log.warning("embedding_cache_get_failed", error=str(exc))
            return None

    async def set(self, model: str, text: str, embedding: list[float]) -> None:
        """Cache an embedding vector with a TTL. Silently ignores Redis errors."""
        try:
            client = await self._get_client()
            await client.setex(
                name=self._cache_key(model, text),
                time=self._ttl,
                value=json.dumps(embedding),
            )
        except Exception as exc:
            log.warning("embedding_cache_set_failed", error=str(exc))

    async def invalidate(self, model: str, text: str) -> None:
        """Manually evict a cached entry."""
        try:
            client = await self._get_client()
            await client.delete(self._cache_key(model, text))
        except Exception as exc:
            log.warning("embedding_cache_invalidate_failed", error=str(exc))
