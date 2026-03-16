"""Redis sliding-window rate limiter (Phase 6).

Limits per tier:
    free       — 100 alerts/day
    starter    — 500 alerts/month (≈ 16/day)
    team       — 5000 alerts/month (≈ 166/day)
    enterprise — unlimited

Algorithm: Redis sorted set sliding window.
    Key:   rate:{team_id}:{window}
    Score: Unix timestamp (float)
    Value: UUID per request (unique member)

On each request:
    1. Remove all members older than the window
    2. Count current members
    3. If count >= limit → 429
    4. Add current request as new member
    5. Set key TTL to window duration

This is O(log N) per request and accurate to within 1ms.

Usage:
    limiter = RateLimiter(redis_url=settings.redis_url)
    await limiter.check(team_id="01ABC...", tier="starter")  # raises 429 if over limit
"""

from __future__ import annotations

import time
import uuid

from fastapi import HTTPException, status
import structlog

log = structlog.get_logger(__name__)

# (window_seconds, max_requests)
_TIER_LIMITS: dict[str, tuple[int, int]] = {
    "free": (86400, 100),           # 100/day
    "starter": (86400 * 30, 500),   # 500/month
    "team": (86400 * 30, 5000),     # 5000/month
    "enterprise": (0, 0),            # unlimited
}


class RateLimiter:
    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def check(self, team_id: str, tier: str) -> None:
        """
        Check rate limit for *team_id* on *tier*.
        Raises HTTP 429 if the team has exceeded their limit.
        Falls back silently if Redis is unavailable.
        """
        if tier == "enterprise":
            return  # unlimited

        window_seconds, max_requests = _TIER_LIMITS.get(tier, (86400, 100))

        try:
            client = self._get_client()
            now = time.time()
            key = f"rate:{team_id}:{tier}"
            member = str(uuid.uuid4())

            pipe = client.pipeline()
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {member: now})
            pipe.expire(key, window_seconds)
            results = await pipe.execute()

            current_count = results[1]  # count before adding the new request
            if current_count >= max_requests:
                log.warning(
                    "rate_limit.exceeded",
                    team_id=team_id,
                    tier=tier,
                    count=current_count,
                    limit=max_requests,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit exceeded for tier '{tier}'. "
                        f"Limit: {max_requests} requests per {window_seconds // 86400} day(s)."
                    ),
                    headers={"Retry-After": str(window_seconds)},
                )

        except HTTPException:
            raise
        except Exception as exc:
            # Redis unavailable — allow the request through
            log.warning("rate_limiter.redis_error", error=str(exc))
