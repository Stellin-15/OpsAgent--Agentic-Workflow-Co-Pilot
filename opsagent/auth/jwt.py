"""JWT bearer token authentication (Phase 6).

Every API request must include:
    Authorization: Bearer <token>

Token payload:
    sub   — team ID (ULID string)
    tier  — "free" | "starter" | "team" | "enterprise"
    exp   — standard expiry
    iat   — issued at

Usage in route handlers:
    from opsagent.auth.jwt import require_auth
    from opsagent.auth.jwt import TokenPayload

    @router.get("/api/incidents")
    async def list_incidents(token: TokenPayload = Depends(require_auth)):
        # token.team_id, token.tier are available
        ...

The JWT secret is read from settings.jwt_secret (set via JWT_SECRET env var).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = structlog.get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


@dataclass
class TokenPayload:
    team_id: str
    tier: str
    exp: datetime


def create_token(
    team_id: str,
    tier: str,
    secret: str,
    expires_in: timedelta = timedelta(days=30),
) -> str:
    """Create a signed JWT for a team."""
    try:
        import jwt
    except ImportError:
        raise RuntimeError("PyJWT is required: pip install pyjwt")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": team_id,
        "tier": tier,
        "iat": now,
        "exp": now + expires_in,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> TokenPayload:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        import jwt
        from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
    except ImportError:
        raise RuntimeError("PyJWT is required: pip install pyjwt")

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return TokenPayload(
            team_id=payload["sub"],
            tier=payload.get("tier", "free"),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (InvalidTokenError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenPayload:
    """
    FastAPI dependency that extracts + validates the Bearer token.

    Raises 401 if missing or invalid.
    """
    from opsagent.config import get_settings

    settings = get_settings()

    # Auth disabled in development (no JWT_SECRET configured)
    if not settings.jwt_secret:
        return TokenPayload(
            team_id="dev-team",
            tier="enterprise",
            exp=datetime.now(timezone.utc) + timedelta(days=365),
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_token(credentials.credentials, settings.jwt_secret)


def require_tier(*allowed_tiers: str):
    """
    Factory that returns a dependency requiring a minimum tier.

    Usage:
        @router.post("/api/actions/{name}/execute")
        async def execute(token = Depends(require_tier("starter", "team", "enterprise"))):
            ...
    """
    def _check(token: TokenPayload = Depends(require_auth)) -> TokenPayload:
        if token.tier not in allowed_tiers:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {allowed_tiers}",
            )
        return token
    return _check
