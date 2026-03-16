"""FastAPI middleware for observability (Phase 5).

RequestLoggingMiddleware:
    - Generates a correlation ID (UUID) per request if not supplied
    - Binds it to structlog context for all log lines in the request
    - Records request latency and status code to Prometheus

Usage — added in main.py:
    app.add_middleware(RequestLoggingMiddleware)
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from opsagent.logging_config import bind_correlation_id

log = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request middleware that:
      1. Assigns a correlation-ID (from X-Correlation-ID header or generated)
      2. Binds it to structlog so all log lines within the request carry it
      3. Logs request start + end with latency
      4. Records prometheus http_requests_total and http_request_duration_seconds
    """

    def __init__(self, app: ASGIApp, *, exclude_paths: set[str] | None = None) -> None:
        super().__init__(app)
        self._exclude = exclude_paths or {"/health/live", "/health/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._exclude:
            return await call_next(request)

        correlation_id = (
            request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        )
        bind_correlation_id(correlation_id)

        t0 = time.monotonic()
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
        )

        response = await call_next(request)
        latency_ms = (time.monotonic() - t0) * 1000

        log.info(
            "http.response",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            correlation_id=correlation_id,
        )

        # Prometheus metrics — imported lazily to avoid import errors if
        # prometheus_client is not installed.
        try:
            from opsagent.observability.prometheus_middleware import record_request
            record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=(time.monotonic() - t0),
            )
        except Exception:
            pass

        response.headers["X-Correlation-ID"] = correlation_id
        return response
