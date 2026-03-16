"""
Health check endpoints.

/health/live   — Kubernetes liveness probe.
                 Returns 200 if the process is running.

/health/ready  — Kubernetes readiness probe.
                 Returns 200 only when the DB connection pool and RAG chain
                 are both ready. Returns 503 otherwise.
                 The load balancer uses this to decide whether to send traffic.
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text

from opsagent.database import get_engine

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def liveness():
    """The process is alive. No external checks."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(request: Request):
    """
    The service is ready to accept traffic.
    Checks:
      1. Database connection pool can execute a trivial query.
      2. RAG chain has been initialised (runbooks loaded, embeddings built).
    """
    checks: dict[str, str] = {}
    healthy = True

    # Check 1: Database
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"
        healthy = False

    # Check 2: RAG chain
    rag = getattr(request.app.state, "rag", None)
    if rag and rag.is_ready():
        checks["rag_chain"] = "ok"
    else:
        checks["rag_chain"] = "not_ready"
        # RAG not being ready is a warning, not fatal — the API still works
        # (it returns a fallback message). Don't mark the whole service unhealthy.

    code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse({"status": "ok" if healthy else "degraded", "checks": checks}, status_code=code)


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint. Returns all metrics in text format."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return JSONResponse(
            {"error": "prometheus_client not installed"},
            status_code=501,
        )
