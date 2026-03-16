"""
OpsAgent — FastAPI application factory.

This module wires up the application. No business logic here.

Lifespan (startup):
  1. Configure structured JSON logging
  2. Initialise PostgreSQL connection pool (SQLAlchemy async)
  3. Initialise RAG service (loads runbooks, builds FAISS index)
  4. Initialise Slack notifier
  5. Initialise Slack interactive bot (if SLACK_SIGNING_SECRET is set)
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from opsagent.api import actions, alerts, billing, health, incidents, slack_actions, streaming
from opsagent.config import get_settings
from opsagent.database import init_db
from opsagent.logging_config import setup_logging
from opsagent.observability.middleware import RequestLoggingMiddleware
from opsagent.observability.tracing import setup_tracing
from opsagent.services.rag_service import RagService
from opsagent.services.slack_service import SlackService

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. Logging + tracing
    setup_logging(log_level=settings.log_level)
    setup_tracing(service_name="opsagent-api")
    log.info("opsagent_starting", environment=settings.environment, version="0.3.0")

    # 2. Database
    init_db(settings.database_url)
    log.info("database_initialized")

    # 3. RAG service — Phase 4 uses RagPipeline (pgvector + multi-model router)
    #    Falls back to Phase 1-3 RagService when API keys aren't configured.
    try:
        from opsagent.ai.pipeline.rag_pipeline import RagPipeline
        rag = await RagPipeline.create(settings)
        log.info("rag_pipeline_v4_initialized")
    except Exception as exc:
        log.warning("rag_pipeline_v4_failed_falling_back", error=str(exc))
        rag = RagService(settings)
        await rag.initialize()
    app.state.rag = rag

    # 4. Slack notifier (for direct-from-API fallback path)
    slack = SlackService(settings)
    app.state.slack = slack
    if slack.is_configured():
        log.info("slack_configured", channel=settings.slack_channel)
    else:
        log.warning("slack_not_configured")

    # 5. Slack interactive bot (Phase 2)
    from opsagent.integrations.slack.bot import init_slack_bot
    bot_ok = init_slack_bot(settings)
    if bot_ok:
        log.info("slack_bot_initialized")

    # 6. Action catalog (Phase 3)
    from opsagent.api.actions import _get_catalog
    catalog = _get_catalog()
    log.info("action_catalog_loaded", action_count=len(catalog))

    log.info("opsagent_ready")
    yield

    log.info("opsagent_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OpsAgent",
        description=(
            "Open-source AI SRE co-pilot. "
            "Alert fires → AI reads your runbooks → drafts a fix → human approves → actions run."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Core API routes
    app.include_router(health.router)
    app.include_router(alerts.router, prefix="/api")
    app.include_router(incidents.router, prefix="/api")

    # Slack interactive bot endpoint (Phase 2)
    app.include_router(slack_actions.router, prefix="/api")

    # Safe execution engine (Phase 3)
    app.include_router(actions.router, prefix="/api")

    # SSE draft streaming (Phase 4)
    app.include_router(streaming.router, prefix="/api")

    # Billing (Phase 6)
    app.include_router(billing.router, prefix="/api")

    # Serve frontend
    try:
        app.mount("/", StaticFiles(directory="static", html=True), name="static")
    except RuntimeError:
        log.warning("static_dir_missing", path="static")

    return app


app = create_app()
