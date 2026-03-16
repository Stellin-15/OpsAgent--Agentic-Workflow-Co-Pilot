"""
pytest fixtures shared across all tests.

Integration tests use a real PostgreSQL test database.
Run it first: docker compose up postgres -d

Celery tasks run synchronously in tests via task_always_eager=True so tests
don't need a Redis broker or a running worker.
"""

import os
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from opsagent.database import Base, init_db

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://opsagent:opsagent@localhost:5432/opsagent_test",
)


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Rolls back after each test — keeps tests isolated."""
    factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        async with session.begin():
            yield session
            await session.rollback()


@pytest_asyncio.fixture
async def client(test_engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    """
    Full HTTP test client.
    - Real PostgreSQL (test DB)
    - Stub RAG service (returns fixed draft, no LLM calls)
    - Celery tasks run eagerly/synchronously (no Redis needed)
    """
    init_db(TEST_DATABASE_URL)

    # ── Stub Celery so tasks run inline without a broker ──────────────────────
    from opsagent.worker.celery_app import celery_app
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)

    # ── Stub worker state so eager tasks don't fail on missing RAG ────────────
    from opsagent.worker import state as worker_state

    class _StubRag:
        def is_ready(self) -> bool:
            return True

        async def generate_draft(self, alert_name, labels, description) -> str:
            return f"**Stub draft for {alert_name}**\n1. Check logs\n2. Restart service"

        # Phase 4: RagPipeline interface — return (content, ragas_scores)
        async def generate_draft_v4(self, incident_id, alert_name, labels, description):
            content = f"**Stub draft for {alert_name}**\n1. Check logs\n2. Restart service"
            ragas = {"context_precision": 0.8, "faithfulness": 0.7, "answer_relevance": 0.9, "confidence": 0.8, "low_confidence": False}
            return content, ragas

        async def stream_draft(self, incident_id, alert_name, labels, description):
            content = f"**Stub draft for {alert_name}**\n1. Check logs\n2. Restart service"
            # Yield the content in a few chunks to simulate streaming
            for chunk in content.split("\n"):
                yield chunk + "\n"

    worker_state.set_worker_rag(_StubRag())

    # ── Stub Slack notifier so tests don't post to Slack ──────────────────────
    from opsagent.integrations.slack import notifier as notifier_module
    original_notifier = notifier_module.SlackNotifier

    class _StubNotifier:
        def __init__(self, *args, **kwargs):
            pass

        def is_configured(self) -> bool:
            return False

        async def send_incident_draft(self, **kwargs) -> str | None:
            return None

        async def update_message(self, **kwargs) -> bool:
            return False

    notifier_module.SlackNotifier = _StubNotifier

    from opsagent.main import create_app
    app = create_app()
    app.state.rag = _StubRag()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    # Restore
    notifier_module.SlackNotifier = original_notifier
