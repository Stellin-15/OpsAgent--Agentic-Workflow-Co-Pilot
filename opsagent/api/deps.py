"""
FastAPI dependency injection.

All route handlers receive their dependencies via Depends() — never
by importing globals directly. This keeps handlers unit-testable:
swap the dependency in tests, no database or LLM needed.
"""

from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.config import Settings, get_settings
from opsagent.database import get_session_factory
from opsagent.services.rag_service import RagService
from opsagent.services.slack_service import SlackService


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for the duration of a single request."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


def get_rag(request: Request) -> RagService:
    """Return the RAG service stored in app state (initialised at startup)."""
    return request.app.state.rag


def get_slack(request: Request) -> SlackService:
    """Return the Slack service stored in app state (initialised at startup)."""
    return request.app.state.slack


def get_config(request: Request) -> Settings:
    """Return the application settings."""
    return get_settings()
