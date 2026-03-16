"""
Draft repository — all database access for the drafts table.
"""

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.models.draft import Draft

log = structlog.get_logger(__name__)


async def create_draft(
    session: AsyncSession,
    *,
    draft_id: str,
    incident_id: str,
    content: str,
    model_used: str,
    retrieval_score: float | None = None,
    retrieval_chunks: list | None = None,
) -> Draft:
    draft = Draft(
        id=draft_id,
        incident_id=incident_id,
        content=content,
        model_used=model_used,
        retrieval_score=retrieval_score,
        retrieval_chunks=retrieval_chunks,
    )
    session.add(draft)
    await session.commit()
    await session.refresh(draft)
    log.info("draft_created", draft_id=draft_id, incident_id=incident_id, model=model_used)
    return draft


async def get_latest_for_incident(
    session: AsyncSession, incident_id: str
) -> Draft | None:
    stmt = (
        select(Draft)
        .where(Draft.incident_id == incident_id)
        .order_by(Draft.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
