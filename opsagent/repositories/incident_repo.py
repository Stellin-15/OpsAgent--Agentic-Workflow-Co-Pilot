"""
Incident repository — all database access for the incidents table.

Route handlers never import SQLAlchemy directly; they call these functions.
This makes unit-testing the service layer trivial (mock the repo, not the DB).
"""

from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from opsagent.models.incident import Incident, IncidentStatus

log = structlog.get_logger(__name__)


async def create_incident(
    session: AsyncSession,
    *,
    incident_id: str,
    alert_name: str,
    labels: dict,
    description: str | None,
    severity: str,
    source: str,
    fired_at: datetime | None = None,
) -> Incident:
    incident = Incident(
        id=incident_id,
        alert_name=alert_name,
        labels=labels,
        description=description,
        severity=severity,
        source=source,
        status=IncidentStatus.FIRING,
        fired_at=fired_at or datetime.now(timezone.utc),
    )
    session.add(incident)
    await session.commit()
    await session.refresh(incident)
    log.info("incident_created", incident_id=incident_id, alert_name=alert_name)
    return incident


async def get_by_id(
    session: AsyncSession,
    incident_id: str,
    *,
    load_drafts: bool = False,
) -> Incident | None:
    stmt = select(Incident).where(Incident.id == incident_id)
    if load_drafts:
        stmt = stmt.options(selectinload(Incident.drafts))
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_incidents(
    session: AsyncSession,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Incident], int]:
    stmt = select(Incident).options(selectinload(Incident.drafts))
    if status:
        stmt = stmt.where(Incident.status == status)
    stmt = stmt.order_by(Incident.fired_at.desc()).offset(offset).limit(limit)
    result = await session.execute(stmt)
    items = list(result.scalars().all())

    # Count query (no pagination)
    from sqlalchemy import func
    count_stmt = select(func.count()).select_from(Incident)
    if status:
        count_stmt = count_stmt.where(Incident.status == status)
    total = (await session.execute(count_stmt)).scalar_one()

    return items, total


async def update_status(
    session: AsyncSession,
    incident_id: str,
    new_status: str,
) -> Incident | None:
    incident = await get_by_id(session, incident_id)
    if not incident:
        return None
    old_status = incident.status
    incident.status = new_status
    if new_status == IncidentStatus.RESOLVED:
        incident.resolved_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(incident)
    log.info(
        "incident_status_updated",
        incident_id=incident_id,
        old_status=old_status,
        new_status=new_status,
    )
    return incident
