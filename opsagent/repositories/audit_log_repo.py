"""
AuditLog repository — append-only writes only. Never update or delete.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.models.audit_log import AuditLog


async def log_event(
    session: AsyncSession,
    *,
    incident_id: str,
    event_type: str,
    actor: str = "system",
    payload: dict | None = None,
) -> AuditLog:
    entry = AuditLog(
        incident_id=incident_id,
        event_type=event_type,
        actor=actor,
        payload=payload,
    )
    session.add(entry)
    await session.commit()
    return entry
