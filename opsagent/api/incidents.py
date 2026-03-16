"""
Incident management endpoints.

GET  /api/incidents           — list all incidents (paginated, filterable by status)
GET  /api/incidents/{id}      — get single incident with drafts
POST /api/incidents/{id}/approve — approve the latest draft → post to Slack
POST /api/incidents/{id}/reject  — reject the latest draft with a reason
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.api.deps import get_db, get_slack
from opsagent.repositories import audit_log_repo, draft_repo, incident_repo
from opsagent.schemas.incident import IncidentListOut, IncidentOut, RejectBody
from opsagent.services.slack_service import SlackService

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.get("", response_model=IncidentListOut, summary="List incidents")
async def list_incidents(
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    items, total = await incident_repo.list_incidents(
        db, status=status_filter, limit=limit, offset=offset
    )
    return IncidentListOut(
        items=[IncidentOut.model_validate(i) for i in items],
        total=total,
    )


@router.get("/{incident_id}", response_model=IncidentOut, summary="Get incident by ID")
async def get_incident(incident_id: str, db: AsyncSession = Depends(get_db)):
    incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return IncidentOut.model_validate(incident)


@router.post(
    "/{incident_id}/approve",
    response_model=IncidentOut,
    summary="Approve the latest draft and post to Slack",
)
async def approve_incident(
    incident_id: str,
    db: AsyncSession = Depends(get_db),
    slack: SlackService = Depends(get_slack),
):
    incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.status not in ("DRAFT_READY", "REJECTED"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot approve incident in status '{incident.status}'",
        )

    draft = await draft_repo.get_latest_for_incident(db, incident_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="No draft found for this incident"
        )

    # Post to Slack (best-effort; don't fail the approval if Slack is down)
    slack_ok = await slack.post_draft(
        incident_id=incident_id,
        alert_name=incident.alert_name,
        draft=draft.content,
    )

    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="draft_approved",
        actor="user",  # Phase 5: replace with JWT user identity
        payload={"draft_id": draft.id, "slack_posted": slack_ok},
    )

    updated = await incident_repo.update_status(db, incident_id, "RESOLVED")
    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="status_changed",
        payload={"from": "DRAFT_READY", "to": "RESOLVED"},
    )

    incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
    return IncidentOut.model_validate(incident)


@router.post(
    "/{incident_id}/reject",
    response_model=IncidentOut,
    summary="Reject the latest draft with an optional reason",
)
async def reject_incident(
    incident_id: str,
    body: RejectBody,
    db: AsyncSession = Depends(get_db),
):
    incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    if incident.status not in ("DRAFT_READY", "APPROVED"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot reject incident in status '{incident.status}'",
        )

    draft = await draft_repo.get_latest_for_incident(db, incident_id)

    # Record rejection — this feeds the Phase 4 feedback loop
    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="draft_rejected",
        actor="user",  # Phase 5: replace with JWT user identity
        payload={
            "draft_id": draft.id if draft else None,
            "reason": body.reason,
            "notes": body.notes,
        },
    )

    await incident_repo.update_status(db, incident_id, "REJECTED")
    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="status_changed",
        payload={"from": "DRAFT_READY", "to": "REJECTED"},
    )

    log.info(
        "draft_rejected",
        incident_id=incident_id,
        reason=body.reason,
        notes=body.notes,
    )

    incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
    return IncidentOut.model_validate(incident)
