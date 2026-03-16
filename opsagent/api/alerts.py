"""
Alert ingestion endpoints.

Phase 2 architecture change:
  BEFORE: POST webhook → run RAG synchronously (~5s) → return full incident
  AFTER:  POST webhook → create incident in DB (~5ms) → enqueue Celery task
          → return {incident_id, status: "PROCESSING"}

  The Celery worker picks up the task, runs the RAG chain, stores the draft,
  and sends the Slack Block Kit notification with Approve/Reject buttons.
  The frontend (or Slack) polls GET /api/incidents/{id} to check progress.

Supported webhook sources:
  POST /api/alerts/webhook/alertmanager  — Prometheus AlertManager v4
  POST /api/alerts/webhook/grafana       — Grafana v8 legacy format
  POST /api/alerts/webhook/grafana/v9    — Grafana v9+ (AlertManager-compatible)
  POST /api/alerts/webhook/pagerduty     — PagerDuty webhook v3
  POST /api/alerts/webhook/manual        — Simple JSON for testing
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from opsagent.api.deps import get_db
from opsagent.integrations.alert_sources.alertmanager import AlertManagerAdapter
from opsagent.integrations.alert_sources.grafana import GrafanaAdapter
from opsagent.integrations.alert_sources.pagerduty import PagerDutyAdapter
from opsagent.repositories import audit_log_repo, incident_repo
from opsagent.schemas.alert import AlertManagerWebhook, ManualAlert, NormalisedAlert
from opsagent.schemas.incident import IncidentOut

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])

_am_adapter = AlertManagerAdapter()
_grafana_adapter = GrafanaAdapter()
_pd_adapter = PagerDutyAdapter()


async def _create_and_enqueue(
    normalised: NormalisedAlert,
    db: AsyncSession,
) -> IncidentOut:
    """
    Create an incident record and enqueue the Celery processing task.
    Returns immediately — the heavy RAG work happens in the worker.
    """
    incident_id = str(ULID())

    incident = await incident_repo.create_incident(
        db,
        incident_id=incident_id,
        alert_name=normalised.alert_name,
        labels=normalised.labels,
        description=normalised.description,
        severity=normalised.severity,
        source=normalised.source,
        fired_at=normalised.fired_at or datetime.now(timezone.utc),
    )

    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="incident_created",
        actor=normalised.source,
        payload={"alert_name": normalised.alert_name, "severity": normalised.severity},
    )

    # Update status → PROCESSING immediately
    await incident_repo.update_status(db, incident_id, "PROCESSING")
    await audit_log_repo.log_event(
        db,
        incident_id=incident_id,
        event_type="status_changed",
        payload={"from": "FIRING", "to": "PROCESSING"},
    )

    # Enqueue Celery task — fire and forget from the API's perspective
    from opsagent.worker.tasks.incident_tasks import process_incident
    process_incident.delay(incident_id)

    log.info(
        "incident_enqueued",
        incident_id=incident_id,
        alert_name=normalised.alert_name,
        source=normalised.source,
    )

    # Return incident without drafts (they'll be populated by the worker)
    incident = await incident_repo.get_by_id(db, incident_id)
    return IncidentOut.model_validate(incident)


# ── Prometheus AlertManager ────────────────────────────────────────────────────

@router.post(
    "/webhook/alertmanager",
    response_model=list[IncidentOut],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Prometheus AlertManager v4 webhook",
)
async def alertmanager_webhook(
    body: AlertManagerWebhook,
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts Prometheus AlertManager v4 payloads. One payload may contain
    multiple alerts — each becomes a separate incident.

    alertmanager.yml:
        receivers:
          - name: opsagent
            webhook_configs:
              - url: http://opsagent:8000/api/alerts/webhook/alertmanager
    """
    alerts = _am_adapter.normalise(body.model_dump())
    if not alerts:
        return []

    log.info("alertmanager_webhook", count=len(alerts))
    return [await _create_and_enqueue(a, db) for a in alerts]


# ── Grafana ────────────────────────────────────────────────────────────────────

@router.post(
    "/webhook/grafana",
    response_model=IncidentOut | None,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Grafana legacy (v8) webhook",
)
async def grafana_webhook(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Grafana v8 and earlier alert webhook.
    Grafana v9+ unified alerting → use /webhook/alertmanager instead.
    """
    alerts = _grafana_adapter.normalise(body)
    if not alerts:
        return None
    return await _create_and_enqueue(alerts[0], db)


@router.post(
    "/webhook/grafana/v9",
    response_model=list[IncidentOut],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Grafana v9+ unified alerting (AlertManager-compatible)",
)
async def grafana_v9_webhook(
    body: AlertManagerWebhook,
    db: AsyncSession = Depends(get_db),
):
    """
    Grafana v9+ uses AlertManager-compatible format.
    Configure Grafana → Contact Points → OpsAgent → Webhook URL:
        http://opsagent:8000/api/alerts/webhook/grafana/v9
    """
    alerts = _am_adapter.normalise(body.model_dump())
    if not alerts:
        return []
    return [await _create_and_enqueue(a, db) for a in alerts]


# ── PagerDuty ─────────────────────────────────────────────────────────────────

@router.post(
    "/webhook/pagerduty",
    response_model=IncidentOut | None,
    status_code=status.HTTP_202_ACCEPTED,
    summary="PagerDuty webhook v3",
)
async def pagerduty_webhook(
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    PagerDuty webhook v3 for incident.triggered events.
    PagerDuty → Integrations → Webhooks → Add Webhook Subscription
        URL: http://opsagent:8000/api/alerts/webhook/pagerduty
        Events: incident.triggered
    """
    alerts = _pd_adapter.normalise(body)
    if not alerts:
        return None
    return await _create_and_enqueue(alerts[0], db)


# ── Manual / testing ──────────────────────────────────────────────────────────

@router.post(
    "/webhook/manual",
    response_model=IncidentOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Manually create an incident (testing / demo)",
)
async def manual_alert(
    body: ManualAlert,
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a single incident from a simple JSON body.

        curl -X POST http://localhost:8000/api/alerts/webhook/manual \\
          -H 'Content-Type: application/json' \\
          -d '{"alert_name": "HighCPUUsage", "severity": "critical",
               "description": "web-01 CPU at 95% for 10 minutes"}'

    The response comes back immediately with status PROCESSING.
    Poll GET /api/incidents/{id} to see when the draft is ready.
    """
    normalised = NormalisedAlert(
        alert_name=body.alert_name,
        severity=body.severity,
        description=body.description,
        labels=body.labels,
        source="manual",
    )
    return await _create_and_enqueue(normalised, db)
