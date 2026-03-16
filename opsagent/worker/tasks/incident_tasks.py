"""
Celery task: process_incident

Runs asynchronously in a worker process after an alert webhook is received.
The API returns ~5ms; this task handles the heavy lifting (RAG, Slack notify).

Design principles:
  - Idempotent: running the task twice for the same incident_id is safe.
    We check if a draft already exists before running the RAG chain.
  - Retriable: autoretry_for=(Exception,) with exponential backoff handles
    transient Gemini API errors, network blips, etc.
  - Observable: every step logs structured JSON (structlog).
"""

import asyncio
import logging

from opsagent.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="opsagent.tasks.process_incident",
    autoretry_for=(Exception,),
    retry_backoff=True,           # 2s, 4s, 8s, ...
    retry_backoff_max=60,
    max_retries=3,
    acks_late=True,
)
def process_incident(self, incident_id: str) -> dict:
    """
    Process a single incident:
      1. Check idempotency (skip if draft already exists)
      2. Generate RAG draft
      3. Store draft in DB
      4. Update incident status → DRAFT_READY
      5. Send Slack Block Kit notification with Approve / Reject buttons
    """
    return asyncio.run(_process(incident_id))


async def _process(incident_id: str) -> dict:
    from opsagent.database import get_session_factory
    from opsagent.repositories import audit_log_repo, draft_repo, incident_repo
    from opsagent.worker.state import get_worker_rag
    from opsagent.integrations.slack.notifier import SlackNotifier
    from opsagent.config import get_settings
    from ulid import ULID

    settings = get_settings()
    factory = get_session_factory()

    async with factory() as db:
        # ── 1. Load incident ──────────────────────────────────────────────────
        incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
        if not incident:
            log.error("process_incident: incident not found", extra={"incident_id": incident_id})
            return {"status": "error", "reason": "not_found"}

        # ── 2. Idempotency check ──────────────────────────────────────────────
        # If a draft already exists (e.g. task was retried), skip RAG generation.
        existing_draft = await draft_repo.get_latest_for_incident(db, incident_id)
        if existing_draft and incident.status == "DRAFT_READY":
            log.info(
                "process_incident: draft already exists, skipping",
                extra={"incident_id": incident_id},
            )
            return {"status": "skipped", "draft_id": existing_draft.id}

        # ── 3. Generate draft ─────────────────────────────────────────────────
        rag = get_worker_rag()
        draft_content = await rag.generate_draft(
            alert_name=incident.alert_name,
            labels=incident.labels,
            description=incident.description or "",
        )

        # ── 4. Store draft ────────────────────────────────────────────────────
        draft_id = str(ULID())
        draft = await draft_repo.create_draft(
            db,
            draft_id=draft_id,
            incident_id=incident_id,
            content=draft_content,
            model_used="gemini-2.0-flash",
        )

        # ── 5. Update status → DRAFT_READY ────────────────────────────────────
        await incident_repo.update_status(db, incident_id, "DRAFT_READY")
        await audit_log_repo.log_event(
            db,
            incident_id=incident_id,
            event_type="status_changed",
            actor="worker",
            payload={"from": "PROCESSING", "to": "DRAFT_READY"},
        )

        # ── 6. Slack notification with interactive Approve/Reject buttons ──────
        notifier = SlackNotifier(settings)
        slack_ts = await notifier.send_incident_draft(
            incident=incident,
            draft=draft,
        )
        if slack_ts:
            await audit_log_repo.log_event(
                db,
                incident_id=incident_id,
                event_type="slack_notification_sent",
                payload={"ts": slack_ts},
            )

    log.info(
        "process_incident: complete",
        extra={"incident_id": incident_id, "draft_id": draft_id},
    )
    return {"status": "ok", "incident_id": incident_id, "draft_id": draft_id}
