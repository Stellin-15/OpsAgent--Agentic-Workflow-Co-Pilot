"""
Slack interactive bot — handles button click events from Block Kit messages.

When a user clicks ✅ Approve or ❌ Reject in Slack, Slack sends an HTTP POST
to our /api/slack/actions endpoint. Slack Bolt handles verification and parsing.

Setup (Slack App dashboard):
  1. Interactivity & Shortcuts → Request URL: https://your-domain/api/slack/actions
  2. OAuth & Permissions → Bot Token Scopes: chat:write, channels:read

Required env vars:
  SLACK_BOT_TOKEN    - xoxb-...
  SLACK_SIGNING_SECRET - from Slack App → Basic Information
"""

import asyncio

import structlog
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from opsagent.config import get_settings

log = structlog.get_logger(__name__)

_bolt_app: AsyncApp | None = None
_handler: AsyncSlackRequestHandler | None = None


def get_bolt_app() -> AsyncApp | None:
    return _bolt_app


def get_slack_handler() -> AsyncSlackRequestHandler | None:
    return _handler


def init_slack_bot(settings=None) -> bool:
    """
    Initialise the Slack Bolt app.
    Returns True if configured, False if signing secret / token are missing.
    """
    global _bolt_app, _handler
    if settings is None:
        settings = get_settings()

    if not settings.slack_bot_token or not settings.slack_signing_secret:
        log.warning(
            "slack_bot_disabled",
            reason="SLACK_BOT_TOKEN or SLACK_SIGNING_SECRET not set",
        )
        return False

    _bolt_app = AsyncApp(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )
    _handler = AsyncSlackRequestHandler(_bolt_app)

    # ── Register action handlers ───────────────────────────────────────────────
    @_bolt_app.action("approve_incident")
    async def handle_approve(ack, body, client):
        await ack()
        await _on_approve(body, client, settings)

    @_bolt_app.action("reject_incident")
    async def handle_reject(ack, body, client):
        await ack()
        await _on_reject(body, client, settings)

    log.info("slack_bot_initialized")
    return True


async def _on_approve(body: dict, client, settings):
    """Called when a user clicks ✅ Approve in Slack."""
    from opsagent.database import get_session_factory, init_db
    from opsagent.repositories import audit_log_repo, draft_repo, incident_repo
    from opsagent.integrations.slack.blocks import build_approved_message

    incident_id = body["actions"][0]["value"]
    user_id = body["user"]["id"]
    user_name = body["user"].get("name", user_id)
    message_ts = body["message"]["ts"]
    channel = body["channel"]["id"]

    log.info("slack_approve_clicked", incident_id=incident_id, user=user_name)

    factory = get_session_factory()
    async with factory() as db:
        incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
        if not incident:
            await client.chat_postEphemeral(
                channel=channel, user=user_id, text=f"Incident `{incident_id}` not found."
            )
            return

        draft = await draft_repo.get_latest_for_incident(db, incident_id)
        if not draft:
            await client.chat_postEphemeral(
                channel=channel, user=user_id, text="No draft found for this incident."
            )
            return

        # Update incident status
        await incident_repo.update_status(db, incident_id, "RESOLVED")
        await audit_log_repo.log_event(
            db,
            incident_id=incident_id,
            event_type="draft_approved",
            actor=f"slack_user:{user_name}",
            payload={"draft_id": draft.id, "via": "slack_button"},
        )

        # Update the Slack message in-place (remove buttons, show resolved state)
        approved_blocks = build_approved_message(incident, draft, approved_by=f"@{user_name}")
        await client.chat_update(
            channel=channel,
            ts=message_ts,
            text=f"✅ Incident {incident.alert_name} approved by @{user_name}",
            blocks=approved_blocks,
        )

    log.info("slack_approve_complete", incident_id=incident_id)


async def _on_reject(body: dict, client, settings):
    """Called when a user clicks ❌ Reject in Slack."""
    from opsagent.database import get_session_factory
    from opsagent.repositories import audit_log_repo, draft_repo, incident_repo
    from opsagent.integrations.slack.blocks import build_rejected_message

    incident_id = body["actions"][0]["value"]
    user_id = body["user"]["id"]
    user_name = body["user"].get("name", user_id)
    message_ts = body["message"]["ts"]
    channel = body["channel"]["id"]

    log.info("slack_reject_clicked", incident_id=incident_id, user=user_name)

    factory = get_session_factory()
    async with factory() as db:
        incident = await incident_repo.get_by_id(db, incident_id, load_drafts=True)
        if not incident:
            return

        draft = await draft_repo.get_latest_for_incident(db, incident_id)

        await incident_repo.update_status(db, incident_id, "REJECTED")
        await audit_log_repo.log_event(
            db,
            incident_id=incident_id,
            event_type="draft_rejected",
            actor=f"slack_user:{user_name}",
            payload={
                "draft_id": draft.id if draft else None,
                "reason": "rejected_via_slack",
                "via": "slack_button",
            },
        )

        rejected_blocks = build_rejected_message(
            incident, rejected_by=f"@{user_name}", reason="manual rejection"
        )
        await client.chat_update(
            channel=channel,
            ts=message_ts,
            text=f"❌ Incident {incident.alert_name} rejected by @{user_name}",
            blocks=rejected_blocks,
        )

    log.info("slack_reject_complete", incident_id=incident_id)
