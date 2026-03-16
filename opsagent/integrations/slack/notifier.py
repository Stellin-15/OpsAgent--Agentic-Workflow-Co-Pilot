"""
Slack notifier — sends incident draft messages to a Slack channel.

This replaces the simple text post from Phase 1 with rich Block Kit messages
including interactive Approve/Reject buttons.

Returns the message timestamp (`ts`) which is stored in the audit_log so the
Slack bot can update the message in-place when a user clicks Approve/Reject.
"""

import structlog
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from opsagent.config import Settings
from opsagent.integrations.slack.blocks import build_incident_draft_message
from opsagent.models.draft import Draft
from opsagent.models.incident import Incident

log = structlog.get_logger(__name__)


class SlackNotifier:
    def __init__(self, settings: Settings) -> None:
        self._channel = settings.slack_channel
        self._client: WebClient | None = None
        if settings.slack_bot_token:
            self._client = WebClient(token=settings.slack_bot_token)

    def is_configured(self) -> bool:
        return self._client is not None and bool(self._channel)

    async def send_incident_draft(
        self,
        incident: Incident,
        draft: Draft,
    ) -> str | None:
        """
        Post a Block Kit message with the AI draft + Approve/Reject buttons.
        Returns the Slack message `ts` (timestamp) so it can be updated later.
        Returns None if Slack is not configured or post fails.
        """
        if not self.is_configured():
            log.warning("slack_not_configured", incident_id=incident.id)
            return None

        blocks = build_incident_draft_message(incident, draft)
        fallback_text = (
            f"OpsAgent: New incident {incident.alert_name} ({incident.severity}). "
            "Review and approve in the OpsAgent dashboard."
        )

        try:
            response = self._client.chat_postMessage(
                channel=self._channel,
                text=fallback_text,    # shown in notifications/unfurls
                blocks=blocks,
            )
            ts = response["ts"]
            log.info(
                "slack_draft_sent",
                incident_id=incident.id,
                channel=self._channel,
                ts=ts,
            )
            return ts
        except SlackApiError as exc:
            log.error(
                "slack_draft_send_failed",
                incident_id=incident.id,
                error=exc.response.get("error"),
            )
            return None

    async def update_message(
        self,
        ts: str,
        blocks: list[dict],
        fallback_text: str = "",
    ) -> bool:
        """Update an existing Slack message in-place (used after approve/reject)."""
        if not self.is_configured():
            return False
        try:
            self._client.chat_update(
                channel=self._channel,
                ts=ts,
                text=fallback_text,
                blocks=blocks,
            )
            return True
        except SlackApiError as exc:
            log.error("slack_update_failed", ts=ts, error=exc.response.get("error"))
            return False
