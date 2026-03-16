"""
Slack notification service.

Wraps the slack-sdk WebClient with structured logging and error handling.
Phase 2 will replace the simple text post with rich Block Kit messages
and interactive Approve/Reject buttons.
"""

import structlog
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from opsagent.config import Settings

log = structlog.get_logger(__name__)


class SlackService:
    def __init__(self, settings: Settings) -> None:
        self._channel = settings.slack_channel
        self._client: WebClient | None = None
        if settings.slack_bot_token:
            self._client = WebClient(token=settings.slack_bot_token)

    def is_configured(self) -> bool:
        return self._client is not None and bool(self._channel)

    async def post_draft(self, incident_id: str, alert_name: str, draft: str) -> bool:
        """Post an approved draft to the configured Slack channel. Returns True on success."""
        if not self.is_configured():
            log.warning("slack_not_configured", incident_id=incident_id)
            return False

        message = f"*✅ OpsAgent — Incident Resolved: {alert_name}*\n\n{draft}"
        try:
            self._client.chat_postMessage(channel=self._channel, text=message)
            log.info("slack_message_posted", incident_id=incident_id, channel=self._channel)
            return True
        except SlackApiError as exc:
            log.error(
                "slack_post_failed",
                incident_id=incident_id,
                error=exc.response.get("error"),
            )
            return False
