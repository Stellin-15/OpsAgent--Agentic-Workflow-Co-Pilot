"""
Slack interactive actions endpoint.

When a user clicks ✅ Approve or ❌ Reject in a Slack Block Kit message,
Slack sends an HTTP POST to this endpoint. Slack Bolt handles verification
(HMAC signature check using SLACK_SIGNING_SECRET) and routing to the
correct action handler registered in opsagent/integrations/slack/bot.py.

Setup:
  Slack App → Interactivity & Shortcuts → Request URL:
    https://your-domain.com/api/slack/actions
"""

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter(prefix="/slack", tags=["slack"])


@router.post("/actions", summary="Slack interactive component actions")
async def slack_actions(request: Request):
    """
    Receives button-click payloads from Slack Block Kit messages.
    Slack Bolt handles signature verification and dispatches to the
    correct action handler (approve_incident / reject_incident).
    """
    from opsagent.integrations.slack.bot import get_slack_handler

    handler = get_slack_handler()
    if handler is None:
        # Slack bot not configured — return 200 so Slack doesn't retry
        return Response(content="Slack bot not configured", status_code=200)

    return await handler.handle(request)
