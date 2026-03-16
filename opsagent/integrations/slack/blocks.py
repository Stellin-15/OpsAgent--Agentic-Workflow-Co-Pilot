"""
Slack Block Kit message builders.

Block Kit is Slack's structured message format — richer than plain text,
supports interactive elements (buttons, dropdowns, modals).

Reference: https://api.slack.com/block-kit

The incident draft message has three sections:
  1. Header: alert name + severity badge
  2. Context: source, fired_at, incident ID
  3. Draft content (truncated at 3000 chars — Slack limit)
  4. Action buttons: ✅ Approve | ❌ Reject
"""

from opsagent.models.draft import Draft
from opsagent.models.incident import Incident

# Slack text block max length
_MAX_TEXT = 3000
_SEVERITY_EMOJI = {
    "critical": "🔴",
    "warning": "🟡",
    "info": "🟢",
}


def build_incident_draft_message(incident: Incident, draft: Draft) -> list[dict]:
    """
    Return a list of Slack Block Kit blocks for a new incident draft.
    These blocks are passed as the `blocks` parameter to chat.postMessage.
    """
    severity_emoji = _SEVERITY_EMOJI.get(incident.severity, "⚪")
    draft_text = draft.content
    if len(draft_text) > _MAX_TEXT:
        draft_text = draft_text[:_MAX_TEXT - 3] + "..."

    fired_at_str = (
        incident.fired_at.strftime("%Y-%m-%d %H:%M UTC")
        if incident.fired_at
        else "unknown"
    )

    blocks = [
        # ── Header ────────────────────────────────────────────────────────────
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity_emoji} OpsAgent Alert: {incident.alert_name}",
                "emoji": True,
            },
        },
        # ── Context row ───────────────────────────────────────────────────────
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Severity:* `{incident.severity}`  "
                        f"*Source:* `{incident.source}`  "
                        f"*Fired:* {fired_at_str}  "
                        f"*ID:* `{incident.id}`"
                    ),
                }
            ],
        },
        {"type": "divider"},
        # ── AI-generated response plan ─────────────────────────────────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🤖 AI Response Plan:*\n\n{draft_text}",
            },
        },
        {"type": "divider"},
        # ── Action buttons ────────────────────────────────────────────────────
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve & Post", "emoji": True},
                    "style": "primary",
                    "action_id": "approve_incident",
                    "value": incident.id,
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Approve this response?"},
                        "text": {
                            "type": "mrkdwn",
                            "text": "This will mark the incident as resolved and post the response to the channel.",
                        },
                        "confirm": {"type": "plain_text", "text": "Yes, approve"},
                        "deny": {"type": "plain_text", "text": "Cancel"},
                    },
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject", "emoji": True},
                    "style": "danger",
                    "action_id": "reject_incident",
                    "value": incident.id,
                },
            ],
        },
    ]
    return blocks


def build_approved_message(incident: Incident, draft: Draft, approved_by: str) -> list[dict]:
    """Replacement blocks after a draft is approved — removes buttons."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"✅ *Incident `{incident.alert_name}` resolved* by {approved_by}\n\n"
                    f"{draft.content}"
                ),
            },
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Incident ID: `{incident.id}`"}],
        },
    ]


def build_rejected_message(incident: Incident, rejected_by: str, reason: str) -> list[dict]:
    """Replacement blocks after a draft is rejected — removes buttons."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"❌ *Draft rejected* by {rejected_by} — reason: `{reason}`\n"
                    f"Incident `{incident.alert_name}` (`{incident.id}`) needs manual attention."
                ),
            },
        },
    ]
