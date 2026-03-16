"""
PagerDuty webhook v3 adapter.

Handles PagerDuty's webhook v3 event format for `incident.triggered` events:
{
  "event": {
    "event_type": "incident.triggered",
    "data": {
      "id": "Q2JTSNZYLBPSNQ",
      "summary": "CPU Usage Alert",
      "status": "triggered",
      "urgency": "high",
      "service": {"summary": "Production API"},
      "body": {
        "cef_details": {
          "details": {"severity": "critical", "description": "CPU at 95%"}
        }
      }
    }
  }
}

Configure PagerDuty to send webhook v3 to:
  http://opsagent:8000/api/alerts/webhook/pagerduty

Required: PagerDuty → Integrations → Webhooks → Add Webhook Subscription
"""

from opsagent.schemas.alert import NormalisedAlert


class PagerDutyAdapter:
    SOURCE = "pagerduty"

    def can_handle(self, body: dict) -> bool:
        return "event" in body and "event_type" in body.get("event", {})

    def normalise(self, body: dict) -> list[NormalisedAlert]:
        event = body.get("event", {})
        event_type = event.get("event_type", "")

        if event_type not in ("incident.triggered", "incident.acknowledged"):
            return []  # only care about new/acknowledged alerts

        data = event.get("data", {})
        incident_summary = data.get("summary", "PagerDutyAlert")
        urgency = data.get("urgency", "low")
        service_name = data.get("service", {}).get("summary", "unknown")

        # Map PagerDuty urgency to our severity
        severity = "critical" if urgency == "high" else "warning"

        # Extract description from body if available
        details = (
            data.get("body", {})
            .get("cef_details", {})
            .get("details", {})
        )
        description = details.get("description", incident_summary)

        labels = {
            "alertname": incident_summary,
            "severity": severity,
            "service": service_name,
            "pagerduty_id": data.get("id", ""),
            "urgency": urgency,
        }

        return [
            NormalisedAlert(
                alert_name=incident_summary,
                severity=severity,
                description=description,
                labels=labels,
                source=self.SOURCE,
            )
        ]
