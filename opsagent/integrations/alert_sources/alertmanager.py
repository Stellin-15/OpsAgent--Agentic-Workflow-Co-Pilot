"""
Prometheus AlertManager v4 webhook adapter.

AlertManager fires webhook payloads like:
{
  "version": "4",
  "status": "firing",
  "alerts": [
    {
      "status": "firing",
      "labels": {"alertname": "HighCPU", "severity": "critical", "instance": "web-01"},
      "annotations": {"summary": "CPU at 95%", "description": "..."},
      "startsAt": "2026-03-17T10:00:00Z"
    }
  ]
}

Configuration (alertmanager.yml):
  receivers:
    - name: opsagent
      webhook_configs:
        - url: http://opsagent:8000/api/alerts/webhook/alertmanager
          send_resolved: false
"""

from datetime import datetime, timezone

from opsagent.schemas.alert import NormalisedAlert


class AlertManagerAdapter:
    SOURCE = "alertmanager"

    def can_handle(self, body: dict) -> bool:
        return body.get("version") == "4" and "alerts" in body

    def normalise(self, body: dict) -> list[NormalisedAlert]:
        common_labels: dict = body.get("commonLabels", {})
        results: list[NormalisedAlert] = []

        for alert in body.get("alerts", []):
            if alert.get("status") == "resolved":
                continue  # skip resolved — handle in Phase 3

            labels = {**common_labels, **alert.get("labels", {})}
            annotations = {
                **body.get("commonAnnotations", {}),
                **alert.get("annotations", {}),
            }

            alert_name = labels.get("alertname", "UnknownAlert")
            severity = labels.get("severity", "warning")
            description = annotations.get("description") or annotations.get("summary", "")

            fired_at = None
            if starts_at := alert.get("startsAt"):
                try:
                    fired_at = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    fired_at = datetime.now(timezone.utc)

            results.append(
                NormalisedAlert(
                    alert_name=alert_name,
                    severity=severity,
                    description=description,
                    labels=labels,
                    source=self.SOURCE,
                    fired_at=fired_at,
                )
            )

        return results
