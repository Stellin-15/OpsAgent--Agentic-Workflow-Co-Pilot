"""
Grafana webhook adapter.

Grafana v9+ (Unified Alerting) uses AlertManager-compatible format — handled
by the AlertManagerAdapter automatically.

This adapter handles the legacy Grafana v8 and earlier webhook format:
{
  "title": "Alert Title",
  "state": "alerting",
  "ruleName": "CPU Usage High",
  "message": "CPU has been above threshold",
  "evalMatches": [{"metric": "cpu_usage", "value": 95.2}],
  "tags": {"severity": "critical", "team": "infra"},
  "ruleUrl": "http://grafana:3000/d/..."
}

Grafana webhook configuration (Grafana → Alerting → Contact points → Webhook):
  URL: http://opsagent:8000/api/alerts/webhook/grafana
"""

from opsagent.schemas.alert import NormalisedAlert


class GrafanaAdapter:
    SOURCE = "grafana"

    def can_handle(self, body: dict) -> bool:
        # Legacy Grafana format has "ruleName" or "title" + "state" but no "version"
        return "state" in body and ("ruleName" in body or "title" in body) and "version" not in body

    def normalise(self, body: dict) -> list[NormalisedAlert]:
        state = body.get("state", "")
        if state not in ("alerting", "pending"):
            return []  # 'ok' = resolved — skip for now

        rule_name = body.get("ruleName") or body.get("title", "GrafanaAlert")
        message = body.get("message", "")
        tags: dict = body.get("tags", {})
        severity = tags.get("severity", "warning")

        # Build labels from tags + eval matches
        labels: dict[str, str] = {k: str(v) for k, v in tags.items()}
        labels["alertname"] = rule_name
        labels["severity"] = severity

        # Include metric values from evalMatches as context
        eval_matches = body.get("evalMatches", [])
        if eval_matches:
            metrics = ", ".join(
                f"{m.get('metric', '?')}={m.get('value', '?')}"
                for m in eval_matches[:5]  # cap at 5
            )
            description = f"{message}\nMetrics: {metrics}" if message else f"Metrics: {metrics}"
        else:
            description = message

        return [
            NormalisedAlert(
                alert_name=rule_name,
                severity=severity,
                description=description,
                labels=labels,
                source=self.SOURCE,
            )
        ]
