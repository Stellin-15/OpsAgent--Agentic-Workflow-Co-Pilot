"""
Pydantic schemas for normalising incoming alert webhooks.

Supported sources (Phase 1 → 2):
  - Prometheus AlertManager  (version 4 webhook format)
  - Manual / generic         (simple JSON body for testing)

Phase 2 will add: Grafana, PagerDuty.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Prometheus AlertManager ────────────────────────────────────────────────────

class AlertManagerAlert(BaseModel):
    """A single alert inside an AlertManager webhook payload."""

    status: str                          # "firing" | "resolved"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: datetime | None = None
    endsAt: datetime | None = None
    generatorURL: str = ""
    fingerprint: str = ""


class AlertManagerWebhook(BaseModel):
    """Full Prometheus AlertManager v4 webhook body."""

    version: str = "4"
    groupKey: str = ""
    status: str                          # "firing" | "resolved"
    receiver: str = ""
    groupLabels: dict[str, str] = Field(default_factory=dict)
    commonLabels: dict[str, str] = Field(default_factory=dict)
    commonAnnotations: dict[str, str] = Field(default_factory=dict)
    externalURL: str = ""
    alerts: list[AlertManagerAlert] = Field(default_factory=list)


# ── Generic / manual alert ─────────────────────────────────────────────────────

class ManualAlert(BaseModel):
    """Simple payload for testing or custom webhook sources."""

    alert_name: str
    severity: str = "warning"
    description: str = ""
    labels: dict[str, Any] = Field(default_factory=dict)


# ── Normalised incident data ───────────────────────────────────────────────────

class NormalisedAlert(BaseModel):
    """
    Internal representation after normalising any alert source.
    This is what the service layer works with — not the raw webhook body.
    """

    alert_name: str
    severity: str
    description: str
    labels: dict[str, Any]
    source: str                          # "alertmanager" | "grafana" | "pagerduty" | "manual"
    fired_at: datetime | None = None
