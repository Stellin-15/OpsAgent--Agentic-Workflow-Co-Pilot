"""
Pydantic v2 schemas for Incident API responses and request bodies.
These are the shapes that leave/enter the HTTP layer — separate from ORM models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DraftOut(BaseModel):
    id: str
    incident_id: str
    content: str
    model_used: str
    retrieval_score: float | None
    ragas_scores: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class IncidentOut(BaseModel):
    id: str
    alert_name: str
    labels: dict[str, Any]
    description: str | None
    status: str
    severity: str
    source: str
    fired_at: datetime
    resolved_at: datetime | None
    drafts: list[DraftOut] = []

    model_config = {"from_attributes": True}


class IncidentListOut(BaseModel):
    items: list[IncidentOut]
    total: int


class RejectBody(BaseModel):
    reason: str = "other"
    notes: str = ""
