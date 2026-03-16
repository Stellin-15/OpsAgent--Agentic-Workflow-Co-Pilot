import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opsagent.database import Base


class IncidentStatus(str, enum.Enum):
    FIRING = "FIRING"
    PROCESSING = "PROCESSING"
    DRAFT_READY = "DRAFT_READY"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class IncidentSeverity(str, enum.Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Incident(Base):
    __tablename__ = "incidents"

    # ULID: sortable, URL-safe, collision-resistant (generated in Python, not DB)
    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    alert_name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    # Raw labels from the alert source (alertname, instance, severity, etc.)
    labels: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Human-readable description / annotations from the alert
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=IncidentStatus.FIRING, index=True
    )
    severity: Mapped[str] = mapped_column(
        String(50), nullable=False, default=IncidentSeverity.WARNING
    )

    # Which system sent this alert (alertmanager, grafana, pagerduty, manual)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="alertmanager")

    fired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships (populated lazily — use selectinload in queries if needed)
    drafts: Mapped[list["Draft"]] = relationship(  # noqa: F821
        "Draft", back_populates="incident", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(  # noqa: F821
        "AuditLog", back_populates="incident", cascade="all, delete-orphan"
    )
    action_logs: Mapped[list["ActionLog"]] = relationship(  # noqa: F821
        "ActionLog", back_populates="incident", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Incident id={self.id} alert={self.alert_name} status={self.status}>"
