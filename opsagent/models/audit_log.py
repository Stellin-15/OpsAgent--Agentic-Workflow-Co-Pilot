from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opsagent.database import Base


class AuditLog(Base):
    """
    Append-only log of every state transition.
    Never UPDATE or DELETE rows in this table.

    Examples:
        event_type="incident_created", actor="alertmanager"
        event_type="status_changed",   actor="system",      payload={"from": "PROCESSING", "to": "DRAFT_READY"}
        event_type="draft_approved",   actor="user:alice"
        event_type="draft_rejected",   actor="user:bob",    payload={"reason": "hallucination"}
        event_type="action_executed",  actor="system",      payload={"command": "kubectl rollout restart ..."}
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Who or what triggered this event: "system", "user:<id>", "alertmanager", etc.
    actor: Mapped[str] = mapped_column(String(200), nullable=False, default="system")

    # Arbitrary structured context for the event
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    incident: Mapped["Incident"] = relationship(  # noqa: F821
        "Incident", back_populates="audit_logs"
    )
