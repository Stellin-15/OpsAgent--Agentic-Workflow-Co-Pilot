from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opsagent.database import Base


class ActionLog(Base):
    """
    Append-only log of every command executed by the execution engine.
    Populated in Phase 3 when the safe execution engine is added.

    Columns:
        action_type   - e.g. "kubectl", "aws_cli", "custom_script"
        command       - the exact command string that was run
        stdout        - captured stdout/stderr output
        exit_code     - 0 = success, non-zero = failure
        approved_by   - the user ID or "system" that approved execution
        executed_at   - when the command ran
    """

    __tablename__ = "action_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    stdout: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approved_by: Mapped[str] = mapped_column(String(200), nullable=False, default="system")

    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    incident: Mapped["Incident"] = relationship(  # noqa: F821
        "Incident", back_populates="action_logs"
    )
