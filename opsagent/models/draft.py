from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opsagent.database import Base


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    incident_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # The AI-generated response text
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Which model generated this draft (e.g. "gemini-flash-latest")
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)

    # Average retrieval similarity score (0.0 – 1.0) from FAISS/pgvector
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Phase 4: RAGAS evaluation scores stored as JSON
    # {"context_precision": 0.82, "faithfulness": 0.91, "answer_relevance": 0.78}
    ragas_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # The chunks that were retrieved and used as context
    retrieval_chunks: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    incident: Mapped["Incident"] = relationship(  # noqa: F821
        "Incident", back_populates="drafts"
    )

    def __repr__(self) -> str:
        return f"<Draft id={self.id} incident={self.incident_id} model={self.model_used}>"
