"""RunbookChunk ORM model — pgvector-backed runbook store (Phase 4).

Stores document chunks with their dense embeddings so we can perform
cosine similarity search directly in PostgreSQL via the pgvector extension.

The ``embedding`` column uses the ``VECTOR`` type from pgvector-sqlalchemy.
Dimension is 768 to match Google text-embedding-004.
"""

from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector
    _VECTOR_TYPE = Vector(768)
except ImportError:
    # Graceful degradation: pgvector not installed — store as text (dev/test).
    from sqlalchemy import Text as _TextFallback
    _VECTOR_TYPE = _TextFallback()  # type: ignore[assignment]

from opsagent.database import Base


class RunbookChunk(Base):
    """
    One overlapping text chunk from a runbook document.

    Columns:
        content      - the raw text of this chunk
        source       - relative path of the source runbook file
        chunk_index  - position within the source document (0-based)
        embedding    - dense vector (768-dim, pgvector VECTOR type)
        indexed_at   - when this row was last written
    """

    __tablename__ = "runbook_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[object] = mapped_column(_VECTOR_TYPE, nullable=True)

    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
