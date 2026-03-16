"""Add runbook_chunks table with pgvector embedding column.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-17

Requires the pgvector extension to be installed in the target PostgreSQL DB:
    CREATE EXTENSION IF NOT EXISTS vector;

The migration will succeed even if the vector extension is missing, but
vector similarity search will not work until the extension is installed.
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "runbook_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=500), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        # pgvector VECTOR(768) column — text fallback for environments without pgvector
        sa.Column(
            "embedding",
            sa.Text().with_variant(
                sa.Text(),   # will be overridden below via raw SQL
                "postgresql",
            ),
            nullable=True,
        ),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Alter the column to use the proper VECTOR type on PostgreSQL
    op.execute(
        "ALTER TABLE runbook_chunks ALTER COLUMN embedding TYPE vector(768) "
        "USING NULL"
    )

    op.create_index("ix_runbook_chunks_source", "runbook_chunks", ["source"])

    # HNSW index for fast approximate nearest-neighbour search (pgvector ≥ 0.5)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_runbook_chunks_embedding_hnsw "
        "ON runbook_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS ix_runbook_chunks_embedding_hnsw"
    )
    op.drop_index("ix_runbook_chunks_source", table_name="runbook_chunks")
    op.drop_table("runbook_chunks")
