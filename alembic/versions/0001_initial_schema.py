"""Initial schema: incidents, drafts, audit_log, action_log

Revision ID: 0001
Revises:
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── incidents ──────────────────────────────────────────────────────────────
    op.create_table(
        "incidents",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column("alert_name", sa.String(500), nullable=False),
        sa.Column("labels", JSONB, nullable=False, server_default="{}"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="FIRING"),
        sa.Column("severity", sa.String(50), nullable=False, server_default="warning"),
        sa.Column("source", sa.String(100), nullable=False, server_default="alertmanager"),
        sa.Column(
            "fired_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_incidents_alert_name", "incidents", ["alert_name"])
    op.create_index("ix_incidents_status", "incidents", ["status"])
    op.create_index("ix_incidents_fired_at", "incidents", ["fired_at"])

    # ── drafts ────────────────────────────────────────────────────────────────
    op.create_table(
        "drafts",
        sa.Column("id", sa.String(26), primary_key=True),
        sa.Column(
            "incident_id",
            sa.String(26),
            sa.ForeignKey("incidents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("model_used", sa.String(100), nullable=False),
        sa.Column("retrieval_score", sa.Float, nullable=True),
        sa.Column("ragas_scores", JSONB, nullable=True),
        sa.Column("retrieval_chunks", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_drafts_incident_id", "drafts", ["incident_id"])

    # ── audit_log (append-only) ───────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "incident_id",
            sa.String(26),
            sa.ForeignKey("incidents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("actor", sa.String(200), nullable=False, server_default="system"),
        sa.Column("payload", JSONB, nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_audit_log_incident_id", "audit_log", ["incident_id"])
    op.create_index("ix_audit_log_event_type", "audit_log", ["event_type"])
    op.create_index("ix_audit_log_occurred_at", "audit_log", ["occurred_at"])

    # ── action_log (append-only) ──────────────────────────────────────────────
    op.create_table(
        "action_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "incident_id",
            sa.String(26),
            sa.ForeignKey("incidents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action_type", sa.String(100), nullable=False),
        sa.Column("command", sa.Text, nullable=False),
        sa.Column("stdout", sa.Text, nullable=True),
        sa.Column("exit_code", sa.Integer, nullable=True),
        sa.Column("approved_by", sa.String(200), nullable=False, server_default="system"),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_action_log_incident_id", "action_log", ["incident_id"])
    op.create_index("ix_action_log_executed_at", "action_log", ["executed_at"])


def downgrade() -> None:
    op.drop_table("action_log")
    op.drop_table("audit_log")
    op.drop_table("drafts")
    op.drop_table("incidents")
