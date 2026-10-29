"""add autonomous production goals and delivery leases

Revision ID: a3e794c5d821
Revises: b5d1e8f3a627
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a3e794c5d821"
down_revision: str | Sequence[str] | None = "b5d1e8f3a627"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "autonomous_production",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("users.id"),
            nullable=False,
            server_default="default",
        ),
        sa.Column("project_name", sa.String(), nullable=False),
        sa.Column("episode_id", sa.Integer(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("client_key", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(), nullable=True),
        sa.Column("lease_until", sa.Float(), nullable=False, server_default="0"),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "project_name", "client_key", name="uq_autonomous_production_client_key"),
    )
    op.create_index(
        "uq_autonomous_production_active",
        "autonomous_production",
        ["user_id", "project_name"],
        unique=True,
        sqlite_where=sa.text("state IN ('pending', 'running', 'waiting', 'stopping')"),
        postgresql_where=sa.text("state IN ('pending', 'running', 'waiting', 'stopping')"),
    )
    op.create_index("idx_autonomous_production_state", "autonomous_production", ["state"])
    op.create_index("ix_autonomous_production_user_id", "autonomous_production", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_autonomous_production_user_id", table_name="autonomous_production")
    op.drop_index("idx_autonomous_production_state", table_name="autonomous_production")
    op.drop_index("uq_autonomous_production_active", table_name="autonomous_production")
    op.drop_table("autonomous_production")
