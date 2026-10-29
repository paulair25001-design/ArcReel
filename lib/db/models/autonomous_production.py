"""Durable autonomous goals and delivery leases; project tasks own production."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from lib.db.base import Base, UserOwnedMixin


class AutonomousProduction(UserOwnedMixin, Base):
    __tablename__ = "autonomous_production"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_name: Mapped[str] = mapped_column(String, nullable=False)
    episode_id: Mapped[int] = mapped_column(Integer, nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    client_key: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String)
    state: Mapped[str] = mapped_column(String, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    lease_owner: Mapped[str | None] = mapped_column(String)
    lease_until: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "project_name", "client_key", name="uq_autonomous_production_client_key"),
        Index(
            "uq_autonomous_production_active",
            "user_id",
            "project_name",
            unique=True,
            sqlite_where=text("state IN ('pending', 'running', 'waiting', 'stopping')"),
            postgresql_where=text("state IN ('pending', 'running', 'waiting', 'stopping')"),
        ),
        Index("idx_autonomous_production_state", "state"),
    )
