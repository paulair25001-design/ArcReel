"""Autonomous goal migration preserves scoped idempotency and active uniqueness."""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from alembic import command


def test_upgrade_enforces_user_project_scope_and_downgrade_is_local(alembic_cfg, migration_revisions):
    cfg, db_path = alembic_cfg
    revision, parent = migration_revisions("*_add_autonomous_production.py")
    command.upgrade(cfg, revision)
    engine = sa.create_engine(f"sqlite:///{db_path}")
    try:
        inspect = sa.inspect(engine)
        assert "autonomous_production" in inspect.get_table_names()
        columns = {column["name"] for column in inspect.get_columns("autonomous_production")}
        assert {
            "user_id",
            "project_name",
            "episode_id",
            "objective",
            "client_key",
            "session_id",
            "state",
            "revision",
            "lease_owner",
            "lease_until",
            "payload_json",
        } <= columns
        indexes = {index["name"]: index for index in inspect.get_indexes("autonomous_production")}
        assert indexes["uq_autonomous_production_active"]["unique"]
        assert indexes["ix_autonomous_production_user_id"]["column_names"] == ["user_id"]
        foreign_keys = inspect.get_foreign_keys("autonomous_production")
        assert foreign_keys[0]["referred_table"] == "users"
        assert foreign_keys[0].get("options", {}).get("ondelete") is None
        insert = sa.text(
            "INSERT INTO autonomous_production (id,user_id,project_name,episode_id,objective,client_key,state,"
            "payload_json,created_at,updated_at) VALUES (:id,:user,:project,2,'Finish episode',:key,:state,"
            ":payload,'2026-10-29','2026-10-29')"
        )
        base = {
            "user": "default",
            "project": "project-one",
            "key": "one",
            "state": "pending",
            "payload": json.dumps({}),
        }
        with engine.begin() as conn:
            conn.execute(insert, {**base, "id": "a" * 32})
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(insert, {**base, "id": "b" * 32, "key": "two"})
        with engine.begin() as conn:
            conn.execute(insert, {**base, "id": "c" * 32, "user": "u1"})
            conn.execute(sa.text("UPDATE autonomous_production SET state='completed' WHERE id=:id"), {"id": "a" * 32})
            conn.execute(insert, {**base, "id": "d" * 32, "key": "two"})
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(insert, {**base, "id": "e" * 32, "state": "completed"})
    finally:
        engine.dispose()
    command.downgrade(cfg, parent)
    engine = sa.create_engine(f"sqlite:///{db_path}")
    try:
        tables = sa.inspect(engine).get_table_names()
        assert "autonomous_production" not in tables
        assert "tasks" in tables
        assert "users" in tables
    finally:
        engine.dispose()
