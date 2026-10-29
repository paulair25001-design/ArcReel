"""Async goal and delivery envelopes in ArcReel's authoritative runtime database.

Session events and project artifacts retain their existing owners. This store
only binds a user's objective to its session, delivery state, and driver lease.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, case, or_, select
from sqlalchemy import update as sql_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lib.db import safe_session_factory
from lib.db.models.autonomous_production import AutonomousProduction
from lib.infra.data_root_layout import PROJECT_NAME_PATTERN

ACTIVE_STATES = frozenset({"pending", "running", "waiting", "stopping"})
TERMINAL_STATES = frozenset({"completed", "stopped", "attention"})
_RUN_ID = re.compile(r"^[a-f0-9]{32}$")
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,200}")
_PAYLOAD_DEFAULTS: dict[str, Any] = {
    "pending_delivery": None,
    "last_progress_digest": None,
    "unchanged_turns": 0,
    "dispatch_count": 0,
    "last_error": None,
}
_MUTABLE_FIELDS = frozenset({"state", "session_id", "lease_owner", "lease_until", *_PAYLOAD_DEFAULTS})
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _project_name(value: object) -> str:
    name = _nonempty(value, "project_name")
    if not PROJECT_NAME_PATTERN.fullmatch(name):
        raise ValueError("project_name must be an ArcReel project slug")
    return name


def _run_id(value: object) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value):
        raise ValueError("Invalid autonomous goal identity")
    return value


def _iso(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat() if value.tzinfo is None else value.isoformat()


def _record(row: AutonomousProduction) -> dict[str, Any]:
    return {
        "id": row.id,
        "project_name": row.project_name,
        "episode_id": row.episode_id,
        "objective": row.objective,
        "user_id": row.user_id,
        "client_key": row.client_key,
        "session_id": row.session_id,
        "state": row.state,
        "revision": row.revision,
        "lease_owner": row.lease_owner,
        "lease_until": row.lease_until,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        **json.loads(row.payload_json),
    }


def _validated_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
    if not changes.keys() <= _MUTABLE_FIELDS:
        raise ValueError("Only autonomous goal state and delivery fields may be updated")
    result = dict(changes)
    if "state" in result and (
        not isinstance(result["state"], str) or result["state"] not in ACTIVE_STATES | TERMINAL_STATES
    ):
        raise ValueError("Unknown autonomous goal state")
    for field in ("session_id", "last_progress_digest", "last_error"):
        if field in result and result[field] is not None:
            _nonempty(result[field], field)
    for field in ("unchanged_turns", "dispatch_count"):
        if field in result and (type(result[field]) is not int or result[field] < 0):
            raise ValueError(f"{field} must be a nonnegative integer")
    if result.get("pending_delivery") is not None and not isinstance(result["pending_delivery"], dict):
        raise ValueError("pending_delivery must be an object or null")
    if ("lease_owner" in result or "lease_until" in result) and (
        result.get("lease_owner", "missing") is not None or result.get("lease_until") != 0
    ):
        raise ValueError("Only paired lease_owner=null and lease_until=0 may release a lease")
    try:
        return json.loads(_json(result))
    except (TypeError, ValueError) as exc:
        raise ValueError("Goal changes must be finite JSON values") from exc


class AutonomyStore:
    def __init__(
        self, *, session_factory: SessionFactory = safe_session_factory, clock: Callable[[], float] = time.time
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def _now(self) -> tuple[float, datetime]:
        epoch = self._clock()
        return epoch, datetime.fromtimestamp(epoch, UTC)

    @staticmethod
    def _same_objective(
        row: AutonomousProduction, episode_id: int, objective: str, session_id: str | None = None
    ) -> dict[str, Any]:
        if row.episode_id != episode_id or row.objective != objective:
            raise ValueError("The client key already belongs to a different objective")
        if session_id is not None and row.session_id != session_id:
            raise ValueError("The client key already belongs to a different session")
        return _record(row)

    async def create(
        self,
        project_name: str,
        episode_id: int,
        objective: str,
        user_id: str,
        client_key: str,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        project_name = _project_name(project_name)
        _nonempty(user_id, "user_id")
        _nonempty(objective, "objective")
        _nonempty(client_key, "client_key")
        if session_id is not None and _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("Invalid Hermes session identity")
        if type(episode_id) is not int or episode_id < 1:
            raise ValueError("episode_id must be a positive integer")
        identity = and_(AutonomousProduction.user_id == user_id, AutonomousProduction.project_name == project_name)
        keyed = select(AutonomousProduction).where(identity, AutonomousProduction.client_key == client_key)
        async with self._session_factory() as session:
            existing = (await session.execute(keyed)).scalar_one_or_none()
            if existing is not None:
                return self._same_objective(existing, episode_id, objective, session_id)
            _, now = self._now()
            row = AutonomousProduction(
                id=uuid.uuid4().hex,
                project_name=project_name,
                episode_id=episode_id,
                objective=objective,
                user_id=user_id,
                client_key=client_key,
                session_id=session_id,
                state="pending",
                revision=0,
                lease_owner=None,
                lease_until=0,
                payload_json=_json(_PAYLOAD_DEFAULTS),
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = (await session.execute(keyed)).scalar_one_or_none()
                if existing is not None:
                    return self._same_objective(existing, episode_id, objective, session_id)
                active = (
                    await session.execute(
                        select(AutonomousProduction.id).where(identity, AutonomousProduction.state.in_(ACTIVE_STATES))
                    )
                ).scalar_one_or_none()
                if active is not None:
                    raise ValueError("This user already has an active goal for the project") from None
                raise
            return _record(row)

    async def get(self, run_id: str, user_id: str) -> dict[str, Any] | None:
        _run_id(run_id)
        _nonempty(user_id, "user_id")
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AutonomousProduction).where(
                        AutonomousProduction.id == run_id, AutonomousProduction.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            return _record(row) if row is not None else None

    async def get_latest(self, project_name: str, user_id: str) -> dict[str, Any] | None:
        project_name = _project_name(project_name)
        _nonempty(user_id, "user_id")
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(AutonomousProduction)
                    .where(AutonomousProduction.project_name == project_name, AutonomousProduction.user_id == user_id)
                    .order_by(AutonomousProduction.created_at.desc(), AutonomousProduction.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return _record(row) if row is not None else None

    async def list_active(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Without a user filter, only the internal scheduler may scan all goals."""
        query = select(AutonomousProduction).where(AutonomousProduction.state.in_(ACTIVE_STATES))
        if user_id is not None:
            _nonempty(user_id, "user_id")
            query = query.where(AutonomousProduction.user_id == user_id)
        async with self._session_factory() as session:
            return [
                _record(row)
                for row in (await session.execute(query.order_by(AutonomousProduction.created_at))).scalars()
            ]

    async def claim(self, run_id: str, owner_id: str, ttl_seconds: float, *, user_id: str) -> dict[str, Any] | None:
        _run_id(run_id)
        _nonempty(user_id, "user_id")
        _nonempty(owner_id, "owner_id")
        if isinstance(ttl_seconds, bool) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("Lease TTL must be a positive finite number")
        epoch, now = self._now()
        statement = (
            sql_update(AutonomousProduction)
            .where(
                AutonomousProduction.id == run_id,
                AutonomousProduction.user_id == user_id,
                AutonomousProduction.state.in_(ACTIVE_STATES),
                or_(
                    AutonomousProduction.lease_owner.is_(None),
                    AutonomousProduction.lease_owner == owner_id,
                    AutonomousProduction.lease_until <= epoch,
                ),
            )
            .values(
                lease_owner=owner_id,
                lease_until=epoch + ttl_seconds,
                updated_at=now,
                revision=AutonomousProduction.revision + 1,
            )
            .returning(AutonomousProduction)
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
            await session.commit()
            return _record(row) if row is not None else None

    async def update(self, run_id: str, owner_id: str, changes: Mapping[str, Any], *, user_id: str) -> dict[str, Any]:
        _run_id(run_id)
        _nonempty(user_id, "user_id")
        _nonempty(owner_id, "owner_id")
        changes = _validated_changes(changes)
        epoch, now = self._now()
        async with self._session_factory() as session:
            current = (
                await session.execute(
                    select(AutonomousProduction).where(
                        AutonomousProduction.id == run_id,
                        AutonomousProduction.user_id == user_id,
                        AutonomousProduction.lease_owner == owner_id,
                        AutonomousProduction.lease_until > epoch,
                        AutonomousProduction.state.in_(ACTIVE_STATES),
                    )
                )
            ).scalar_one_or_none()
            if current is None:
                raise ValueError("The autonomous goal lease is not owned by this driver")
            if current.state == "stopping" and changes.get("state", "stopping") not in {
                "stopping",
                "stopped",
                "attention",
            }:
                raise ValueError("A stopping autonomous goal cannot resume production")
            payload = {
                **json.loads(current.payload_json),
                **{k: v for k, v in changes.items() if k in _PAYLOAD_DEFAULTS},
            }
            values = {k: v for k, v in changes.items() if k not in _PAYLOAD_DEFAULTS}
            if values.get("state") in TERMINAL_STATES:
                values.update(lease_owner=None, lease_until=0)
            values.update(payload_json=_json(payload), updated_at=now, revision=current.revision + 1)
            row = (
                await session.execute(
                    sql_update(AutonomousProduction)
                    .where(
                        AutonomousProduction.id == run_id,
                        AutonomousProduction.user_id == user_id,
                        AutonomousProduction.lease_owner == owner_id,
                        AutonomousProduction.lease_until > epoch,
                        AutonomousProduction.revision == current.revision,
                        AutonomousProduction.state == current.state,
                    )
                    .values(**values)
                    .returning(AutonomousProduction)
                    .execution_options(synchronize_session="fetch")
                )
            ).scalar_one_or_none()
            if row is None:
                raise ValueError("The autonomous goal changed before this driver update")
            await session.commit()
            return _record(row)

    async def retry(self, run_id: str, user_id: str, expected_revision: int) -> dict[str, Any]:
        """Explicitly recover an attention goal, retaining its exact SDK session.

        The observed terminal revision is the retry identity. A lost response can
        be reconciled without starting another attempt or reviving a stopped goal.
        """
        _run_id(run_id)
        _nonempty(user_id, "user_id")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        _, now = self._now()
        async with self._session_factory() as session:
            current = (
                await session.execute(
                    select(AutonomousProduction).where(
                        AutonomousProduction.id == run_id, AutonomousProduction.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if current is None:
                raise ValueError("The autonomous goal was not found")
            payload = json.loads(current.payload_json)
            pending = payload.get("pending_delivery") or {}
            if pending.get("retry_of_revision") == expected_revision:
                return _record(current)
            if current.state != "attention" or current.revision != expected_revision:
                raise ValueError("Only the observed attention goal can be retried")
            payload.update(
                last_error=None,
                last_progress_digest=None,
                unchanged_turns=0,
                pending_delivery={
                    "phase": "reconcile_required",
                    "client_key": pending.get("client_key", f"autonomy:{run_id}:retry:{expected_revision}"),
                    "retry_of_revision": expected_revision,
                },
            )
            try:
                row = (
                    await session.execute(
                        sql_update(AutonomousProduction)
                        .where(
                            AutonomousProduction.id == run_id,
                            AutonomousProduction.user_id == user_id,
                            AutonomousProduction.state == "attention",
                            AutonomousProduction.revision == expected_revision,
                        )
                        .values(
                            state="pending",
                            lease_owner=None,
                            lease_until=0,
                            payload_json=_json(payload),
                            updated_at=now,
                            revision=current.revision + 1,
                        )
                        .returning(AutonomousProduction)
                        .execution_options(synchronize_session="fetch")
                    )
                ).scalar_one_or_none()
                if row is None:
                    await session.rollback()
                    reconciled = await self.get(run_id, user_id)
                    if (
                        reconciled is not None
                        and (reconciled.get("pending_delivery") or {}).get("retry_of_revision") == expected_revision
                    ):
                        return reconciled
                    raise ValueError("The autonomous goal changed before retry admission")
                await session.commit()
                return _record(row)
            except IntegrityError as error:
                await session.rollback()
                raise ValueError("The project already has another active autonomous goal") from error

    async def stop(self, run_id: str, user_id: str) -> dict[str, Any]:
        _run_id(run_id)
        _nonempty(user_id, "user_id")
        _, now = self._now()
        clean_pending = and_(
            AutonomousProduction.state == "pending",
            AutonomousProduction.lease_owner.is_(None),
            AutonomousProduction.session_id.is_(None),
            AutonomousProduction.payload_json == _json(_PAYLOAD_DEFAULTS),
        )
        statement = (
            sql_update(AutonomousProduction)
            .where(
                AutonomousProduction.id == run_id,
                AutonomousProduction.user_id == user_id,
                AutonomousProduction.state.in_(ACTIVE_STATES - {"stopping"}),
            )
            .values(
                state=case((clean_pending, "stopped"), else_="stopping"),
                updated_at=now,
                revision=AutonomousProduction.revision + 1,
            )
            .returning(AutonomousProduction)
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
            if row is None:
                row = (
                    await session.execute(
                        select(AutonomousProduction).where(
                            AutonomousProduction.id == run_id, AutonomousProduction.user_id == user_id
                        )
                    )
                ).scalar_one_or_none()
            if row is None:
                raise ValueError("Autonomous goal not found")
            await session.commit()
            return _record(row)


async def create(
    project_name: str,
    episode_id: int,
    objective: str,
    user_id: str,
    client_key: str,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    return await AutonomyStore().create(project_name, episode_id, objective, user_id, client_key, session_id=session_id)


async def get(run_id: str, user_id: str) -> dict[str, Any] | None:
    return await AutonomyStore().get(run_id, user_id)


async def list_active(*, user_id: str | None = None) -> list[dict[str, Any]]:
    return await AutonomyStore().list_active(user_id=user_id)


async def claim(run_id: str, owner_id: str, ttl_seconds: float, *, user_id: str) -> dict[str, Any] | None:
    return await AutonomyStore().claim(run_id, owner_id, ttl_seconds, user_id=user_id)


async def update(run_id: str, owner_id: str, changes: Mapping[str, Any], *, user_id: str) -> dict[str, Any]:
    return await AutonomyStore().update(run_id, owner_id, changes, user_id=user_id)


async def stop(run_id: str, user_id: str) -> dict[str, Any]:
    return await AutonomyStore().stop(run_id, user_id)


async def get_latest(project_name: str, user_id: str) -> dict[str, Any] | None:
    return await AutonomyStore().get_latest(project_name, user_id)
