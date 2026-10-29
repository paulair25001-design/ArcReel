"""Durable goals, scoped idempotency, and lease races on the native database."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from sqlalchemy import func, select

from lib.db.models.autonomous_production import AutonomousProduction
from server.agent_runtime.autonomy_store import AutonomyStore


@dataclass
class Clock:
    now: float = 1000

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def goal_store(db_factory):
    clock = Clock()
    return AutonomyStore(session_factory=db_factory, clock=clock), clock, db_factory


async def create(store, *, user_id="default", client_key="goal-one", objective="Finish episode 2"):
    return await store.create("project-one", 2, objective, user_id, client_key)


async def test_pending_delivery_and_idempotency_survive_new_store(goal_store):
    store, clock, factory = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver-one", 30, user_id="default")
    delivery = {"client_key": "stable-delivery-one", "status": "uncertain", "message_id": None}
    await store.update(goal["id"], "driver-one", {"pending_delivery": delivery, "dispatch_count": 1}, user_id="default")
    clock.now += 31
    restarted = AutonomyStore(session_factory=factory, clock=clock)
    recovered = await create(restarted)
    assert recovered["id"] == goal["id"]
    assert recovered["pending_delivery"] == delivery
    assert recovered["dispatch_count"] == 1
    assert recovered["state"] == "pending"
    claimed = await restarted.claim(goal["id"], "driver-two", 30, user_id="default")
    assert claimed is not None
    assert claimed["lease_owner"] == "driver-two"
    assert claimed["pending_delivery"]["client_key"] == "stable-delivery-one"
    assert claimed["dispatch_count"] == 1


async def test_client_key_and_active_scope_conflicts_are_distinct(goal_store):
    store, _, _ = goal_store
    goal = await create(store)
    with pytest.raises(ValueError, match="different objective"):
        await create(store, objective="A different goal")
    with pytest.raises(ValueError, match="active goal"):
        await create(store, client_key="different-key")
    other_user = await create(store, user_id="u1")
    assert other_user["id"] != goal["id"]
    assert len(await store.list_active()) == 2
    assert [row["id"] for row in await store.list_active(user_id="default")] == [goal["id"]]


async def test_user_scope_applies_to_every_specific_goal_operation(goal_store):
    store, _, _ = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver-one", 30, user_id="default")
    assert await store.get(goal["id"], "u1") is None
    assert await store.get_latest("project-one", "u1") is None
    assert await store.claim(goal["id"], "driver-two", 30, user_id="u1") is None
    with pytest.raises(ValueError, match="lease"):
        await store.update(goal["id"], "driver-one", {"state": "running"}, user_id="u1")
    with pytest.raises(ValueError, match="not found"):
        await store.stop(goal["id"], "u1")
    assert (await store.get(goal["id"], "default"))["state"] == "pending"


async def test_lease_renewal_expiry_and_release_guard_writes(goal_store):
    store, clock, _ = goal_store
    goal = await create(store)
    first = await store.claim(goal["id"], "driver-one", 10, user_id="default")
    assert first["lease_until"] == 1010
    assert await store.claim(goal["id"], "driver-two", 10, user_id="default") is None
    clock.now += 2
    renewed = await store.claim(goal["id"], "driver-one", 10, user_id="default")
    assert renewed["lease_until"] == 1012
    clock.now = 1012
    with pytest.raises(ValueError, match="lease"):
        await store.update(goal["id"], "driver-one", {"state": "running"}, user_id="default")
    second = await store.claim(goal["id"], "driver-two", 10, user_id="default")
    assert second["lease_owner"] == "driver-two"
    with pytest.raises(ValueError, match="lease"):
        await store.update(goal["id"], "driver-one", {"session_id": "wrong-session"}, user_id="default")
    released = await store.update(goal["id"], "driver-two", {"lease_owner": None, "lease_until": 0}, user_id="default")
    assert released["lease_owner"] is None
    assert released["lease_until"] == 0
    assert (await store.claim(goal["id"], "driver-three", 10, user_id="default"))["lease_owner"] == "driver-three"


async def test_stop_pending_is_terminal_and_completed_goal_is_not_revived(goal_store):
    store, clock, _ = goal_store
    goal = await create(store)
    stopped = await store.stop(goal["id"], "default")
    assert stopped["state"] == "stopped"
    assert await store.claim(goal["id"], "driver", 30, user_id="default") is None
    assert await store.stop(goal["id"], "default") == stopped
    assert (await create(store))["state"] == "stopped"
    clock.now += 1
    next_goal = await create(store, client_key="second-goal")
    await store.claim(next_goal["id"], "driver", 30, user_id="default")
    completed = await store.update(next_goal["id"], "driver", {"state": "completed"}, user_id="default")
    assert completed["lease_owner"] is None
    assert completed["lease_until"] == 0
    with pytest.raises(ValueError, match="lease"):
        await store.update(next_goal["id"], "driver", {"state": "running"}, user_id="default")
    assert await store.stop(next_goal["id"], "default") == completed
    assert (await store.get_latest("project-one", "default"))["id"] == next_goal["id"]
    assert await store.list_active() == []


async def test_stop_inflight_requires_ack_and_does_not_allow_running(goal_store):
    store, clock, factory = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver", 10, user_id="default")
    await store.update(goal["id"], "driver", {"state": "running", "session_id": "hermes-session"}, user_id="default")
    stopping = await store.stop(goal["id"], "default")
    assert stopping["state"] == "stopping"
    with pytest.raises(ValueError, match="cannot resume"):
        await store.update(goal["id"], "driver", {"state": "running"}, user_id="default")
    clock.now += 11
    restarted = AutonomyStore(session_factory=factory, clock=clock)
    claimed = await restarted.claim(goal["id"], "replacement", 30, user_id="default")
    assert claimed["state"] == "stopping"
    final = await restarted.update(goal["id"], "replacement", {"state": "stopped"}, user_id="default")
    assert final["state"] == "stopped"
    assert final["lease_owner"] is None


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"objective": "replace"}, "Only autonomous"),
        ({"user_id": "u1"}, "Only autonomous"),
        ({"revision": 0}, "Only autonomous"),
        ({"lease_until": 99999}, "Only paired"),
        ({"lease_owner": "other", "lease_until": 0}, "Only paired"),
        ({"state": "unknown"}, "Unknown autonomous"),
        ({"pending_delivery": []}, "pending_delivery"),
        ({"dispatch_count": -1}, "dispatch_count"),
    ],
)
async def test_updates_cannot_change_identity_or_assign_arbitrary_lease(goal_store, changes, error):
    store, _, _ = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    before = await store.get(goal["id"], "default")
    with pytest.raises(ValueError, match=error):
        await store.update(goal["id"], "driver", changes, user_id="default")
    assert await store.get(goal["id"], "default") == before


@pytest.mark.parametrize("project", ["../project", "folder/project", "project_name", "项目", "", "."])
async def test_project_identity_is_a_slug(goal_store, project):
    store, _, _ = goal_store
    with pytest.raises(ValueError, match="project_name"):
        await store.create(project, 2, "A goal", "default", "key")
    assert await store.list_active() == []


async def test_concurrent_create_and_claim_use_database_uniqueness(file_db_factory):
    clock = Clock()
    first = AutonomyStore(session_factory=file_db_factory, clock=clock)
    second = AutonomyStore(session_factory=file_db_factory, clock=clock)
    created = await asyncio.gather(create(first), create(second))
    assert created[0]["id"] == created[1]["id"]
    async with file_db_factory() as session:
        assert (await session.execute(select(func.count()).select_from(AutonomousProduction))).scalar_one() == 1
    claims = await asyncio.gather(
        first.claim(created[0]["id"], "first", 30, user_id="default"),
        second.claim(created[0]["id"], "second", 30, user_id="default"),
    )
    accepted = [claim for claim in claims if claim is not None]
    assert len(accepted) == 1
    assert (await first.get(created[0]["id"], "default"))["lease_owner"] == accepted[0]["lease_owner"]


async def test_stop_racing_driver_update_cannot_leave_running(file_db_factory):
    store = AutonomyStore(session_factory=file_db_factory, clock=Clock())
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    results = await asyncio.gather(
        store.stop(goal["id"], "default"),
        store.update(goal["id"], "driver", {"state": "running"}, user_id="default"),
        return_exceptions=True,
    )
    assert any(isinstance(result, dict) and result["state"] == "stopping" for result in results)
    assert (await store.get(goal["id"], "default"))["state"] == "stopping"


async def test_attention_retry_preserves_identity_and_requests_fresh_reconciliation(goal_store):
    store, _, _ = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    attention = await store.update(
        goal["id"],
        "driver",
        {
            "state": "attention",
            "session_id": "original-hermes",
            "dispatch_count": 7,
            "unchanged_turns": 3,
            "last_progress_digest": "old",
            "last_error": "fixed_problem",
        },
        user_id="default",
    )
    retried = await store.retry(goal["id"], "default", attention["revision"])
    assert retried["id"] == goal["id"]
    assert retried["session_id"] == "original-hermes"
    assert retried["dispatch_count"] == 7
    assert retried["state"] == "pending"
    assert retried["unchanged_turns"] == 0
    assert retried["last_progress_digest"] is None
    assert retried["pending_delivery"]["phase"] == "reconcile_required"
    assert await store.retry(goal["id"], "default", attention["revision"]) == retried
    assert await store.claim(goal["id"], "next-driver", 30, user_id="default") is not None


@pytest.mark.parametrize("state", ["pending", "stopping", "stopped", "completed"])
async def test_retry_cannot_resume_user_stop_or_completed_goal(goal_store, state):
    store, _, _ = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    row = await store.update(goal["id"], "driver", {"state": state}, user_id="default")
    with pytest.raises(ValueError, match="attention"):
        await store.retry(goal["id"], "default", row["revision"])
    assert (await store.get(goal["id"], "default"))["state"] == state


async def test_retry_rejects_foreign_user_stale_revision_and_other_active_goal(goal_store):
    store, _, _ = goal_store
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    row = await store.update(goal["id"], "driver", {"state": "attention"}, user_id="default")
    with pytest.raises(ValueError, match="not found"):
        await store.retry(goal["id"], "foreign", row["revision"])
    with pytest.raises(ValueError, match="attention"):
        await store.retry(goal["id"], "default", row["revision"] - 1)
    other = await create(store, client_key="another-authorization")
    with pytest.raises(ValueError, match="active"):
        await store.retry(goal["id"], "default", row["revision"])
    assert (await store.get(goal["id"], "default"))["state"] == "attention"
    assert (await store.get(other["id"], "default"))["state"] == "pending"


async def test_concurrent_retry_is_one_reconciliation(file_db_factory):
    store = AutonomyStore(session_factory=file_db_factory, clock=Clock())
    goal = await create(store)
    await store.claim(goal["id"], "driver", 30, user_id="default")
    row = await store.update(goal["id"], "driver", {"state": "attention"}, user_id="default")
    retried = await asyncio.gather(
        store.retry(goal["id"], "default", row["revision"]),
        store.retry(goal["id"], "default", row["revision"]),
    )
    assert retried[0]["revision"] == retried[1]["revision"]
    assert retried[0]["state"] == "pending"
