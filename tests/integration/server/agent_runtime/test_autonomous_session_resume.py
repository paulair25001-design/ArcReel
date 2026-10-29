"""New native production goals can retain a verified existing Hermes session."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select

from lib.db.models.autonomous_production import AutonomousProduction
from server.agent_runtime.autonomous_production import AutonomousProductionService
from server.agent_runtime.autonomy_store import AutonomyStore
from server.auth import CurrentUserInfo, get_current_user
from server.error_handlers import register_error_handlers
from server.routers import autonomous_production as api


async def test_store_session_binding_is_persistent_and_idempotency_cannot_replace_it(db_factory):
    store = AutonomyStore(session_factory=db_factory)
    args = ("one", 2, "Finish native episode", "owner", "new-authorization")
    first = await store.create(*args, session_id="existing-native")
    assert first["session_id"] == "existing-native"
    assert first["state"] == "pending"
    assert first["dispatch_count"] == 0
    restarted = AutonomyStore(session_factory=db_factory)
    assert await restarted.create(*args) == first
    assert await restarted.create(*args, session_id="existing-native") == first
    with pytest.raises(ValueError, match="different session"):
        await restarted.create(*args, session_id="replacement-native")
    assert await store.get(first["id"], "owner") == first


@pytest.mark.parametrize("state", ["completed", "stopped", "attention"])
async def test_new_authorization_keeps_terminal_goal_history_and_original_session(db_factory, state):
    store = AutonomyStore(session_factory=db_factory)
    old = await store.create("one", 2, "Earlier objective", "owner", "old-key", session_id="original-native")
    await store.claim(old["id"], "driver", 60, user_id="owner")
    previous = await store.update(old["id"], "driver", {"state": state}, user_id="owner")
    new = await store.create("one", 2, "Continue final native edit", "owner", "new-key", session_id="original-native")
    assert new["id"] != old["id"]
    assert new["session_id"] == "original-native"
    assert new["state"] == "pending"
    assert await store.get(old["id"], "owner") == previous
    assert await store.create("one", 2, "Earlier objective", "owner", "old-key") == previous
    assert [row["id"] for row in await store.list_active(user_id="owner")] == [new["id"]]


async def test_concurrent_same_key_with_different_explicit_sessions_accepts_only_one_binding(file_db_factory):
    stores = [AutonomyStore(session_factory=file_db_factory) for _ in range(2)]
    results = await asyncio.gather(
        stores[0].create("one", 2, "Finish native episode", "owner", "key", session_id="first-native"),
        stores[1].create("one", 2, "Finish native episode", "owner", "key", session_id="second-native"),
        return_exceptions=True,
    )
    accepted = [value for value in results if isinstance(value, dict)]
    assert len(accepted) == 1
    assert sum(isinstance(value, ValueError) for value in results) == 1
    assert (await stores[0].get(accepted[0]["id"], "owner"))["session_id"] == accepted[0]["session_id"]
    async with file_db_factory() as session:
        assert (await session.execute(select(func.count()).select_from(AutonomousProduction))).scalar_one() == 1


@pytest.fixture
async def native_resume_api(db_factory, tmp_path, monkeypatch):
    runtime = tmp_path / "hermes"
    home = runtime / "sessions/original"
    binding = runtime / "bindings/original-native.json"

    def prepare_files():
        home.mkdir(parents=True)
        binding.parent.mkdir()
        (home / ".arcreel-local-hermes").write_text("arcreel-local-hermes/v1\n")
        (home / "native-agent-reports.jsonl").write_text("original immutable review receipt\n")
        binding.write_text(
            json.dumps({"schema": "arcreel-hermes-session/v1", "home": "sessions/original", "user_id": "owner"})
        )

    await asyncio.to_thread(prepare_files)
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    monkeypatch.setenv("ARCREEL_HERMES_HOME", str(runtime))
    store = AutonomyStore(session_factory=db_factory)
    meta = SimpleNamespace(project_name="one", superseded_by=None, status="idle")
    lookups, fact_reads = [], []

    async def get_session(sid):
        lookups.append(sid)
        return meta if sid == "original-native" else None

    async def facts(project, episode, *, user_id):
        fact_reads.append((project, episode, user_id))
        return {"project": project, "episode_id": episode, "units": []}

    assistant: Any = SimpleNamespace(get_session=get_session)
    service = AutonomousProductionService(assistant, store=store, queue=SimpleNamespace(), facts_reader=facts)
    monkeypatch.setattr(api, "get_autonomous_production", lambda: service)
    app = FastAPI()
    app.include_router(api.router, prefix="/api/v1")
    register_error_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: CurrentUserInfo(id="owner", sub="owner")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(
            client=client,
            store=store,
            service=service,
            meta=meta,
            binding=binding,
            home=home,
            lookups=lookups,
            fact_reads=fact_reads,
        )


def _request(**changes):
    return {
        "episode_id": 2,
        "objective": "Continue native editing",
        "client_key": "new-goal",
        "resume_session_id": "original-native",
        **changes,
    }


async def test_http_start_binds_existing_owned_session_without_modifying_review_receipts(native_resume_api):
    rig = native_resume_api
    response = await rig.client.post("/api/v1/projects/one/autonomous-production", json=_request())
    assert response.status_code == 200
    goal = response.json()["goal"]
    assert goal["session_id"] == "original-native"
    assert goal["state"] == "pending"
    assert goal["dispatch_count"] == 0
    assert "pending_delivery" not in goal
    assert "hermes" not in response.text
    assert rig.lookups == ["original-native"]
    assert rig.fact_reads == [("one", 2, "owner")]
    assert (
        await asyncio.to_thread((rig.home / "native-agent-reports.jsonl").read_text)
        == "original immutable review receipt\n"
    )
    repeated = await rig.client.post("/api/v1/projects/one/autonomous-production", json=_request())
    assert repeated.json() == response.json()


@pytest.mark.parametrize(
    "failure",
    ["missing-meta", "wrong-project", "superseded", "missing-binding", "wrong-user", "missing-marker", "wrong-marker"],
)
async def test_http_start_refuses_unowned_or_superseded_session_without_creating_goal(native_resume_api, failure):
    rig = native_resume_api
    body = _request()
    if failure == "missing-meta":
        body["resume_session_id"] = "missing-native"
        await asyncio.to_thread(
            (rig.binding.parent / "missing-native.json").write_bytes, await asyncio.to_thread(rig.binding.read_bytes)
        )
    elif failure == "wrong-project":
        rig.meta.project_name = "other-project"
    elif failure == "superseded":
        rig.meta.superseded_by = "replacement-native"
    elif failure == "missing-binding":
        await asyncio.to_thread(rig.binding.unlink)
    elif failure == "wrong-user":
        await asyncio.to_thread(
            rig.binding.write_text,
            json.dumps({"schema": "arcreel-hermes-session/v1", "home": "sessions/original", "user_id": "other-user"}),
        )
    elif failure == "missing-marker":
        await asyncio.to_thread((rig.home / ".arcreel-local-hermes").unlink)
    else:
        await asyncio.to_thread((rig.home / ".arcreel-local-hermes").write_text, "other-owner")
    response = await rig.client.post("/api/v1/projects/one/autonomous-production", json=body)
    assert response.status_code == 409
    assert await rig.store.list_active() == []
    assert await rig.store.get_latest("one", "owner") is None


@pytest.mark.parametrize("sid", ["", "../other", "a/b", "x" * 201, "with space", 123, True])
async def test_http_session_identity_rejected_before_facts_or_dispatch(native_resume_api, sid):
    rig = native_resume_api
    response = await rig.client.post("/api/v1/projects/one/autonomous-production", json=_request(resume_session_id=sid))
    assert response.status_code == 422
    assert rig.lookups == []
    assert rig.fact_reads == []
    assert await rig.store.list_active() == []


async def test_http_omitted_session_keeps_default_new_goal_behavior(native_resume_api):
    rig = native_resume_api
    body = _request()
    body.pop("resume_session_id")
    response = await rig.client.post("/api/v1/projects/one/autonomous-production", json=body)
    assert response.status_code == 200
    assert response.json()["goal"]["session_id"] is None
    assert rig.lookups == []


async def test_first_driver_turn_on_resumed_session_reconciles_without_creating_a_new_session(native_resume_api):
    rig = native_resume_api
    calls = []

    async def stats(project):
        assert project == "one"
        return {"queued": 0, "running": 0}

    async def send(project, message, *, session_id, client_key, on_session_prepared):
        assert session_id == "original-native"
        await on_session_prepared(session_id)
        calls.append({"project": project, "message": message, "session_id": session_id, "client_key": client_key})
        return {"session_id": session_id, "entry": {"uuid": "native-entry"}}

    rig.service.queue = SimpleNamespace(get_task_stats=stats)
    rig.service.assistant.session_manager = SimpleNamespace(sessions={})
    rig.service.assistant.send_or_create = send
    goal = await rig.service.create(
        "one", 2, "Continue native editing", "owner", "key", resume_session_id="original-native"
    )
    await rig.service.advance(goal)
    assert len(calls) == 1
    assert json.loads(calls[0]["message"].splitlines()[1])["mode"] == "reconcile"
    saved = await rig.store.get(goal["id"], "owner")
    assert saved["session_id"] == "original-native"
    assert saved["dispatch_count"] == 1
    assert saved["pending_delivery"]["phase"] == "delivered"
