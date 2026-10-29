"""Native event wake, shutdown and restart preserve one durable Hermes goal."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from server.agent_runtime import autonomous_production as driver
from server.agent_runtime.autonomy_store import AutonomyStore


@pytest.fixture
async def runner_rig(db_factory, monkeypatch):
    listeners = []

    def register(listener):
        listeners.append(listener)

        def unregister():
            listeners.remove(listener)

        return unregister

    monkeypatch.setattr(driver, "register_project_change_batch_listener", register)
    store = AutonomyStore(session_factory=db_factory)
    waiting, stopped, interrupted, scanned = (asyncio.Event() for _ in range(4))
    deliveries: asyncio.Queue[dict] = asyncio.Queue()
    original_update, original_list = store.update, store.list_active

    async def observed_update(*args, **kwargs):
        result = await original_update(*args, **kwargs)
        if result["state"] == "waiting":
            waiting.set()
        if result["state"] == "stopped":
            stopped.set()
        return result

    async def observed_list(*args, **kwargs):
        result = await original_list(*args, **kwargs)
        scanned.set()
        return result

    monkeypatch.setattr(store, "update", observed_update)
    monkeypatch.setattr(store, "list_active", observed_list)
    calls = []
    metas = {}
    entries = {}
    residents = {}
    interruptions = []
    restored = []
    stats = {"queued": 0, "running": 0}
    facts = {
        "project": "runner-project",
        "episode_id": 2,
        "script_file": "episode_2.json",
        "script_state": "current",
        "units": [
            {
                "unit_id": "E2S01",
                "text": "原文",
                "declared_seconds": 6,
                "audio": {"status": "current", "actual_seconds": 5},
                "first": {"status": "missing"},
                "video": {"status": "missing"},
                "end_frame": {"present": False},
            }
        ],
        "active_tasks": [],
        "failed_tasks": [],
        "timelines": [],
        "goal_complete_technical": False,
        "resource": {"owner_project": "runner-project", "state": "restored", "original_service_restored": True},
    }

    async def queue_stats(_project):
        return dict(stats)

    async def read_facts(_project, _episode, *, user_id):
        assert user_id == "default"
        return copy.deepcopy(facts)

    async def get_session(sid):
        return metas.get(sid)

    async def find_entry(sid, key):
        return entries.get((sid, key))

    async def send(project, content, *, session_id=None, client_key=None, on_session_prepared=None):
        sid = session_id or "same-native-hermes-session"
        if on_session_prepared is not None:
            await on_session_prepared(sid)
        calls.append({"session_id": sid, "client_key": client_key, "message": content, "project": project})
        metas[sid] = SimpleNamespace(status="running")
        residents[sid] = SimpleNamespace(queued_messages=[], pending_questions={})
        entry = {"uuid": "entry-" + str(len(calls))}
        entries[sid, client_key] = entry
        deliveries.put_nowait(calls[-1])
        return {"session_id": sid, "entry": entry, "queued_message": None}

    async def interrupt(sid):
        interruptions.append(sid)
        metas[sid].status = "interrupted"
        interrupted.set()

    async def finish(*, project):
        restored.append(project)
        facts["resource"].update(state="restored", original_service_restored=True)

    assistant = SimpleNamespace(
        session_manager=SimpleNamespace(sessions=residents),
        event_log_store=SimpleNamespace(find_by_client_key=find_entry),
        get_session=get_session,
        send_or_create=send,
        interrupt_session=interrupt,
    )

    def service():
        return driver.AutonomousProductionService(
            assistant,
            store=store,
            queue=SimpleNamespace(get_task_stats=queue_stats),
            facts_reader=read_facts,
            finalizer=finish,
            poll_seconds=60,
        )

    goal = await store.create("runner-project", 2, "Finish the native episode", "default", "authorized-runner-goal")

    def event(action):
        for listener in tuple(listeners):
            listener("runner-project", "worker", ({"action": action},))

    return SimpleNamespace(
        store=store,
        goal=goal,
        calls=calls,
        metas=metas,
        residents=residents,
        restored=restored,
        stats=stats,
        facts=facts,
        interruptions=interruptions,
        service=service,
        event=event,
        listeners=listeners,
        waiting=waiting,
        stopped=stopped,
        interrupted=interrupted,
        scanned=scanned,
        deliveries=deliveries,
    )


async def test_runner_event_wake_and_restart_reconcile_the_same_durable_session(runner_rig):
    rig = runner_rig
    first = rig.service()

    try:
        await first.start()
        await asyncio.wait_for(rig.deliveries.get(), timeout=2)
        await asyncio.shield(first._inflight[rig.goal["id"]])
        sid = rig.calls[0]["session_id"]
        rig.stats["running"] = 1
        rig.event("task_updated")
        await asyncio.wait_for(rig.waiting.wait(), timeout=2)
        await asyncio.shield(first._inflight[rig.goal["id"]])
        assert len(rig.calls) == 1
        rig.stats["running"] = 0
        rig.metas[sid].status = "completed"
        rig.facts["units"][0]["first"] = {"status": "current", "sha256": "committed-image"}
        rig.event("task_succeeded")
        await asyncio.wait_for(rig.deliveries.get(), timeout=2)
        await asyncio.shield(first._inflight[rig.goal["id"]])
        assert rig.calls[1]["session_id"] == sid
        assert rig.calls[1]["client_key"] != rig.calls[0]["client_key"]
    finally:
        await first.close()
    assert rig.listeners == []
    assert rig.restored == []
    stored = await rig.store.get(rig.goal["id"], "default")
    assert stored["state"] in {"running", "waiting"}
    assert stored["session_id"] == sid
    assert stored["lease_owner"] is None

    rig.residents.clear()
    rig.metas[sid].status = "interrupted"
    restarted = rig.service()
    try:
        await restarted.start()
        await asyncio.wait_for(rig.deliveries.get(), timeout=2)
        await asyncio.shield(restarted._inflight[rig.goal["id"]])
        assert rig.calls[2]["session_id"] == sid
        context = json.loads(rig.calls[2]["message"].splitlines()[1])
        assert context["mode"] == "reconcile"
        assert context["run_id"] == rig.goal["id"]
        assert rig.calls[2]["client_key"] != rig.calls[1]["client_key"]
    finally:
        await restarted.close()
    assert rig.listeners == []
    assert rig.restored == []


async def test_runner_stop_drains_native_writers_and_late_event_cannot_restart_goal(runner_rig):
    rig = runner_rig
    service = rig.service()

    try:
        await service.start()
        await asyncio.wait_for(rig.deliveries.get(), timeout=2)
        await asyncio.shield(service._inflight[rig.goal["id"]])
        rig.stats["running"] = 1
        await service.stop(rig.goal["id"], "default")
        await asyncio.wait_for(rig.interrupted.wait(), timeout=2)
        await asyncio.shield(service._inflight[rig.goal["id"]])
        assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopping"
        assert rig.restored == []
        rig.stats["running"] = 0
        rig.event("task_succeeded")
        await asyncio.wait_for(rig.stopped.wait(), timeout=2)
        await asyncio.shield(service._inflight[rig.goal["id"]])
        assert rig.restored == ["runner-project"]
        rig.scanned.clear()
        rig.event("task_succeeded")
        await asyncio.wait_for(rig.scanned.wait(), timeout=2)
        assert len(rig.calls) == 1
        assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopped"
    finally:
        await service.close()
    assert rig.listeners == []
