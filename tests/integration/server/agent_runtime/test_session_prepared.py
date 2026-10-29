"""Hermes init identity is durable before native goal dispatch can send a prompt."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent_runtime.service import AssistantService
from server.agent_runtime.session_manager import SessionManager
from server.agent_runtime.session_store import SessionMetaStore
from tests.factories import make_session_meta
from tests.fakes import FakeSDKClient, system_frame

SID = "baac5f8e-5f05-431a-ab88-cf9cbe4c63fe"


class _InitBeforeQueryClient(FakeSDKClient):
    def __init__(self, order: list[str]):
        super().__init__()
        self.order = order

    async def __aenter__(self):
        client = await super().__aenter__()
        self.order.append("binding")
        self.push_frame(system_frame("init", session_id=SID))
        return client

    async def query(self, prompt, session_id="default"):
        self.order.append("query")
        await super().query(prompt, session_id=session_id)


def _manager(tmp_path: Path, meta_store: SessionMetaStore, monkeypatch, client) -> SessionManager:
    manager = SessionManager(project_root=tmp_path, meta_store=meta_store, sdk_id_timeout=1)
    project = manager.layout.projects_dir / "demo"
    project.mkdir(parents=True)
    (project / "project.json").write_text('{"title":"demo"}', encoding="utf-8")

    async def options(*_args, **_kwargs):
        return SimpleNamespace(env={})

    monkeypatch.setattr(manager, "_build_options", options)
    monkeypatch.setattr(manager, "_client_for_options", lambda _options: client)
    return manager


async def test_hermes_prepared_callback_waits_for_init_and_metadata_before_first_query(
    tmp_path,
    meta_store,
    monkeypatch,
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    order: list[str] = []
    client = _InitBeforeQueryClient(order)
    manager = _manager(tmp_path, meta_store, monkeypatch, client)
    entered, persisted = asyncio.Event(), asyncio.Event()

    async def on_prepared(sid: str) -> None:
        assert sid == SID
        assert (await meta_store.get(sid)).project_name == "demo"
        assert manager.sessions[sid].sdk_id_event.is_set()
        order.append("metadata")
        entered.set()
        await persisted.wait()
        order.append("prepared")

    task = asyncio.create_task(manager.send_new_session("demo", "make episode", on_session_prepared=on_prepared))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert client.sent_queries == []
        persisted.set()
        assert await asyncio.wait_for(task, timeout=1) == SID
        assert len(client.sent_queries) == 1
        assert order == ["binding", "metadata", "prepared", "query"]
    finally:
        persisted.set()
        await manager.close_session(SID)


@pytest.mark.parametrize("exception_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_preparation_closes_hermes_without_sending_a_query(
    tmp_path,
    meta_store,
    monkeypatch,
    exception_type,
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    client = _InitBeforeQueryClient([])
    manager = _manager(tmp_path, meta_store, monkeypatch, client)

    async def on_prepared(_sid: str) -> None:
        raise exception_type("goal persistence failed")

    with pytest.raises(exception_type, match="goal persistence failed"):
        await manager.send_new_session("demo", "make episode", on_session_prepared=on_prepared)

    assert client.sent_queries == []
    assert client.disconnected
    assert manager.sessions == {}
    assert (await meta_store.get(SID)).status == "error"


async def test_legacy_client_keeps_send_first_and_does_not_invoke_hermes_preparation(
    tmp_path,
    meta_store,
    monkeypatch,
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "false")
    client = FakeSDKClient(frames=[system_frame("init", session_id=SID)])
    manager = _manager(tmp_path, meta_store, monkeypatch, client)
    callbacks = []

    async def on_prepared(sid: str) -> None:
        callbacks.append(sid)

    try:
        assert await manager.send_new_session("demo", "hello", on_session_prepared=on_prepared) == SID
        assert len(client.sent_queries) == 1
        assert callbacks == []
    finally:
        await manager.close_session(SID)


async def test_stop_after_existing_session_connect_prevents_queue_registration_and_query(
    tmp_path,
    meta_store,
    monkeypatch,
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    client = _InitBeforeQueryClient([])
    manager = _manager(tmp_path, meta_store, monkeypatch, client)
    await manager.send_new_session("demo", "first goal")
    existing_queries = len(client.sent_queries)
    stopped = asyncio.Event()
    original_connect = manager.get_or_connect

    async def connect_then_stop(*args, **kwargs):
        managed = await original_connect(*args, **kwargs)
        stopped.set()
        return managed

    async def admission(sid: str) -> None:
        assert stopped.is_set()
        assert sid == SID
        assert manager.sessions[sid].send_lock.locked()
        raise RuntimeError("goal stopped before admission")

    monkeypatch.setattr(manager, "get_or_connect", connect_then_stop)
    try:
        with pytest.raises(RuntimeError, match="goal stopped before admission"):
            await manager.send_message(
                SID, "continue goal", client_key="stopped-delivery", on_session_prepared=admission
            )
        assert len(client.sent_queries) == existing_queries
        assert manager.sessions[SID].queued_messages == []
        assert "stopped-delivery" not in manager.sessions[SID].sent_client_keys
    finally:
        await manager.close_session(SID)


@pytest.mark.parametrize("local", [True, False])
async def test_service_forwards_preparation_only_for_local_existing_session(tmp_path, monkeypatch, local):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", str(local).lower())
    service = AssistantService(project_root=tmp_path)
    service.pm = SimpleNamespace(get_project_path=lambda _project: tmp_path)
    forwarded = []

    async def prepared(_sid):
        return None

    async def get_meta(_sid):
        return make_session_meta(id=SID, project_name="demo")

    async def backfill(*_args):
        return None

    async def send(_sid, _prompt, **kwargs):
        forwarded.append(kwargs)
        return {"queued_message": {"id": "one"}}

    async def turn(_project, operation):
        return await operation()

    service.meta_store = SimpleNamespace(get=get_meta)
    service.session_manager = SimpleNamespace(send_message=send)
    service.event_log = SimpleNamespace(ensure_backfilled=backfill)
    monkeypatch.setattr(service, "_with_local_project_turn", turn)
    await service.send_or_create("demo", "continue goal", session_id=SID, on_session_prepared=prepared)
    assert len(forwarded) == 1
    assert forwarded[0].get("on_session_prepared") is (prepared if local else None)


@pytest.mark.parametrize("local", [True, False])
async def test_service_preparation_reuses_same_client_key_identity_without_second_prompt(tmp_path, monkeypatch, local):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", str(local).lower())
    service = AssistantService(project_root=tmp_path)
    service.pm = SimpleNamespace(get_project_path=lambda _project: tmp_path)
    entries = {}
    new_calls = []
    callbacks = []
    sessions = {}

    async def prepared(sid: str) -> None:
        callbacks.append(sid)

    async def new_session(project, content, *, user_entry, client_key, **kwargs):
        new_calls.append((project, content))
        if local:
            await kwargs["on_session_prepared"](SID)
        else:
            assert "on_session_prepared" not in kwargs
        entries[client_key] = (SID, user_entry)
        sessions[SID] = SimpleNamespace(initial_user_log_entry=user_entry)
        return SID

    async def find_new(key):
        return entries.get(key)

    async def find_by(sid, key):
        pair = entries.get(key)
        return pair[1] if pair and pair[0] == sid else None

    async def get_meta(_sid):
        return make_session_meta(id=SID, project_name="demo")

    async def turn(_project, operation):
        return await operation()

    service.session_manager = SimpleNamespace(send_new_session=new_session, sessions=sessions)
    service.meta_store = SimpleNamespace(get=get_meta)
    service.event_log_store = SimpleNamespace(find_new_session_by_client_key=find_new, find_by_client_key=find_by)
    monkeypatch.setattr(service, "_with_local_project_turn", turn)

    first = await service.send_or_create(
        "demo", "make episode", client_key="goal-delivery", on_session_prepared=prepared
    )
    second = await service.send_or_create(
        "demo", "make episode", client_key="goal-delivery", on_session_prepared=prepared
    )
    service._new_session_client_keys.clear()
    recovered = await service.send_or_create(
        "demo", "make episode", client_key="goal-delivery", on_session_prepared=prepared
    )

    assert first["session_id"] == second["session_id"] == recovered["session_id"] == SID
    assert len(new_calls) == 1
    assert callbacks == ([SID, SID, SID] if local else [])
