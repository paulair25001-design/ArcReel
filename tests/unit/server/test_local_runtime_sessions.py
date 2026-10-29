"""Deferred local restoration respects active jobs and exact project ownership."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.infra.async_thread import run_noninterruptible_async, run_sync_transaction
from server.local_runtime_sessions import (
    LocalSessionFinalizer,
    drain_and_finalize_local_media,
    has_other_project_activity,
)


def _active_gateway(tmp_path: Path):
    # These tests isolate scheduling. Resource transitions are exercised by
    # test_l20_resource_session with the current native router transport.
    config = SimpleNamespace(l20_state_dir=tmp_path)
    native = SimpleNamespace(restorations=0)
    path = tmp_path / "active-session.json"
    path.write_text(json.dumps({"project": "demo", "state": "active"}))

    class Gateway:
        def finalize(self, *, project):
            if project != "demo":
                raise RuntimeError("Project does not own this resource session")
            native.restorations += 1
            path.write_text(json.dumps({"project": project, "state": "closed"}))

    return config, native, Gateway()


async def test_stopped_project_waits_for_jobs_and_restores_once(tmp_path: Path):
    config, native, gateway = _active_gateway(tmp_path)
    stats = {"queued": 1, "running": 1}
    waiting, released = asyncio.Event(), asyncio.Event()

    async def read_stats(_project: str):
        return dict(stats)

    async def sleep(seconds: float):
        assert 0 < seconds <= 1
        waiting.set()
        await released.wait()

    async def finish(project: str):
        return await run_sync_transaction(gateway.finalize, project=project)

    coordinator = LocalSessionFinalizer(read_stats, finish, sleep=sleep)
    task = coordinator.request("demo")
    assert coordinator.request("demo") is task
    await waiting.wait()
    assert native.restorations == 0
    stats.update(queued=0, running=0)
    released.set()
    await task
    await coordinator.close()

    assert native.restorations == 1
    assert json.loads((config.l20_state_dir / "active-session.json").read_text())["state"] == "closed"


async def test_unrelated_project_stop_cannot_restore_the_owned_session(tmp_path: Path):
    config, native, gateway = _active_gateway(tmp_path)

    async def idle(_project: str):
        return {"queued": 0, "running": 0}

    async def finish(project: str):
        return await run_sync_transaction(gateway.finalize, project=project)

    coordinator = LocalSessionFinalizer(idle, finish)
    with pytest.raises(RuntimeError, match="does not own"):
        await coordinator.request("unrelated")
    await coordinator.close()

    assert native.restorations == 0
    owner = json.loads((config.l20_state_dir / "active-session.json").read_text())
    assert owner["project"] == "demo"
    assert owner["state"] == "active"


async def test_resuming_while_busy_invalidates_the_old_stop_watcher(tmp_path: Path):
    config, native, gateway = _active_gateway(tmp_path)
    waiting, release = asyncio.Event(), asyncio.Event()
    stats = {"running": 1, "queued": 0}

    async def read_stats(_project: str):
        return dict(stats)

    async def sleep(_seconds: float):
        waiting.set()
        await release.wait()

    async def finish(project: str):
        await run_sync_transaction(gateway.finalize, project=project)

    coordinator = LocalSessionFinalizer(read_stats, finish, sleep=sleep)
    previous_stop = coordinator.request("demo")
    await waiting.wait()
    await coordinator.cancel("demo")
    stats.update(running=0)
    release.set()
    await coordinator.close()

    assert previous_stop.cancelled() is True
    assert native.restorations == 0
    assert json.loads((config.l20_state_dir / "active-session.json").read_text())["state"] == "active"


async def test_interrupt_acknowledgement_precedes_queue_idle_restoration(tmp_path: Path):
    _config, native, gateway = _active_gateway(tmp_path)
    waiting, release = asyncio.Event(), asyncio.Event()
    acknowledged = False

    async def idle(_project: str):
        return {"running": 0, "queued": 0}

    async def sleep(_seconds: float):
        waiting.set()
        await release.wait()

    async def finish(project: str):
        await run_sync_transaction(gateway.finalize, project=project)

    coordinator = LocalSessionFinalizer(idle, finish, sleep=sleep)
    task = coordinator.request("demo", ready=lambda: acknowledged)
    await waiting.wait()
    assert native.restorations == 0
    acknowledged = True
    release.set()
    await task
    await coordinator.close()

    assert native.restorations == 1


def test_another_running_project_session_blocks_restore_even_without_queued_messages():
    active = SimpleNamespace(project_name="demo", session_id="new-chat", status="running", queued_messages=[])
    stopped = SimpleNamespace(project_name="demo", session_id="old-chat", status="running", queued_messages=[])

    assert has_other_project_activity("demo", "old-chat", [stopped, active]) is True
    assert has_other_project_activity("demo", "old-chat", [stopped]) is False
    stopped.queued_messages.append("accepted-future-turn")
    assert has_other_project_activity("demo", "old-chat", [stopped]) is True


async def test_shutdown_restores_after_worker_stop_failure(tmp_path: Path):
    config, native, gateway = _active_gateway(tmp_path)
    stopped = asyncio.Event()
    failure = RuntimeError("worker drain failed")

    async def stop_pending():
        return None

    async def stop_worker():
        stopped.set()
        raise failure

    async def finish():
        assert stopped.is_set()
        await run_sync_transaction(gateway.finalize, project="demo")

    with pytest.raises(RuntimeError) as caught:
        await drain_and_finalize_local_media(stop_worker, stop_pending=stop_pending, finish=finish)

    assert caught.value is failure
    assert native.restorations == 1
    assert json.loads((config.l20_state_dir / "active-session.json").read_text())["state"] == "closed"


async def test_shutdown_cancellation_waits_for_media_drain_before_restoration(tmp_path: Path):
    _config, native, gateway = _active_gateway(tmp_path)
    draining, release = asyncio.Event(), asyncio.Event()

    async def stop_pending():
        return None

    async def stop_worker():
        draining.set()
        await release.wait()

    async def finish():
        await run_sync_transaction(gateway.finalize, project="demo")

    task = asyncio.create_task(
        run_noninterruptible_async(
            drain_and_finalize_local_media(stop_worker, stop_pending=stop_pending, finish=finish)
        )
    )
    await draining.wait()
    task.cancel()
    assert native.restorations == 0
    release.set()
    await task

    assert native.restorations == 1
