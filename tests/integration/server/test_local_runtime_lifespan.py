"""Local lifespan termination drains workers without importing a live controller."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import server.app as app_module
from lib.local_runtime.l20_direct_config import get_l20_direct_config
from tests.unit.server.app_lifespan_support import stub_lifespan_dependencies


@pytest.mark.parametrize("termination", ["exception", "cancellation"])
async def test_local_lifespan_termination_preserves_error_and_drains_worker(
    monkeypatch: pytest.MonkeyPatch, termination: str
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    config = get_l20_direct_config()
    worker = stub_lifespan_dependencies(monkeypatch)
    app = app_module.app
    app.state = SimpleNamespace()
    entered = asyncio.Event()
    hold = asyncio.Event()
    failure = RuntimeError("error inside the application lifespan")

    async def run():
        async with app_module.lifespan(app):
            entered.set()
            if termination == "exception":
                raise failure
            await hold.wait()

    task = asyncio.create_task(run())
    entered_waiter = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait({task, entered_waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done and not entered.is_set():
            await task
    finally:
        if not entered_waiter.done():
            entered_waiter.cancel()
    assert worker.started is True
    if termination == "cancellation":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        # The MCP server's AnyIO task group wraps yielded exceptions without changing the original leaf.
        with pytest.raises(ExceptionGroup) as caught:
            await task
        assert caught.value.exceptions == (failure,)

    assert worker.stopped is True
    assert not (config.state_dir / "resource-session.json").exists()
