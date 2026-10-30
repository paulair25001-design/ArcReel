"""Restore only explicitly stopped local projects after their media queue settles."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from lib.generation.generation_queue import get_generation_queue
from lib.infra.logging_utils import redact_diagnostic_text
from server.agent_runtime.keyed_locks import KeyedLocks

logger = logging.getLogger(__name__)
StatsReader = Callable[[str], Awaitable[dict[str, int]]]
ProjectFinalizer = Callable[[str], Awaitable[Any]]


class LocalSessionFinalizer:
    """One deferred restoration per project; normal assistant turns do not schedule it."""

    def __init__(
        self,
        stats: StatsReader,
        finish: ProjectFinalizer,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        poll_seconds: float = 1.0,
    ) -> None:
        if not 0 < poll_seconds <= 1:
            raise ValueError("Local finalization polling must be between zero and one second")
        self._stats = stats
        self._finish = finish
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def request(self, project: str, *, ready: Callable[[], bool] | None = None) -> asyncio.Task[None]:
        current = self._tasks.get(project)
        if current is not None and not current.done():
            return current
        task = asyncio.create_task(self._wait_and_finish(project, ready), name="finish-local-project:" + project)
        self._tasks[project] = task
        task.add_done_callback(lambda completed: self._finished(project, completed))
        return task

    async def _wait_and_finish(self, project: str, ready: Callable[[], bool] | None) -> None:
        while True:
            if ready is not None and not ready():
                await self._sleep(self._poll_seconds)
                continue
            stats = await self._stats(project)
            if not stats.get("queued", 0) and not stats.get("running", 0):
                await self._finish(project)
                return
            await self._sleep(self._poll_seconds)

    def _finished(self, project: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(project) is task:
            del self._tasks[project]
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(
                "Local project restoration remains unconfirmed project=%s: %s", project, redact_diagnostic_text(error)
            )

    async def close(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def cancel(self, project: str) -> None:
        """A resumed turn invalidates a previous stop before admitting new work."""
        task = self._tasks.get(project)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


_coordinator: LocalSessionFinalizer | None = None
_project_locks = KeyedLocks()


@asynccontextmanager
async def local_project_lock(project: str) -> AsyncGenerator[None]:
    async with _project_locks.lock_for(project):
        yield


async def cancel_local_project_finalization(project: str) -> None:
    if _coordinator is not None:
        await _coordinator.cancel(project)


@asynccontextmanager
async def local_project_turn(project: str) -> AsyncGenerator[None]:
    """Serialize acceptance against stop/delete and settle any old restoration first."""
    async with local_project_lock(project):
        await cancel_local_project_finalization(project)
        yield


def has_other_project_activity(project: str, stopped_session: str, sessions: Iterable[Any]) -> bool:
    return any(
        session.project_name == project
        and (bool(session.queued_messages) or (session.session_id != stopped_session and session.status == "running"))
        for session in sessions
    )


async def _finish_project(project: str) -> None:
    from lib.local_runtime.l20_resource_session import finalize_l20_runtime

    await finalize_l20_runtime(project=project)


def schedule_local_project_finalization(project: str, *, ready: Callable[[], bool] | None = None) -> None:
    global _coordinator
    if _coordinator is None:
        _coordinator = LocalSessionFinalizer(get_generation_queue().get_task_stats, _finish_project)
    _coordinator.request(project, ready=ready)


async def shutdown_local_project_finalizers() -> None:
    global _coordinator
    coordinator, _coordinator = _coordinator, None
    if coordinator is not None:
        await coordinator.close()


async def _finish_active_project() -> None:
    from lib.local_runtime.l20_resource_session import finalize_l20_runtime

    await finalize_l20_runtime()


async def drain_and_finalize_local_media(
    stop_worker: Callable[[], Awaitable[None]] | None,
    *,
    stop_pending: Callable[[], Awaitable[None]] = shutdown_local_project_finalizers,
    finish: Callable[[], Awaitable[None]] = _finish_active_project,
) -> None:
    """Cancel watchers, drain generation, and finalize with the configured model policy."""
    errors: list[BaseException] = []
    for operation in (stop_pending, stop_worker, finish):
        if operation is not None:
            try:
                await operation()
            except BaseException as exc:
                errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Local media shutdown failures", errors)
