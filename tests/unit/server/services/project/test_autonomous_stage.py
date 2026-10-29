"""Autonomous stages serialize native host calls without changing manual behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from lib.db.base import DEFAULT_USER_ID
from server.agent_toolset.declaration import Exempt, ToolDeclaration, invoke_declaration
from server.tool_runtime import CallerContext, ProjectScope, ToolOutcome


class NoArguments(BaseModel):
    pass


class Goal:
    def __init__(self, state="running"):
        self.state = state

    async def get_latest(self, project_name, user_id):
        return {"state": self.state} if self.state else None


class Queue:
    def __init__(self, tasks=()):
        self.tasks = list(tasks)

    async def list_tasks(self, *, project_name, status, page, page_size):
        tasks = [task for task in self.tasks if task["status"] == status]
        return {"items": tasks, "page_size": page_size, "total": len(tasks)}


def _declaration(name, handler):
    return ToolDeclaration(
        name=name,
        description="test",
        request_model=NoArguments,
        migration=Exempt("test"),
        domain_key="test",
        handler=handler,
    )


async def _invoke(tmp_path, name, handler, queue, goal):
    return await invoke_declaration(
        _declaration(name, handler),
        {},
        ProjectScope("demo", tmp_path),
        CallerContext(DEFAULT_USER_ID, "mcp"),
        SimpleNamespace(queue=queue),
        goal_store=goal,
    )


@pytest.mark.parametrize(
    "name",
    [
        "generate_storyboards",
        "generate_narration_audio",
        "patch_episode_script",
        "set_storyboard_end_frame",
        "select_image_version",
        "create_timeline",
        "render_final_cut",
        "finish_local_session",
    ],
)
async def test_active_goal_blocks_writers_before_handler_and_returns_exact_native_task_ids(tmp_path, name):
    writes = []

    async def handler(*args):
        writes.append(name)
        return ToolOutcome(value={"written": name})

    queue = Queue([{"task_id": "tts-original", "status": "running"}, {"task_id": "edit-original", "status": "queued"}])
    result = await _invoke(tmp_path, name, handler, queue, Goal())
    assert result.problem.code == "autonomous_stage_busy"
    assert result.problem.action == "wait_for_task"
    assert set(result.problem.params["task_ids"]) == {"tts-original", "edit-original"}
    assert result.value is None
    assert writes == []


async def test_manual_calls_are_unchanged_and_reads_do_not_even_query_goal(tmp_path):
    async def handler(*args):
        return ToolOutcome(value={"native_handler": "accepted"})

    queue = Queue([{"task_id": "running", "status": "running"}])
    result = await _invoke(tmp_path, "generate_storyboards", handler, queue, Goal(None))
    assert result.value == {"native_handler": "accepted"}

    class UnreadableGoal:
        async def get_latest(self, *args):
            raise AssertionError("Read tools must not query the goal store")

    result = await _invoke(tmp_path, "get_episode_production_status", handler, queue, UnreadableGoal())
    assert result.value == {"native_handler": "accepted"}


async def test_parallel_host_calls_admit_one_whole_batch_once(tmp_path):
    queue = Queue()
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(*args):
        started.set()
        await release.wait()
        queue.tasks.extend({"task_id": f"native-{index}", "status": "queued"} for index in range(3))
        return ToolOutcome(value={"batch_members": [task["task_id"] for task in queue.tasks]})

    first = asyncio.create_task(_invoke(tmp_path, "generate_storyboards", handler, queue, Goal()))
    await started.wait()
    second = asyncio.create_task(_invoke(tmp_path, "generate_narration_audio", handler, queue, Goal()))
    release.set()
    admitted, refused = await asyncio.gather(first, second)
    assert admitted.value == {"batch_members": ["native-0", "native-1", "native-2"]}
    assert refused.problem.code == "autonomous_stage_busy"
    assert len(queue.tasks) == 3


async def test_manual_parallel_tools_keep_their_existing_concurrency(tmp_path):
    entered = 0
    both_entered = asyncio.Event()

    async def handler(*args):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()
        return ToolOutcome(value={"manual": True})

    async with asyncio.timeout(2):
        outcomes = await asyncio.gather(
            *[_invoke(tmp_path, "generate_storyboards", handler, Queue(), Goal(None)) for _ in range(2)]
        )
    assert [outcome.value for outcome in outcomes] == [{"manual": True}, {"manual": True}]


async def test_stopping_goal_only_allows_idle_resource_finalization(tmp_path):
    async def handler(*args):
        return ToolOutcome(value={"native_handler": "accepted"})

    blocked = await _invoke(tmp_path, "generate_storyboards", handler, Queue(), Goal("stopping"))
    assert blocked.problem.code == "autonomous_stopping"
    finished = await _invoke(tmp_path, "finish_local_session", handler, Queue(), Goal("stopping"))
    assert finished.value == {"native_handler": "accepted"}


async def test_goal_closed_during_queue_read_refuses_the_unaccepted_write(tmp_path):
    class ClosingGoal:
        def __init__(self):
            self.reads = 0

        async def get_latest(self, *args):
            self.reads += 1
            return {"id": "same-goal", "state": "running" if self.reads == 1 else "stopped"}

    async def handler(*args):
        return ToolOutcome(value={"would_write": True})

    outcome = await _invoke(tmp_path, "generate_storyboards", handler, Queue(), ClosingGoal())
    assert outcome.value is None
    assert outcome.problem.code == "autonomous_goal_changed"
