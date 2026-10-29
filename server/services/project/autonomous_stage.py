"""Admission barrier for production writers while a native autonomous goal is active."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from server.agent_runtime.keyed_locks import KeyedLocks
from server.tool_runtime import CallerContext, ProjectScope, Services, ToolProblem


class GoalReader(Protocol):
    async def get_latest(self, project_name: str, user_id: str) -> dict[str, Any] | None: ...


PRODUCTION_WRITER_TOOLS = frozenset(
    {
        "edit_images",
        "patch_episode_script",
        "patch_episode_meta",
        "patch_project",
        "set_storyboard_end_frame",
        "select_image_version",
        "select_video_version",
        "retry_artifact_download",
        "create_timeline",
        "edit_timeline",
        "rename_timeline",
        "restore_revision",
        "render_final_cut",
        "export_jianying_draft",
        "finish_local_session",
        "confirm_script_review",
        "open_draft",
        "patch_draft",
        "promote_draft",
        "discard_draft",
        "plan_episodes",
        "reset_episode_planning",
        "complete_script_plan_rebuild",
        "split_grids",
        "rename_asset",
        "merge_asset",
        "edit_source_text",
        "upload_source",
        "delete_project_file",
        "restore_trash_record",
    }
)
_locks = KeyedLocks()


def _requested_episode(name: str, arguments: dict[str, Any], scope: ProjectScope, services: Services) -> int | None:
    if name == "render_final_cut":
        from lib.edit_timeline.store import EditTimelineStore

        return EditTimelineStore(services.projects, scope.project_name).find(arguments["timeline"]).episode
    project = services.projects.load_project(scope.project_name)
    filename = services.projects.normalize_script_filename(arguments["script"])
    matches = [
        row["episode"]
        for row in project.get("episodes", [])
        if row.get("script_file") and services.projects.normalize_script_filename(row["script_file"]) == filename
    ]
    return matches[0] if len(matches) == 1 else None


@asynccontextmanager
async def production_tool_admission(
    name: str,
    scope: ProjectScope,
    caller: CallerContext,
    services: Services,
    *,
    goal_store: GoalReader | None = None,
    visual_reader: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    arguments: dict[str, Any] | None = None,
) -> AsyncGenerator[ToolProblem | None]:
    if not name.startswith("generate_") and name not in PRODUCTION_WRITER_TOOLS:
        yield None
        return
    # Import lazily: constructing the tool schema must not open a runtime DB or
    # import the Hermes session implementation.
    from server.agent_runtime.autonomy_store import ACTIVE_STATES, AutonomyStore

    store = goal_store or AutonomyStore()
    goal = await store.get_latest(scope.project_name, caller.user_id)
    guarded_states = ACTIVE_STATES | {"attention"}
    if goal is None or goal.get("state") not in guarded_states:
        yield None
        return
    async with _locks.lock_for(scope.project_name):
        tasks: list[dict[str, Any]] = []
        for status in ("queued", "running"):
            page = 1
            while True:
                listed = await services.queue.list_tasks(
                    project_name=scope.project_name, status=status, page=page, page_size=500
                )
                tasks.extend(listed["items"])
                if page * listed["page_size"] >= listed["total"]:
                    break
                page += 1
        if tasks:
            yield ToolProblem(
                "autonomous_stage_busy",
                "Wait for the project's accepted tasks to finish before another production write.",
                action="wait_for_task",
                params={"task_ids": [task["task_id"] for task in tasks]},
            )
            return
        fresh = await store.get_latest(scope.project_name, caller.user_id)
        if fresh is None or fresh.get("state") not in guarded_states or fresh.get("id") != goal.get("id"):
            yield ToolProblem(
                "autonomous_goal_changed",
                "The production objective changed while this write was being admitted. Read its current state before continuing.",
            )
            return
        if fresh.get("state") == "stopping" and name != "finish_local_session":
            yield ToolProblem(
                "autonomous_stopping", "The autonomous goal is stopping; only local resource finalization is allowed."
            )
            return
        if name in {"generate_videos", "render_final_cut"}:
            episode_id, session_id = fresh.get("episode_id"), fresh.get("session_id")
            if type(episode_id) is not int or episode_id < 1 or not isinstance(session_id, str) or not session_id:
                yield ToolProblem("autonomous_goal_scope_unavailable", "The native goal session is not ready.")
                return
            if arguments is not None:
                episode = await asyncio.to_thread(_requested_episode, name, arguments, scope, services)
                if episode != episode_id:
                    yield ToolProblem(
                        "autonomous_episode_mismatch",
                        "This production write targets a different episode from the authorized objective.",
                        params={"authorized_episode": fresh.get("episode_id"), "requested_episode": episode},
                    )
                    return
            if visual_reader is None:
                from server.services.project.visual_observation import read_visual_observation_status

                visual_reader = read_visual_observation_status
            observation = await visual_reader(
                scope.project_name,
                episode_id,
                session_id,
                projects=services.projects,
                user_id=caller.user_id,
            )
            admitted = await store.get_latest(scope.project_name, caller.user_id)
            if (
                admitted is None
                or admitted.get("id") != fresh.get("id")
                or admitted.get("session_id") != fresh.get("session_id")
                or admitted.get("state") not in (guarded_states - {"stopping"})
            ):
                yield ToolProblem(
                    "autonomous_goal_changed",
                    "The production objective changed while its image observations were being checked.",
                )
                return
            if observation.get("ready") is not True:
                yield ToolProblem(
                    "autonomous_visual_observation_required",
                    "Inspect the actual current first and bound tail images before continuing. Model statements and MEDIA paths are not visual evidence.",
                    action="inspect_images",
                    params={
                        "episode_id": fresh.get("episode_id"),
                        "missing": observation.get("missing", []),
                        "inspection_requests": observation.get("inspection_requests", []),
                        "proof_digest": observation.get("proof_digest"),
                    },
                )
                return
        # The lock spans this one host invocation, including enqueue. Parallel
        # tool calls cannot both see an empty queue. A batch's internal enqueue
        # operations do not re-enter this boundary.
        yield None
