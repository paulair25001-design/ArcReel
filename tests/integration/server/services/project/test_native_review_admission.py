"""Native render admission requires executed footage review and a live goal."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace

import pytest

from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.edit_timeline.model import (
    EditClip,
    EditTimelineContent,
    EditTimelineDocument,
    RevisionAuthor,
    TimelineRevision,
)
from lib.edit_timeline.store import EditTimelineStore
from lib.generation.generation_queue import GenerationQueue
from lib.local_runtime.config import get_local_runtime_config
from lib.project.project_manager import ProjectManager
from server.agent_runtime.autonomy_store import AutonomyStore
from server.services.project.autonomous_stage import production_tool_admission
from server.services.project.native_review import read_native_review_status
from tests.integration.server.agent_tool_support import ToolHarness


@pytest.fixture
async def admission_case(tmp_path, db_factory):
    projects = ProjectManager(tmp_path / "data")
    projects.create_project("demo")
    projects.create_project_metadata("demo", "Demo", "Anime", "narration")
    unit_ids = ["E2S01", "E2S02"]
    projects.save_script(
        "demo",
        {
            "episode": 2,
            "content_mode": "narration",
            "segments": [{"segment_id": unit, "novel_text": "旁白原文。", "duration_seconds": 8} for unit in unit_ids],
        },
        "episode_2.json",
        validate=False,
    )
    version_manager = VersionManager(projects.get_project_path("demo"))
    staged = tmp_path / "video.mp4"
    for unit in unit_ids:
        staged.write_bytes(("native video " + unit).encode())
        version_manager.add_version("videos", unit, "Native visual prompt", source_file=staged)
    timeline = EditTimelineDocument(
        id="tl-1234abcd",
        episode=2,
        name="原生时间线",
        created_at="2026-10-29T00:00:00Z",
        next_clip_number=3,
        revisions=(
            TimelineRevision(
                number=1,
                author=RevisionAuthor(kind="arcreel_agent"),
                summary="Native clips",
                created_at="2026-10-29T00:00:00Z",
                content=EditTimelineContent(
                    clips=tuple(
                        EditClip(id=f"c{index}", unit_id=unit, source_volume=0, carries_narration=True)
                        for index, unit in enumerate(unit_ids, 1)
                    )
                ),
            ),
        ),
    )
    timelines = EditTimelineStore(projects, "demo")
    with timelines.locked_episode(2):
        timelines.write(timeline)
    queue = GenerationQueue(session_factory=db_factory, project_manager=projects)
    ctx = ToolHarness("demo", tmp_path / "data", projects, queue=queue)
    store = AutonomyStore(session_factory=db_factory)
    goal = await store.create("demo", 2, "Complete native episode", DEFAULT_USER_ID, "native-review-test")
    await store.claim(goal["id"], "driver", 60, user_id=DEFAULT_USER_ID)
    await store.update(goal["id"], "driver", {"state": "running", "session_id": "parent-sid"}, user_id=DEFAULT_USER_ID)
    profile = tmp_path / "profile"
    definition = profile / ".claude/agents/review-footage.md"
    definition.parent.mkdir(parents=True)
    definition.write_text("---\nname: review-footage\n---\nNative read-only footage review.\n")
    config = replace(get_local_runtime_config(), hermes_home=tmp_path / "hermes", hermes_model="local-qwen")
    home = config.hermes_home / "sessions/parent"
    home.mkdir(parents=True, mode=0o700)
    bindings = config.hermes_home / "bindings"
    bindings.mkdir()
    (bindings / "parent-sid.json").write_text(
        json.dumps(
            {
                "schema": "arcreel-hermes-session/v1",
                "home": "sessions/parent",
                "user_id": DEFAULT_USER_ID,
            }
        )
    )

    async def native_review(project_name, episode_id, session_id, **kwargs):
        return await read_native_review_status(
            project_name,
            episode_id,
            session_id,
            config=config,
            profile_dir=profile,
            **kwargs,
        )

    return {
        "ctx": ctx,
        "store": store,
        "goal": goal,
        "timeline": timeline.id,
        "review": native_review,
        "audit": home / "native-agent-reports.jsonl",
        "profile_sha256": hashlib.sha256(definition.read_bytes()).hexdigest(),
    }


def _complete_report(case):
    case["audit"].write_text(
        json.dumps(
            {
                "schema": "arcreel-native-agent-report/v1",
                "parent_session_id": "parent-sid",
                "session_id": "child-review",
                "agent_type": "review-footage",
                "status": "completed",
                "model": "local-qwen",
                "profile_sha256": case["profile_sha256"],
                "report": "E2S01 v1 可用；E2S02 v1 局部存疑，建议交付时说明。",
                "inspections": [{"unit_id": unit, "version": 1} for unit in ("E2S01", "E2S02")],
            }
        )
    )


async def _visual_prerequisite(*_args, **_kwargs):
    return {"ready": True}


async def _admit(case, *, review_reader=None):
    ctx = case["ctx"]
    async with production_tool_admission(
        "render_final_cut",
        ctx.scope,
        ctx.caller,
        ctx.services,
        goal_store=case["store"],
        visual_reader=_visual_prerequisite,
        review_reader=review_reader or case["review"],
        arguments={"timeline": case["timeline"]},
    ) as problem:
        return problem


@pytest.mark.parametrize("reported", [False, True])
async def test_native_render_requires_reports_for_current_video_versions(admission_case, reported):
    if reported:
        _complete_report(admission_case)
    problem = await _admit(admission_case)

    if reported:
        # A report may contain concerns: admission establishes execution, not a quality verdict.
        assert problem is None
    else:
        assert problem.code == "native_footage_review_required"
        assert problem.action == "dispatch_agent"
        assert {row["unit_id"] for row in problem.params["missing"]} == {"E2S01", "E2S02"}
    stats = await admission_case["ctx"].queue.get_task_stats("demo")
    assert stats["queued"] == stats["running"] == 0


async def test_stop_during_native_review_read_refuses_the_unaccepted_render(admission_case):
    _complete_report(admission_case)
    inspected, release = asyncio.Event(), asyncio.Event()

    async def paused_review(*args, **kwargs):
        result = await admission_case["review"](*args, **kwargs)
        assert result["ready"] is True
        inspected.set()
        await release.wait()
        return result

    pending = asyncio.create_task(_admit(admission_case, review_reader=paused_review))
    await inspected.wait()
    stopped = await admission_case["store"].stop(admission_case["goal"]["id"], DEFAULT_USER_ID)
    assert stopped["state"] == "stopping"
    release.set()
    problem = await pending

    assert problem.code == "autonomous_goal_changed"
    assert (await admission_case["store"].get_latest("demo", DEFAULT_USER_ID))["state"] == "stopping"
    stats = await admission_case["ctx"].queue.get_task_stats("demo")
    assert stats["queued"] == stats["running"] == 0
