"""Production facts stay bound to native artifacts and explicit delivery variants."""

from __future__ import annotations

import copy
import hashlib
import json
import wave
from types import SimpleNamespace

import pytest

from lib.artifacts.artifact_activation import register_current_resource_artifact
from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.project.project_manager import ProjectManager
from lib.project.project_schema import CURRENT_PROJECT_SCHEMA_VERSION
from lib.script import script_review
from lib.script.draft_quarantine import QUARANTINE_KIND_NARRATION_SCRIPT_PLAN, quarantine_envelope, quarantine_path
from server.media_tools.production import SelectImageVersionRequest, select_image_version
from server.services.project.production_status import episode_final_cut_complete, read_episode_production_status
from server.tool_runtime import CallerContext, ProjectScope, ToolRequest


class QueueFacts:
    def __init__(self, tasks=()):
        self.tasks = list(tasks)

    async def list_tasks(self, *, project_name, status, page, page_size):
        rows = [task for task in self.tasks if task["project_name"] == project_name and task["status"] == status]
        return {"items": rows[(page - 1) * page_size : page * page_size], "page_size": page_size, "total": len(rows)}


class GoalFacts:
    def __init__(self, goal):
        self.goal = goal
        self.calls = []

    async def get_latest(self, project_name, user_id):
        self.calls.append((project_name, user_id))
        return self.goal


def _project(tmp_path, *, script=True):
    directory = tmp_path / "projects" / "demo"
    (directory / "scripts").mkdir(parents=True)
    project = {
        "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
        "title": "Demo",
        "content_mode": "narration",
        "generation_mode": "storyboard",
        "grid_storyboard": False,
        "style": "Anime",
        "style_description": "",
        "aspect_ratio": "16:9",
        "episodes": [{"episode": 2, "title": "Two", "script_file": "scripts/episode_2.json" if script else None}],
        "characters": {},
        "scenes": {},
        "props": {},
        "products": {},
    }
    (directory / "project.json").write_text(json.dumps(project), encoding="utf-8")
    if script:
        (directory / "scripts/episode_2.json").write_text(
            json.dumps(
                {
                    "episode": 2,
                    "content_mode": "narration",
                    "segments": [
                        {
                            "segment_id": "E2S01",
                            "novel_text": "这是正式旁白。",
                            "duration_seconds": 8,
                            "image_prompt": "A complete glass jar",
                            "segment_break": True,
                            "generated_assets": {
                                "storyboard_image": "storyboards/scene_E2S01.png",
                                "narration_audio": "audio/segment_E2S01.wav",
                            },
                            "end_frame_image": "end_frames/scene_E2S01.png",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    return ProjectManager(tmp_path), directory


@pytest.mark.parametrize("state", ["running", "attention"])
async def test_local_goal_status_requires_owned_main_visual_evidence(tmp_path, monkeypatch, state):
    projects, _ = _project(tmp_path)
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    goals = GoalFacts({"episode_id": 2, "state": state, "session_id": "sid"})
    status = await read_episode_production_status(
        "demo",
        2,
        projects=projects,
        queue=QueueFacts(),
        resource_state_path=tmp_path / "resource.json",
        goal_store=goals,
    )
    assert goals.calls == [("demo", DEFAULT_USER_ID)]
    assert status["visual_observation"]["enabled"] is True
    assert status["visual_observation"]["ready"] is False
    assert status["visual_observation"]["action"] == "inspect_images"
    assert status["goal_complete_technical"] is False


async def test_manual_status_preserves_delivery_predicate_without_goal_lookup(tmp_path, monkeypatch):
    projects, _ = _project(tmp_path)
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "false")
    goals = GoalFacts({"episode_id": 2, "state": "running", "session_id": "sid"})
    status = await read_episode_production_status(
        "demo",
        2,
        projects=projects,
        queue=QueueFacts(),
        resource_state_path=tmp_path / "resource.json",
        goal_store=goals,
    )
    assert goals.calls == []
    assert status["visual_observation"]["enabled"] is False


async def test_read_status_reports_exact_wav_and_all_native_writer_types_without_mutating(tmp_path):
    projects, directory = _project(tmp_path)
    (directory / "audio").mkdir()
    with wave.open(str(directory / "audio/segment_E2S01.wav"), "wb") as source:
        source.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        source.writeframes(b"\0\0" * 30000)
    (directory / "storyboards").mkdir()
    (directory / "storyboards/scene_E2S01.png").write_bytes(b"first-image")
    (directory / "end_frames").mkdir()
    (directory / "end_frames/scene_E2S01.png").write_bytes(b"independent-tail")
    versions = VersionManager(directory)
    versions.add_version("storyboards", "E2S01", "first", source_file=directory / "storyboards/scene_E2S01.png")
    tasks = [
        {
            "task_id": name,
            "project_name": "demo",
            "status": "running",
            "task_type": name,
            "provider_job_id": "accepted-" + name,
            "user_id": DEFAULT_USER_ID,
        }
        for name in ("tts", "image_edit", "render_final_cut")
    ]
    tasks += [
        {"task_id": "other-user", "project_name": "demo", "status": "running", "task_type": "tts", "user_id": "other"}
    ]
    state = tmp_path / "resource.json"
    state.write_text(json.dumps({"project": "demo", "state": "active", "private_capture": {"secret": "not returned"}}))
    before = {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = await read_episode_production_status(
        "demo", 2, projects=projects, queue=QueueFacts(tasks), resource_state_path=state
    )
    [unit] = result["units"]
    assert (unit["unit_id"], unit["text"], unit["declared_seconds"], unit["audio"]["actual_seconds"]) == (
        "E2S01",
        "这是正式旁白。",
        8,
        1.25,
    )
    assert unit["audio"]["current"] is False  # Existing bytes alone never fake a formal current claim.
    assert unit["first"]["selected_version"] == 1
    assert unit["first"]["versions"][0]["sha256"] == hashlib.sha256(b"first-image").hexdigest()
    assert unit["end_frame"]["sha256"] == hashlib.sha256(b"independent-tail").hexdigest()
    assert {task["task_type"] for task in result["active_tasks"]} == {"tts", "image_edit", "render_final_cut"}
    assert result["resource"] == {
        "owner_project": "demo",
        "state": "active",
        "active": True,
        "original_service_restored": False,
    }
    assert result["goal_complete_technical"] is False
    assert {str(path): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


async def test_existing_episode_without_formal_script_is_readable_but_unknown_episode_is_refused(tmp_path):
    projects, _ = _project(tmp_path, script=False)
    result = await read_episode_production_status(
        "demo", 2, projects=projects, queue=QueueFacts(), resource_state_path=tmp_path / "no-resource.json"
    )
    assert (result["script_state"], result["units"], result["goal_complete_technical"]) == ("missing", [], False)
    with pytest.raises(ValueError, match="does not belong"):
        await read_episode_production_status(
            "demo", 3, projects=projects, queue=QueueFacts(), resource_state_path=tmp_path / "no-resource.json"
        )


@pytest.mark.parametrize("state", ["running", "attention"])
async def test_owned_goal_with_rejected_plan_and_bound_missing_script_can_resume_draft_repair(
    tmp_path, monkeypatch, state
):
    projects, directory = _project(tmp_path)
    # Native source/planning may bind the future formal filename before its
    # first successful promotion. A rejected plan remains repairable input.
    (directory / "scripts/episode_2.json").unlink()
    (directory / "source").mkdir()
    source = directory / "source/episode_2.txt"
    source.write_text('他说："谢谢。"', encoding="utf-8")
    draft = quarantine_path(directory, 2, QUARANTINE_KIND_NARRATION_SCRIPT_PLAN)
    draft.parent.mkdir(parents=True)
    draft.write_text(
        json.dumps(
            quarantine_envelope(
                QUARANTINE_KIND_NARRATION_SCRIPT_PLAN,
                2,
                content={"segments": [{"novel_text": "他说：“谢谢。”", "duration_seconds": 8}]},
                violations=[{"code": "novel_text_coverage", "message": "Preserve the original quotation marks"}],
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    before = {path: path.read_bytes() for path in (source, draft)}
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    result = await read_episode_production_status(
        "demo",
        2,
        projects=projects,
        queue=QueueFacts(),
        resource_state_path=tmp_path / "no-resource.json",
        goal_store=GoalFacts({"episode_id": 2, "state": state, "session_id": "sid"}),
    )
    assert result["script_state"] == "missing"
    assert result["units"] == []
    assert result["preparation"]["episode_source"]["present"] is True
    assert result["preparation"]["drafts"][0]["needs_repair"] is True
    assert result["workflow"]["next_action"]["type"] == "resolve_draft"
    assert result["native_review"]["ready"] is False
    assert result["native_review"]["missing"][0]["problem"] == "formal_script_missing"
    assert result["goal_complete_technical"] is False
    assert {path: path.read_bytes() for path in before} == before
    assert not (directory / "scripts/episode_2.json").exists()


async def test_failed_facts_exclude_other_episodes_and_keep_only_latest_unresolved_resource(tmp_path):
    projects, _ = _project(tmp_path)
    tasks = [
        {
            "task_id": name,
            "project_name": "demo",
            "status": "failed",
            "task_type": "tts",
            "resource_id": resource_id,
            "script_file": script,
            "error_message": "native input refused",
        }
        for name, resource_id, script in (
            ("latest-E2", "E2S01", "episode_2.json"),
            ("older-E2", "E2S01", "episode_2.json"),
            ("old-E1", "E1S01", "episode_1.json"),
            ("future-E3", "E3S01", "episode_3.json"),
        )
    ]
    result = await read_episode_production_status(
        "demo", 2, projects=projects, queue=QueueFacts(tasks), resource_state_path=tmp_path / "resource-missing.json"
    )
    assert [task["task_id"] for task in result["failed_tasks"]] == ["latest-E2"]
    assert result["failed_tasks"][0]["target_episode_id"] == 2
    assert result["failed_tasks"][0]["relevance"] == "unresolved_episode_output"


async def test_native_preparation_fingerprint_tracks_source_draft_promotion_and_confirmation(tmp_path):
    projects, directory = _project(tmp_path, script=False)

    async def status():
        return await read_episode_production_status(
            "demo", 2, projects=projects, queue=QueueFacts(), resource_state_path=tmp_path / "no-resource.json"
        )

    initial = await status()
    (directory / "source").mkdir()
    (directory / "source/episode_2.txt").write_text("第二集原文", encoding="utf-8")
    source = await status()
    assert source["preparation"]["fingerprint"] != initial["preparation"]["fingerprint"]
    assert source["preparation"]["episode_source"]["present"] is True
    draft_file = quarantine_path(directory, 2, QUARANTINE_KIND_NARRATION_SCRIPT_PLAN)
    draft_file.parent.mkdir(parents=True)
    envelope = quarantine_envelope(
        QUARANTINE_KIND_NARRATION_SCRIPT_PLAN,
        2,
        content={"segments": [{"novel_text": "第二集原文", "duration_seconds": 8}]},
        violations=[],
    )
    draft_file.write_text(json.dumps(envelope), encoding="utf-8")
    draft = await status()
    assert draft["preparation"]["fingerprint"] != source["preparation"]["fingerprint"]
    assert draft["preparation"]["drafts"][0]["revision"].startswith("sha256-v1:")
    draft_file.write_text(json.dumps(envelope, sort_keys=True, indent=4), encoding="utf-8")
    assert (await status())["preparation"]["fingerprint"] == draft["preparation"]["fingerprint"]
    project = projects.load_project("demo")
    plan = script_review.script_plan_path(directory, project, 2)
    plan.write_text(json.dumps(envelope["content"]), encoding="utf-8")
    draft_file.unlink()
    promoted = await status()
    assert promoted["preparation"]["fingerprint"] != draft["preparation"]["fingerprint"]
    assert promoted["preparation"]["script_plan"]["review_state"] == "pending_review"
    script_review.apply_confirmation(project, 2, script_review.content_fingerprint(plan), "2026-10-29T00:00:00Z")
    (directory / "project.json").write_text(json.dumps(project), encoding="utf-8")
    confirmed = await status()
    assert confirmed["preparation"]["fingerprint"] != promoted["preparation"]["fingerprint"]
    assert confirmed["preparation"]["script_plan"]["review_state"] == "confirmed"
    assert "next_action" in confirmed["workflow"]


async def test_preparation_ignores_output_bookkeeping_but_not_formal_prompt_edits(tmp_path):
    projects, directory = _project(tmp_path)

    async def status():
        return await read_episode_production_status(
            "demo", 2, projects=projects, queue=QueueFacts(), resource_state_path=tmp_path / "no-resource.json"
        )

    initial = await status()
    script_path = directory / "scripts/episode_2.json"
    script = json.loads(script_path.read_text())
    script["metadata"] = {"updated_at": "2026-10-29T00:00:00Z"}
    script["segments"][0]["generated_assets"]["narration_audio"] = "audio/other.wav"
    script["segments"][0]["end_frame_image"] = "end_frames/other.png"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    outputs = await status()
    assert outputs["preparation"]["fingerprint"] == initial["preparation"]["fingerprint"]
    script["segments"][0]["image_prompt"] = "A different camera angle of the glass jar"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    assert (await status())["preparation"]["fingerprint"] != outputs["preparation"]["fingerprint"]


def _native_delivery(*, first_audio=7.5, second_audio=7.5, first_volume=0, second_volume=0):
    from lib.edit_timeline.model import (
        EditClip,
        EditTimelineContent,
        EditTimelineDocument,
        RevisionAuthor,
        TimelineRevision,
    )
    from lib.edit_timeline.readout import project_readout
    from lib.edit_timeline.sources import EpisodeScriptUnits, EpisodeSources, ScriptUnit, UnitMedia
    from lib.speech.speech_composition import SpeechMode

    ids = ["E2S01", "E2S02"]
    content = EditTimelineContent(
        clips=tuple(
            EditClip(id=f"c{index + 1}", unit_id=unit_id, source_volume=volume, carries_narration=True)
            for index, (unit_id, volume) in enumerate(zip(ids, [first_volume, second_volume], strict=True))
        )
    )
    revision = TimelineRevision(
        number=1,
        author=RevisionAuthor(kind="arcreel_agent"),
        summary="Native test timeline",
        created_at="2026-10-29T00:00:00Z",
        content=content,
    )
    document = EditTimelineDocument(
        id="tl-12345678",
        episode=2,
        name="Episode two",
        created_at=revision.created_at,
        next_clip_number=3,
        revisions=(revision,),
    )
    script = EpisodeScriptUnits(
        2, "segments", tuple(ScriptUnit(unit_id, SpeechMode.NARRATOR_VOICEOVER, 8_000_000, "test") for unit_id in ids)
    )
    sources = EpisodeSources(
        script,
        {
            unit_id: UnitMedia(1, 8_000_000, round(seconds * 1_000_000))
            for unit_id, seconds in zip(ids, [first_audio, second_audio], strict=True)
        },
        True,
    )
    readout = project_readout(document, revision, sources)
    units = [
        {"unit_id": unit_id, "audio": {"status": "current", "actual_seconds": seconds}}
        for unit_id, seconds in zip(ids, [first_audio, second_audio], strict=True)
    ]
    timeline = {
        "timeline_id": document.id,
        "episode_id": 2,
        "unit_ids": ids,
        "readout": readout.model_dump(mode="json"),
        "final_cut": {
            "episode": 2,
            "timeline_id": document.id,
            "status": "current",
            "version": 1,
            "narration": "with_narration",
            "subtitles": "burned_subtitles",
        },
    }
    return units, timeline


def test_completion_requires_exact_episode_units_and_narrated_subtitled_current_final_cut():
    units, valid = _native_delivery(first_audio=7.532416)
    assert episode_final_cut_complete(2, units, [valid]) is True
    for key, value in (
        ("status", "stale"),
        ("episode", 1),
        ("narration", "without_narration"),
        ("subtitles", "no_subtitles"),
    ):
        invalid = copy.deepcopy(valid)
        invalid["final_cut"][key] = value
        assert episode_final_cut_complete(2, units, [invalid]) is False
    assert episode_final_cut_complete(2, [*units, {"unit_id": "E2S03"}], [valid]) is False
    assert episode_final_cut_complete(2, list(reversed(units)), [valid]) is False
    assert episode_final_cut_complete(2, units, []) is False


@pytest.mark.parametrize("case", ["overlap", "short_last_clip", "unmuted", "source_collision", "stale_audio"])
def test_autonomous_completion_refuses_native_warning_only_speech_loss_and_raw_audio(case):
    options = {
        "overlap": {"first_audio": 9},
        "short_last_clip": {"second_audio": 9},
        "unmuted": {"first_volume": 0.2},
        "source_collision": {"first_audio": 9, "second_volume": 1},
    }
    units, timeline = _native_delivery(**options.get(case, {}))
    if case == "stale_audio":
        units[0]["audio"]["status"] = "stale"
    if case in {"overlap", "short_last_clip", "source_collision"}:
        warnings = [
            issue
            for issue in timeline["readout"]["issues"]
            if issue["code"] in {"narration_overrun", "narration_source_collision"}
        ]
        assert warnings
        assert all(issue["severity"] == "warning" for issue in warnings)
    assert timeline["final_cut"]["status"] == "current"
    assert episode_final_cut_complete(2, units, [timeline]) is False


async def test_select_image_version_uses_native_restore_and_preserves_independent_tail(tmp_path):
    projects, directory = _project(tmp_path)
    current = directory / "storyboards/scene_E2S01.png"
    current.parent.mkdir()
    current.write_bytes(b"original-first")
    versions = VersionManager(directory)
    versions.add_version("storyboards", "E2S01", "first", source_file=current, source="manual_upload")
    current.write_bytes(b"edited-tail")
    versions.add_version("storyboards", "E2S01", "tail", source_file=current, source="manual_upload")
    tail = directory / "end_frames/scene_E2S01.png"
    tail.parent.mkdir()
    tail.write_bytes(b"edited-tail")
    register_current_resource_artifact(directory, resource_type="storyboards", resource_id="E2S01")
    services = SimpleNamespace(projects=projects)
    outcome = await select_image_version(
        ToolRequest(SelectImageVersionRequest(resource_type="storyboard", resource_id="E2S01", version=1)),
        ProjectScope("demo", tmp_path),
        CallerContext(DEFAULT_USER_ID, "mcp"),
        services,
    )
    assert outcome.problem is None
    assert outcome.value["current_version"] == 1
    assert current.read_bytes() == b"original-first"
    assert tail.read_bytes() == b"edited-tail"
    assert (
        projects.load_script_readonly("demo", "episode_2.json")["segments"][0]["end_frame_image"]
        == "end_frames/scene_E2S01.png"
    )
    assert VersionManager(directory).get_current_version("storyboards", "E2S01") == 1
