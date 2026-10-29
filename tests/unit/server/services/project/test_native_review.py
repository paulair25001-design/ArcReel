"""Native review execution covers exact selected versions, not semantic approval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.local_runtime.config import get_local_runtime_config
from lib.project.project_manager import ProjectManager
from server.services.project.native_review import read_native_review_status


@pytest.fixture
def review_case(tmp_path):
    projects = ProjectManager(tmp_path / "data")
    projects.create_project("demo")
    projects.create_project_metadata("demo", "Demo", "Anime", "narration")
    projects.save_script(
        "demo",
        {
            "episode": 2,
            "content_mode": "narration",
            "segments": [
                {"segment_id": unit_id, "novel_text": "完整旁白。", "duration_seconds": 8}
                for unit_id in ("E2S01", "E2S02")
            ],
        },
        "episode_2.json",
        validate=False,
    )
    directory = projects.get_project_path("demo")
    versions = VersionManager(directory)
    staged = tmp_path / "video.mp4"
    for unit_id, version in (("E2S01", 1), ("E2S01", 2), ("E2S02", 1)):
        staged.write_bytes(f"native immutable snapshot {unit_id} v{version}".encode())
        versions.add_version("videos", unit_id, "Native prompt", source_file=staged)
    profile = tmp_path / "profile"
    definition = profile / ".claude/agents/review-footage.md"
    definition.parent.mkdir(parents=True)
    definition.write_text("---\nname: review-footage\n---\nRead the native clips and report concerns.\n")
    config = replace(get_local_runtime_config(), hermes_home=tmp_path / "hermes", hermes_model="local-qwen")
    home = config.hermes_home / "sessions/parent"
    home.mkdir(parents=True)
    binding = config.hermes_home / "bindings/parent-sid.json"
    binding.parent.mkdir()
    binding.write_text(
        json.dumps({"schema": "arcreel-hermes-session/v1", "home": "sessions/parent", "user_id": DEFAULT_USER_ID})
    )
    return {
        "projects": projects,
        "directory": directory,
        "config": config,
        "profile_dir": profile,
        "definition": definition,
        "audit": home / "native-agent-reports.jsonl",
        "binding": binding,
    }


def _report(case, **changes):
    return {
        "schema": "arcreel-native-agent-report/v1",
        "parent_session_id": "parent-sid",
        "session_id": "native-review-child",
        "agent_type": "review-footage",
        "status": "completed",
        "report": "E2S01 v2: 可用。E2S02 v1: 存疑，建议修复手部。",
        "inspections": [{"unit_id": "E2S01", "version": 2}, {"unit_id": "E2S02", "version": 1}],
        "profile_sha256": hashlib.sha256(case["definition"].read_bytes()).hexdigest(),
        "model": "local-qwen",
        **changes,
    }


def _write_reports(case, rows):
    case["audit"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


async def _read(case, **kwargs):
    return await read_native_review_status(
        "demo",
        2,
        "parent-sid",
        projects=case["projects"],
        config=case["config"],
        profile_dir=case["profile_dir"],
        **kwargs,
    )


async def test_completed_native_report_covers_current_versions_without_claiming_quality_pass(review_case):
    _write_reports(review_case, [_report(review_case)])
    result = await _read(review_case)

    assert result["ready"] is True
    assert (result["required_count"], result["reviewed_count"]) == (2, 2)
    assert result["missing"] == []
    assert result["action"] is None
    assert {(row["unit_id"], row["version"]) for row in result["reviewed"]} == {("E2S01", 2), ("E2S02", 1)}
    assert all(row["session_id"] == "native-review-child" for row in result["reviewed"])
    assert "建议修复" not in json.dumps(result, ensure_ascii=False)
    assert "quality" not in result
    assert "approved" not in result


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "another-report/v1"},
        {"parent_session_id": "other-parent"},
        {"session_id": ""},
        {"session_id": "parent-sid"},
        {"agent_type": "review-images"},
        {"status": "running"},
        {"status": "failed"},
        {"model": "another-model"},
        {"profile_sha256": "a" * 64},
        {"report": " \n\t"},
        {"inspections": []},
        {"inspections": [{"unit_id": "E2S01", "version": True}]},
        {"inspections": [{"unit_id": "E2S01", "version": "2"}]},
    ],
)
async def test_unfinished_or_wrong_scope_or_self_declared_report_does_not_establish_coverage(review_case, changes):
    _write_reports(review_case, [_report(review_case, **changes)])
    result = await _read(review_case)

    assert result["ready"] is False
    assert result["reviewed_count"] == 0
    assert result["action"] == "dispatch_agent"
    assert {row["unit_id"] for row in result["missing"]} == {"E2S01", "E2S02"}


async def test_reports_cover_only_the_requested_native_versions(review_case):
    _write_reports(
        review_case,
        [_report(review_case, inspections=[{"unit_id": "E2S01", "version": 1}, {"unit_id": "E2S02", "version": 1}])],
    )
    current = await _read(review_case)
    assert current["ready"] is False
    assert current["missing"] == [{"unit_id": "E2S01", "version": 2, "problem": "native_review_report_missing"}]

    selected = await _read(review_case, required_versions=[{"unit_id": "E2S01", "version": 1}])
    assert selected["ready"] is True
    assert selected["required"] == [{"unit_id": "E2S01", "version": 1}]


async def test_new_selected_version_requires_its_own_native_report(review_case, tmp_path):
    _write_reports(review_case, [_report(review_case)])
    assert (await _read(review_case))["ready"] is True
    staged = tmp_path / "replacement.mp4"
    staged.write_bytes(b"replacement native version")
    VersionManager(review_case["directory"]).add_version("videos", "E2S01", "New prompt", source_file=staged)

    changed = await _read(review_case)
    assert changed["ready"] is False
    assert changed["missing"] == [{"unit_id": "E2S01", "version": 3, "problem": "native_review_report_missing"}]


async def test_report_requires_owned_parent_binding_on_cold_recovery(review_case):
    _write_reports(review_case, [_report(review_case)])
    assert (await _read(review_case))["ready"] is True
    binding = json.loads(review_case["binding"].read_text())
    binding["user_id"] = "another-user"
    review_case["binding"].write_text(json.dumps(binding))

    refused = await _read(review_case)
    assert refused["ready"] is False
    assert refused["problem"] == "session_binding_unavailable"
    assert refused["reviewed"] == []


async def test_current_profile_change_invalidates_previous_instruction_receipt(review_case):
    _write_reports(review_case, [_report(review_case)])
    assert (await _read(review_case))["ready"] is True
    review_case["definition"].write_text("---\nname: review-footage\n---\nUpdated native instructions.\n")

    assert (await _read(review_case))["ready"] is False


async def test_multiple_native_reports_and_partial_append_preserve_completed_coverage(review_case):
    _write_reports(
        review_case,
        [
            _report(review_case, session_id="group-one", inspections=[{"unit_id": "E2S01", "version": 2}]),
            _report(review_case, session_id="group-two", inspections=[{"unit_id": "E2S02", "version": 1}]),
        ],
    )
    with review_case["audit"].open("a") as stream:
        stream.write('{"schema":')

    result = await _read(review_case)
    assert result["ready"] is True
    assert {row["session_id"] for row in result["reviewed"]} == {"group-one", "group-two"}


@pytest.mark.parametrize(
    ("selection", "problem"),
    [
        ({"unit_id": "E1S01", "version": 2}, "video_unit_not_in_episode"),
        ({"unit_id": "E2S01", "version": True}, "video_version_missing"),
        ({"unit_id": "E2S01", "version": 99}, "video_version_unavailable"),
    ],
)
async def test_timeline_requirement_cannot_name_foreign_or_nonexistent_versions(review_case, selection, problem):
    _write_reports(review_case, [_report(review_case, inspections=[selection])])
    result = await _read(review_case, required_versions=[selection])

    assert result["ready"] is False
    assert result["missing"] == [{**selection, "problem": problem}]


async def test_explicit_audit_dependency_reads_only_the_supplied_fixture(review_case, tmp_path):
    alternate = tmp_path / "isolated-audit.jsonl"
    alternate.write_text(json.dumps(_report(review_case)))
    review_case["binding"].unlink()

    assert (await _read(review_case, audit_path=alternate))["ready"] is True
    assert (await _read(review_case))["ready"] is False
