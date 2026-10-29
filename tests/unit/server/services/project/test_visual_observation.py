"""Only delivered current native frame pixels establish visual observation."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.generation.generation_result import decode_generation_problem
from lib.infra.image_utils import compress_image_bytes, image_pixel_sha256
from lib.local_runtime.config import get_local_runtime_config
from lib.project.project_manager import ProjectManager
from lib.project.project_schema import CURRENT_PROJECT_SCHEMA_VERSION
from server.agent_runtime.hermes_vision_bridge import VisionTransport
from server.services.project.visual_observation import (
    assert_frozen_video_frames_observed,
    read_visual_observation_status,
)


def _png(color, *, compress_level=6):
    stream = BytesIO()
    Image.new("RGB", (32, 24), color).save(stream, format="PNG", compress_level=compress_level)
    return stream.getvalue()


@pytest.fixture
def frames(tmp_path):
    directory = tmp_path / "projects/demo"
    (directory / "scripts").mkdir(parents=True)
    (directory / "storyboards").mkdir()
    (directory / "end_frames").mkdir()
    (directory / "project.json").write_text(
        json.dumps(
            {
                "schema_version": CURRENT_PROJECT_SCHEMA_VERSION,
                "title": "Demo",
                "content_mode": "narration",
                "generation_mode": "storyboard",
                "grid_storyboard": False,
                "style": "Anime",
                "style_description": "",
                "aspect_ratio": "16:9",
                "episodes": [{"episode": 2, "title": "Two", "script_file": "scripts/episode_2.json"}],
                "characters": {},
                "scenes": {},
                "props": {},
                "products": {},
            }
        )
    )
    (directory / "scripts/episode_2.json").write_text(
        json.dumps(
            {
                "episode": 2,
                "content_mode": "narration",
                "segments": [
                    {
                        "segment_id": "E2S01",
                        "novel_text": "正式旁白。",
                        "duration_seconds": 8,
                        "image_prompt": "Complete jar",
                        "segment_break": True,
                        "generated_assets": {"storyboard_image": "storyboards/scene_E2S01.png"},
                        "end_frame_image": "end_frames/scene_E2S01.png",
                    }
                ],
            }
        )
    )
    first, tail = _png("red"), _png("blue")
    first_path = directory / "storyboards/scene_E2S01.png"
    versions = VersionManager(directory)
    first_path.write_bytes(first)
    versions.add_version("storyboards", "E2S01", "A", source_file=first_path)
    first_path.write_bytes(tail)
    versions.add_version("storyboards", "E2S01", "B", source_file=first_path)
    versions.restore_version("storyboards", "E2S01", 1, first_path)
    # The bound tail has the same pixels as v2 but independent PNG encoding.
    (directory / "end_frames/scene_E2S01.png").write_bytes(_png("blue", compress_level=0))
    config = replace(get_local_runtime_config(), hermes_model="local-vlm", hermes_home=tmp_path / "hermes")
    return ProjectManager(tmp_path), directory, config, first, tail


def _row(data, version):
    return {
        "resource_type": "storyboard",
        "id": "E2S01",
        "version": version,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_pixel_sha256": image_pixel_sha256(data),
        "wire_image_sha256": hashlib.sha256(compress_image_bytes(data, max_long_edge=2000, quality=85)).hexdigest(),
    }


def _events(rows, *, transport="boot-a", session="sid", request=1):
    common = {
        "protocol": "arcreel-mcp-vision/v2",
        "transport_id": transport,
        "session_id": session,
        "request_id": request,
        "request_kind": "main",
        "model": "local-vlm",
        "body_sha256": "a" * 64,
        "endpoint_matches_configured": True,
        "model_matches_configured": True,
    }
    return [
        {
            **common,
            "event": "request",
            "accepted_for_send": True,
            "image_count": len(rows),
            "images": [{"sha256": row["wire_image_sha256"], "mime_type": "image/jpeg", "role": "tool"} for row in rows],
        },
        {**common, "event": "response", "status_code": 200},
        {**common, "event": "observed_images", "status_code": 200, "rows": rows},
    ]


def _audit(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in events))


async def _read(frames, path, *, session="sid"):
    projects, _, config, _, _ = frames
    return await read_visual_observation_status("demo", 2, session, projects=projects, config=config, audit_path=path)


async def test_matching_native_versions_and_reencoded_bound_tail_pass(frames, tmp_path):
    projects, directory, _, first, tail = frames
    path = tmp_path / "audit.jsonl"
    _audit(path, _events([_row(first, 1), _row(tail, 2)]))
    with projects.locked_project_script_snapshot("demo", "episode_2.json"):
        pass
    before = {str(file): file.read_bytes() for file in directory.rglob("*") if file.is_file()}
    result = await _read(frames, path)
    assert result["ready"] is True
    assert (result["required_count"], result["matched_count"]) == (2, 2)
    assert result["missing"] == result["inspection_requests"] == []
    assert before == {str(file): file.read_bytes() for file in directory.rglob("*") if file.is_file()}


@pytest.mark.parametrize("owned_parent", ["sid", "another-parent", None])
async def test_native_delegated_image_review_counts_only_for_its_own_parent(frames, tmp_path, owned_parent):
    *_, first, tail = frames
    path = tmp_path / "audit.jsonl"
    started = {
        "protocol": "arcreel-mcp-vision/v2",
        "event": "agent_started",
        "session_id": "review-child",
        "parent_session_id": owned_parent,
    }
    _audit(path, [started, *_events([_row(first, 1), _row(tail, 2)], session="review-child")])
    result = await _read(frames, path)
    assert result["ready"] is (owned_parent == "sid")
    assert result["matched_count"] == (2 if owned_parent == "sid" else 0)


@pytest.mark.parametrize(
    "change",
    [
        "captured_only",
        "no_response",
        "no_request",
        "rejected",
        "not_main",
        "wrong_model",
        "wrong_session",
        "wrong_boot",
        "wrong_body",
        "wire_not_sent",
        "fake_source",
        "v1",
    ],
)
async def test_capture_or_forged_or_unsent_observations_do_not_pass(frames, tmp_path, change):
    *_, first, tail = frames
    events = _events([_row(first, 1), _row(tail, 2)])
    if change == "captured_only":
        events = [{**events[0], "event": "mcp_images"}]
    elif change == "no_response":
        del events[1]
    elif change == "no_request":
        del events[0]
    elif change == "rejected":
        events[0]["accepted_for_send"] = False
    elif change == "not_main":
        for row in events:
            row["request_kind"] = "compression"
    elif change == "wrong_model":
        for row in events:
            row["model"] = "another-local-model"
    elif change == "wrong_session":
        for row in events:
            row["session_id"] = "other-sid"
    elif change == "wrong_boot":
        events[1]["transport_id"] = "old-boot"
    elif change == "wrong_body":
        events[1]["body_sha256"] = "b" * 64
    elif change == "wire_not_sent":
        events[0]["images"][0]["sha256"] = "b" * 64
    elif change == "fake_source":
        for row in events[2]["rows"]:
            row["source_sha256"] = "b" * 64
    else:
        for row in events:
            row["protocol"] = "arcreel-mcp-vision/v1"
    path = tmp_path / "audit.jsonl"
    _audit(path, events)
    result = await _read(frames, path)
    assert result["ready"] is False
    assert result["missing"]
    assert result["action"] == "inspect_images"


async def test_native_current_pixel_change_invalidates_an_old_observation(frames, tmp_path):
    _, directory, _, first, tail = frames
    path = tmp_path / "audit.jsonl"
    _audit(path, _events([_row(first, 1), _row(tail, 2)]))
    new_first = _png("green")
    first_path = directory / "storyboards/scene_E2S01.png"
    first_path.write_bytes(new_first)
    VersionManager(directory).add_version("storyboards", "E2S01", "C", source_file=first_path)
    result = await _read(frames, path)
    assert result["ready"] is False
    assert result["matched_count"] == 1
    assert result["inspection_requests"] == [{"resource_type": "storyboard", "ids": ["E2S01"], "version": 3}]


async def test_cold_recovery_uses_owned_same_sid_binding_and_unique_boot_identity(frames):
    projects, _, config, first, tail = frames
    home = config.hermes_home / "sessions/child"
    (config.hermes_home / "bindings").mkdir(parents=True)
    (config.hermes_home / "bindings/sid.json").write_text(
        json.dumps(
            {
                "schema": "arcreel-hermes-session/v1",
                "home": "sessions/child",
                "user_id": DEFAULT_USER_ID,
            }
        )
    )
    # Counters restart at 1 after cold recovery; independent boots cannot collide.
    events = _events([_row(first, 1)], transport="boot-a") + _events([_row(tail, 2)], transport="boot-b")
    _audit(home / "vision-wire-audit.jsonl", events)
    result = await read_visual_observation_status("demo", 2, "sid", projects=projects, config=config)
    assert result["ready"] is True
    binding = config.hermes_home / "bindings/sid.json"
    data = json.loads(binding.read_text())
    data["user_id"] = "other-user"
    binding.write_text(json.dumps(data))
    refused = await read_visual_observation_status("demo", 2, "sid", projects=projects, config=config)
    assert refused["ready"] is False
    assert refused["problem"] == "session_binding_unavailable"


async def test_boolean_version_cannot_impersonate_native_version_one(frames, tmp_path):
    *_, first, tail = frames
    rows = [_row(first, True), _row(tail, 2)]
    path = tmp_path / "audit.jsonl"
    _audit(path, _events(rows))
    result = await _read(frames, path)
    assert result["ready"] is False
    assert result["matched_count"] == 1


async def test_observed_row_cannot_cross_request_identity_in_cold_log(frames, tmp_path):
    *_, first, tail = frames
    old = _events([_row(first, 1), _row(tail, 2)], transport="old")
    forged = copy.deepcopy(old[2])
    forged["transport_id"] = "fresh"
    path = tmp_path / "audit.jsonl"
    _audit(path, [old[0], old[1], forged])
    assert (await _read(frames, path))["ready"] is False


async def test_real_bridge_receipt_is_consumed_after_cold_native_read(frames, tmp_path):
    *_, first, tail = frames
    home = tmp_path / "child"
    home.mkdir()

    def deliver():
        transport = VisionTransport(home, base_url="http://127.0.0.1:8000/v1", model="local-vlm")
        transport.begin_turn("sid")
        module = SimpleNamespace()

        def factory(*_args):
            def handler():
                rows = []
                for index, (data, version) in enumerate(((first, 1), (tail, 2)), 1):
                    jpeg = compress_image_bytes(data, max_long_edge=2000, quality=85)
                    module._cache_mcp_image_block(
                        SimpleNamespace(
                            type="image",
                            data=base64.b64encode(jpeg).decode(),
                            mimeType="image/jpeg",
                        )
                    )
                    rows.append({**_row(data, version), "image": index})
                return json.dumps(
                    {
                        "structuredContent": {
                            "inspect_images": {
                                "resource_type": "storyboard",
                                "images": rows,
                            }
                        }
                    }
                )

            return handler

        module._make_tool_handler = factory
        module._cache_mcp_image_block = lambda block: "MEDIA:unused"
        module._run_on_mcp_loop = lambda coro: coro()
        transport.install_mcp(module)
        result = module._make_tool_handler("arcreel", "inspect_images", 30)()

        class LocalClient(httpx.Client):
            pass

        http = SimpleNamespace(Client=LocalClient, RequestNotRead=httpx.RequestNotRead)
        transport.install_http(http)
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        with LocalClient(transport=httpx.MockTransport(respond)) as client:
            client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json={
                    "model": "local-vlm",
                    "messages": [{"role": "tool", "content": result["content"]}],
                },
            )
        assert len(requests) == 1

    await asyncio.to_thread(deliver)
    result = await _read(frames, home / "vision-wire-audit.jsonl")
    assert result["ready"] is True
    assert result["matched_count"] == 2


class FrozenGoal:
    def __init__(self, *, replace_after_read=False):
        self.calls = 0
        self.replace_after_read = replace_after_read

    async def get_latest(self, project_name, user_id):
        self.calls += 1
        return {
            "id": "replacement" if self.replace_after_read and self.calls > 1 else "goal",
            "episode_id": 2,
            "state": "running",
            "session_id": "sid",
        }


@pytest.mark.parametrize("frozen_seen", [True, False])
async def test_frozen_pixels_are_checked_independently_of_latest_current(frames, tmp_path, monkeypatch, frozen_seen):
    projects, directory, config, first, tail = frames
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    staged = directory / ".arcreel/tasks/task/provider_media"
    staged.mkdir(parents=True)
    frozen_first, frozen_tail = staged / "first.png", staged / "tail.png"
    frozen_first.write_bytes(first)
    frozen_tail.write_bytes(_png("blue", compress_level=0))
    # Canonical A3 is observed, which must never authorize a different frozen A1.
    new_first = _png("green")
    current = directory / "storyboards/scene_E2S01.png"
    current.write_bytes(new_first)
    VersionManager(directory).add_version("storyboards", "E2S01", "C", source_file=current)
    rows = [_row(new_first, 3), _row(tail, 2)]
    if frozen_seen:
        rows.append(_row(first, 1))
    audit = tmp_path / "audit.jsonl"
    _audit(audit, _events(rows))
    operation = assert_frozen_video_frames_observed(
        "demo",
        2,
        "E2S01",
        first=frozen_first,
        tail=frozen_tail,
        projects=projects,
        goal_store=FrozenGoal(),
        config=config,
        audit_path=audit,
    )
    if frozen_seen:
        await operation
    else:
        with pytest.raises(ValueError, match="autonomous_visual_observation_required") as refused:
            await operation
        problem = decode_generation_problem(str(refused.value))
        assert problem is not None
        assert problem.code == "autonomous_visual_observation_required"
        assert problem.params["inspection_requests"] == [
            {"resource_type": "storyboard", "ids": ["E2S01"], "version": 1}
        ]


async def test_frozen_observation_rechecks_goal_binding_after_audit_read(frames, tmp_path, monkeypatch):
    projects, directory, config, first, tail = frames
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    audit = tmp_path / "audit.jsonl"
    _audit(audit, _events([_row(first, 1), _row(tail, 2)]))
    with pytest.raises(ValueError, match="autonomous_visual_observation_required"):
        await assert_frozen_video_frames_observed(
            "demo",
            2,
            "E2S01",
            first=directory / "storyboards/scene_E2S01.png",
            tail=directory / "end_frames/scene_E2S01.png",
            projects=projects,
            goal_store=FrozenGoal(replace_after_read=True),
            config=config,
            audit_path=audit,
        )


async def test_manual_frozen_submission_does_not_read_goals_or_media(frames, monkeypatch):
    projects, directory, config, *_ = frames
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "false")
    store = FrozenGoal()
    await assert_frozen_video_frames_observed(
        "demo",
        2,
        "E2S01",
        first=directory / "missing.png",
        tail=None,
        projects=projects,
        goal_store=store,
        config=config,
    )
    assert store.calls == 0
