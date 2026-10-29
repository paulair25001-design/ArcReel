"""The real storyboard admission path requires an explicit L20 target tail."""

from pathlib import Path

import pytest
from PIL import Image

from server.services.admission.video_batch_admission import build_storyboard_video_specs
from tests.integration.server.media_tools.test_videos import _activated_project


def test_storyboard_preflight_refuses_missing_tail_and_admits_explicit_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    first = tmp_path / "storyboards/scene_S01.png"
    first.parent.mkdir()
    Image.new("RGB", (576, 1024), "red").save(first)
    project = _activated_project(tmp_path, {"S01": "storyboards/scene_S01.png"})
    item = {
        "segment_id": "S01",
        "novel_text": "他在旷野上奔跑。",
        "video_prompt": "一个奔跑的镜头",
        "duration_seconds": 2,
        "generated_assets": {"storyboard_image": "storyboards/scene_S01.png"},
    }

    def preflight():
        return build_storyboard_video_specs(
            items=[item],
            id_field="segment_id",
            content_mode="narration",
            skeleton_kind="segments",
            script_filename="episode_1.json",
            project_dir=tmp_path,
            project=project,
            skip_ids=None,
        )

    specs, refused = preflight()
    assert specs == []
    assert len(refused) == 1
    assert refused[0].unit_id == "S01"
    assert refused[0].problems[0].params["required_input"] == "end_frame_image"
    assert refused[0].problems[0].params["tool"] == "set_storyboard_end_frame"

    tail = tmp_path / "end_frames/scene_S01.png"
    tail.parent.mkdir()
    Image.new("RGB", (576, 1024), "blue").save(tail)
    item["end_frame_image"] = "end_frames/scene_S01.png"
    specs, refused = preflight()
    assert refused == []
    assert [spec.resource_id for spec in specs] == ["S01"]
