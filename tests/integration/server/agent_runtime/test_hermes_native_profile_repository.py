"""The real embedded profile remains the sole source for every supported mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lib.agent.profile_manifest import ContentMode, resolve_profile_files_for_mode
from server.agent_runtime.hermes_native_profile import load_hermes_native_profile


@pytest.mark.parametrize("mode", ["narration", "drama", "ad"])
def test_all_native_mode_definitions_and_review_standards_survive_host_projection(mode: ContentMode):
    root = Path(__file__).resolve().parents[4] / "agent_runtime_profile"
    native = load_hermes_native_profile(mode, profile_root=root)
    mapping = resolve_profile_files_for_mode(root, mode)
    markdown = {key: value for key, value in mapping.items() if key.endswith(".md")}
    assert set(native.documents) == set(markdown)
    for logical, source in markdown.items():
        original = (root / source).read_bytes()
        document = native.read_document(logical)
        assert document.content.encode("utf-8") == original
        assert document.source_sha256 == hashlib.sha256(original).hexdigest()
    assert native.read_document("CLAUDE.md").source_path == f"CLAUDE.{mode}.md"
    assert native.read_skill("video-workflow").source_path.endswith(f"SKILL.{mode}.md")

    manifest = json.loads(json.dumps(native.to_manifest("demo")))
    assert len(manifest["agents"]) == 7
    images = manifest["agents"]["review-images"]
    footage = manifest["agents"]["review-footage"]
    assert "mcp__arcreel__inspect_images" in images["tools"]
    assert "mcp__arcreel__inspect_video_units" in footage["tools"]
    assert "崩坏" in images["prompt"]
    assert "角色不一致" in images["prompt"]
    assert "可用区间" in footage["prompt"]
    assert "版本排序" in footage["prompt"]
    assert native.read_document("CLAUDE.md").content in footage["prompt"]
    assert "首轮全量审阅" in native.read_skill("edit-video").content
    assert "审核检查点：审图" in native.read_skill("generate-storyboard").content
    assert "质量检查" in native.read_skill("generate-assets").content
    assert manifest["agents"]["create-episode-script"]["skills"] == ["generate-script"]
    assert native.read_skill("generate-script").content in manifest["agents"]["create-episode-script"]["prompt"]
    assert native.read_document("CLAUDE.md").content in native.bootstrap_prompt("demo")
    for restricted in (images, footage):
        assert "read_file" in restricted["tools"]
        assert "dispatch_agent" not in restricted["tools"]
        assert not any(tool in {"Read", "Write", "Edit", "Bash", "Glob", "Grep"} for tool in restricted["tools"])
        assert not any("generate_" in tool or "patch_" in tool for tool in restricted["tools"])
