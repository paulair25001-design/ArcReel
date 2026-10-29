"""Hermes reads native definitions unchanged and maps only host capabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from server.agent_runtime.hermes_native_profile import load_hermes_native_profile


def profile_root(tmp_path: Path) -> Path:
    files = {
        "CLAUDE.narration.md": "# Native narration persona\nKeep narration unchanged.\n",
        "CLAUDE.drama.md": "# Native drama persona\nKeep dialogue unchanged.\n",
        "CLAUDE.ad.md": "# Native advertising persona\n",
        ".claude/skills/video-workflow/SKILL.narration.md": "---\nname: video-workflow\n---\nNative narration route.\n",
        ".claude/skills/video-workflow/SKILL.drama.md": "---\nname: video-workflow\n---\nNative drama route.\n",
        ".claude/skills/video-workflow/SKILL.ad.md": "---\nname: video-workflow\n---\nNative ad route.\n",
        ".claude/skills/generate-assets/SKILL.md": "---\nname: generate-assets\ndescription: Native quality checkpoint\n---\nReview every generated image.\n",
        ".claude/skills/generate-assets/references/layout.md": "Native layout: preserve identity.\n",
        ".claude/references/generation-results.md": "Native result and repair contract.\n",
        ".claude/agents/review-images.md": "---\nname: review-images\ntools: Read, Glob, Grep, mcp__arcreel__inspect_images\n---\nCheck anatomy, identity and prompt compliance.\n",
        ".claude/agents/create-episode-script.md": "---\nname: create-episode-script\nskills:\n - generate-assets\n---\nRun the native script operation.\n",
        ".claude/settings.json": '{"secret":"not-a-profile-document"}',
        ".claude/skills/generate-assets/scripts/run.py": "raise RuntimeError('not executable in Hermes')",
    }
    root = tmp_path / "profile"
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


def test_mode_projection_preserves_every_document_original_bytes_and_sha(tmp_path: Path):
    root = profile_root(tmp_path)
    native = load_hermes_native_profile("narration", profile_root=root)
    assert native.read_document("CLAUDE.md").content == (root / "CLAUDE.narration.md").read_text()
    assert native.read_skill("video-workflow").source_path.endswith("SKILL.narration.md")
    for document in native.documents.values():
        raw = (root / document.source_path).read_bytes()
        assert document.content.encode() == raw
        assert document.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert ".claude/settings.json" not in native.documents
    assert not any(path.endswith(".py") for path in native.documents)
    assert len(native.documents) == 7

    # The selected source is resolved by the same native mode projection.
    drama = load_hermes_native_profile("drama", profile_root=root)
    assert drama.read_skill("video-workflow").source_path.endswith("SKILL.drama.md")
    assert drama.source_sha256 != native.source_sha256
    assert "drama persona" in drama.bootstrap_prompt("demo")


def test_manifest_keeps_agent_frontmatter_preloads_and_read_only_review_tools(tmp_path: Path):
    native = load_hermes_native_profile("narration", profile_root=profile_root(tmp_path))
    manifest = json.loads(json.dumps(native.to_manifest("demo")))
    assert manifest["schema"] == "arcreel-hermes-native-profile/v1"
    assert manifest["project"] == "demo"
    assert manifest["source_sha256"] == native.source_sha256
    assert manifest["read_profile_tool"] == "read_agent_profile"
    assert set(manifest["agents"]) == {"create-episode-script", "review-images"}
    review = manifest["agents"]["review-images"]
    assert review["native_tools"] == ["Read", "Glob", "Grep", "mcp__arcreel__inspect_images"]
    assert "read_agent_profile" in review["tools"]
    assert "read_file" in review["tools"]
    assert "mcp__arcreel__get_project_content" in review["tools"]
    assert "mcp__arcreel__inspect_images" in review["tools"]
    assert "Read" not in review["tools"]
    assert "dispatch_agent" not in review["tools"]
    assert not any("generate" in name or "patch" in name for name in review["tools"])
    assert native.read_agent("review-images").content in review["prompt"]
    assert native.read_document("CLAUDE.md").content in review["prompt"]
    creation = manifest["agents"]["create-episode-script"]
    assert creation["tools"] is None
    assert creation["skills"] == ["generate-assets"]
    assert native.read_skill("generate-assets").content in creation["prompt"]
    assert manifest["documents"][".claude/agents/review-images.md"]["metadata"]["tools"].startswith("Read,")


def test_bootstrap_has_full_mode_persona_and_navigation_without_all_skill_bodies(tmp_path: Path):
    native = load_hermes_native_profile("narration", profile_root=profile_root(tmp_path))
    prompt = native.bootstrap_prompt("demo", read_profile_tool="profile_read", dispatch_tool="role_dispatch")
    assert native.read_document("CLAUDE.md").content in prompt
    assert ".claude/skills/generate-assets/SKILL.md" in prompt
    assert "Review every generated image." not in prompt
    assert "Check anatomy, identity and prompt compliance." not in prompt
    assert "profile_read" in prompt
    assert "role_dispatch" in prompt
    assert "mcp__arcreel__get_episode_script" in prompt
    assert "home/cache/spillover/" in prompt
    assert "不提供通用文件访问" in prompt
    assert "本适配说明不授予自主制作权限" in prompt
    assert native.agent_spec("review-images", read_profile_tool="profile_read").allowed_tools[0] == "profile_read"


def test_relative_links_resolve_only_to_known_current_mode_documents(tmp_path: Path):
    native = load_hermes_native_profile("narration", profile_root=profile_root(tmp_path))
    workflow = ".claude/skills/video-workflow/SKILL.md"
    linked = native.read_document("../generate-assets/SKILL.md", relative_to=workflow)
    assert linked == native.read_skill("generate-assets")
    reference = native.read_document("../../references/generation-results.md", relative_to=workflow)
    assert reference.content == "Native result and repair contract.\n"
    for path in ("../outside", "/etc/passwd", "..\\outside", ".claude/settings.json", "CLAUDE.drama.md"):
        with pytest.raises(ValueError, match="Native profile document"):
            native.read_document(path)
    with pytest.raises(ValueError, match="origin"):
        native.read_document("anything.md", relative_to="unknown.md")


def test_definition_updates_appear_in_fresh_snapshot_without_rewriting_source(tmp_path: Path):
    root = profile_root(tmp_path)
    previous = load_hermes_native_profile("narration", profile_root=root)
    path = root / ".claude/skills/generate-assets/SKILL.md"
    path.write_text(path.read_text() + "New native quality requirement.\n")
    current_bytes = path.read_bytes()
    updated = load_hermes_native_profile("narration", profile_root=root)
    assert updated.source_sha256 != previous.source_sha256
    assert updated.read_skill("generate-assets").content.encode() == current_bytes
    assert previous.read_skill("generate-assets").content != updated.read_skill("generate-assets").content
    assert path.read_bytes() == current_bytes


def test_spill_reader_is_a_mapping_of_native_read_not_a_global_role_grant(tmp_path: Path):
    root = profile_root(tmp_path)
    role = root / ".claude/agents/review-images.md"
    role.write_text("---\nname: review-images\ntools: mcp__arcreel__inspect_images\n---\nRestricted image role.\n")
    native = load_hermes_native_profile("narration", profile_root=root)
    assert "read_file" not in native.agent_spec("review-images").allowed_tools


def test_unmapped_native_permissions_and_redirected_sources_fail_closed(tmp_path: Path):
    root = profile_root(tmp_path)
    path = root / ".claude/agents/review-images.md"
    path.write_text("---\nname: review-images\ntools: Bash\n---\nMust stay restricted.\n")
    native = load_hermes_native_profile("narration", profile_root=root)
    with pytest.raises(ValueError, match="no approved Hermes adapter"):
        native.agent_spec("review-images")
    outside = tmp_path / "outside.md"
    outside.write_text("must not be read")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        load_hermes_native_profile("narration", profile_root=root)
