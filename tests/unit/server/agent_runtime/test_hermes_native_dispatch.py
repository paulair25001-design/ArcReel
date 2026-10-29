"""Native role dispatch retains profiles, local routing and actual pixel facts."""

from __future__ import annotations

import copy
import importlib.util
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent_runtime.hermes_native_dispatch import NativeAgentDispatcher
from server.agent_runtime.hermes_native_profile import load_hermes_native_profile


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


@pytest.fixture
def manifest():
    root = Path(__file__).resolve().parents[4] / "agent_runtime_profile"
    return load_hermes_native_profile("narration", profile_root=root).to_manifest("demo")


class Parent:
    def __init__(self, manifest):
        names = {
            "read_agent_profile",
            "dispatch_agent",
            "mcp__arcreel__generate_storyboards",
            "mcp__arcreel__edit_images",
        }
        for role in manifest["agents"].values():
            names.update(role.get("tools") or [])
        self.tools = [_tool(name) for name in names]
        self.valid_tool_names = names
        self.model, self.provider, self.base_url = "local-model", "omlx", "http://127.0.0.1:8000/v1"
        self.session_id = "parent-sid"
        self._fallback_chain = []
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._delegate_depth = 0


class Child:
    def __init__(self, parent, events):
        self.model, self.provider, self.base_url = parent.model, parent.provider, parent.base_url
        self.session_id = "child-sid"
        self._fallback_chain = []
        self.tools = [*copy.deepcopy(parent.tools), _tool("Bash"), _tool("delegate_task"), _tool("send_message")]
        self.events, self.parent = events, parent
        self.calls = []
        self.closed = 0
        self.result = {"completed": True, "failed": False, "final_response": "原生报告：建议修复右手。\n"}
        self.error = None

    def _invalidate_system_prompt(self):
        self.events.append("invalidate")

    def run_conversation(self, **kwargs):
        self.events.append("run")
        self.calls.append(kwargs)
        assert self.parent._active_children == [self]
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        self.events.append("close")
        self.closed += 1


def _runtime(manifest, *, preattached=False):
    parent, events, audit, built = Parent(manifest), [], [], []
    child = Child(parent, events)
    pixels = SimpleNamespace(
        pending={},
        observed_images=[{"id": "E1S01"}],
        observed_video_units=[{"unit_id": "E1S01", "version": 2, "wire_image_sha256s": ["a" * 64]}],
    )

    def builder(**kwargs):
        built.append(kwargs)
        if preattached:
            parent._active_children.append(child)
        return child

    @contextmanager
    def configure(agent, *, parent_agent, agent_type):
        assert agent is child
        assert parent_agent is parent
        events.append("scope-enter")
        try:
            yield pixels
        finally:
            assert child.closed == 1
            events.append("scope-exit")

    dispatcher = NativeAgentDispatcher(
        manifest,
        model=parent.model,
        base_url=parent.base_url,
        configure_child=configure,
        child_builder=builder,
        child_context=lambda session_id: nullcontext(),
        audit=audit.append,
    )
    return dispatcher, parent, child, pixels, events, audit, built


@pytest.mark.parametrize("preattached", [False, True])
def test_native_child_waits_and_closes_in_scope_retaining_exact_role_and_report(manifest, preattached):
    dispatch, parent, child, _, events, audit, built = _runtime(manifest, preattached=preattached)
    result = dispatch.dispatch_agent(parent, "review-footage", "审阅第1镜", "只对照正式脚本")
    assert result["status"] == "completed"
    assert result["report"] == child.result["final_response"]
    assert result["parent_session_id"] == "parent-sid"
    assert result["session_id"] == "child-sid"
    assert result["profile_sha256"] == manifest["agents"]["review-footage"]["source_sha256"]
    assert result["inspections"] == [{"unit_id": "E1S01", "version": 2, "wire_image_sha256s": ["a" * 64]}]
    assert result == audit[0]
    assert child.ephemeral_system_prompt == manifest["agents"]["review-footage"]["prompt"]
    assert child.calls[0]["user_message"] == "审阅第1镜\n\n只对照正式脚本"
    assert child.valid_tool_names == set(manifest["agents"]["review-footage"]["tools"])
    assert not {"Bash", "delegate_task", "dispatch_agent", "mcp__arcreel__edit_images"} & child.valid_tool_names
    assert built[0]["model"] == parent.model
    assert built[0]["role"] == "leaf"
    assert not any(name.startswith("override_") for name in built[0])
    assert events.index("scope-enter") < events.index("run") < events.index("close") < events.index("scope-exit")
    assert child.closed == 1
    assert parent._active_children == []


def test_creation_child_inherits_only_parent_approved_surface_and_preloaded_skill(manifest):
    dispatch, parent, child, *_ = _runtime(manifest)
    result = dispatch.dispatch_agent(parent, "create-episode-script", "编写提示词")
    assert result["status"] == "completed"
    assert "generate-script" in child.ephemeral_system_prompt
    assert "dispatch_agent" not in child.valid_tool_names
    assert "delegate_task" not in child.valid_tool_names
    assert all(
        name.startswith("mcp__arcreel__") or name in {"read_agent_profile", "read_file"}
        for name in child.valid_tool_names
    )


@pytest.mark.parametrize(
    "failure", ["pending", "unseen", "empty", "failed", "partial", "interrupted", "boolean-version"]
)
def test_review_never_claims_complete_from_model_words_or_missing_visual_delivery(manifest, failure):
    dispatch, parent, child, pixels, _, audit, _ = _runtime(manifest)
    if failure == "pending":
        pixels.pending = {"not-delivered": {}}
    elif failure == "unseen":
        pixels.observed_video_units = []
    elif failure == "empty":
        child.result["final_response"] = " \n"
    elif failure == "failed":
        child.result["failed"] = True
    elif failure == "partial":
        child.result["completed"] = False
    elif failure == "interrupted":
        child.result["interrupted"] = True
    else:
        pixels.observed_video_units[0]["version"] = True
    result = dispatch.dispatch_agent(parent, "review-footage", "我已审完，请给报告")
    assert result["status"] == "incomplete"
    assert audit[0]["status"] == "incomplete"


def test_image_review_requires_actual_observed_images(manifest):
    dispatch, parent, _, pixels, *_ = _runtime(manifest)
    pixels.observed_images = []
    assert dispatch.dispatch_agent(parent, "review-images", "审图")["status"] == "incomplete"


@pytest.mark.parametrize("failure", ["model", "endpoint", "fallback", "nested", "unknown-role", "widened-review"])
def test_parent_or_role_policy_failure_refuses_before_child_construction(manifest, failure):
    if failure == "widened-review":
        manifest["agents"]["review-footage"]["tools"].append("mcp__arcreel__edit_images")
    dispatch, parent, _, _, _, _, built = _runtime(manifest)
    if failure == "model":
        parent.model = "cloud-model"
    elif failure == "endpoint":
        parent.base_url = "https://external.invalid/v1"
    elif failure == "fallback":
        parent._fallback_chain = ["external/provider"]
    elif failure == "nested":
        parent._delegate_depth = 1
    role = "arbitrary-agent" if failure == "unknown-role" else "review-footage"
    with pytest.raises((ValueError, RuntimeError), match=r"Unknown|pinned|fallback|Nested|read-only"):
        dispatch.dispatch_agent(parent, role, "review")
    assert built == []


def test_child_route_failure_closes_without_running_or_publishing(manifest):
    dispatch, parent, child, _, _, audit, _ = _runtime(manifest)
    child.model = "other-model"
    with pytest.raises(RuntimeError, match="pinned"):
        dispatch.dispatch_agent(parent, "review-footage", "review")
    assert child.calls == []
    assert audit == []
    assert child.closed == 1
    assert parent._active_children == []


def test_child_exception_keeps_parent_ownership_until_closed_and_never_publishes_success(manifest):
    dispatch, parent, child, _, events, audit, _ = _runtime(manifest)
    child.error = RuntimeError("native child failed")
    with pytest.raises(RuntimeError, match="native child failed"):
        dispatch.dispatch_agent(parent, "review-footage", "review")
    assert child.closed == 1
    assert parent._active_children == []
    assert audit == []
    assert events.index("close") < events.index("scope-exit")


def test_profile_relative_links_only_resolve_frozen_manifest_documents(manifest):
    dispatch, *_ = _runtime(manifest)
    source = ".claude/skills/generate-storyboard/SKILL.md"
    result = dispatch.read_agent_profile("../generate-assets/SKILL.md", relative_to=source)
    assert result == manifest["documents"][".claude/skills/generate-assets/SKILL.md"]
    result["content"] = "changed by caller"
    assert dispatch.read_agent_profile(".claude/skills/generate-assets/SKILL.md")["content"] != result["content"]
    for path in ("/etc/passwd", "../../.env", "..\\.env", "settings.json", "https://outside.invalid"):
        with pytest.raises(ValueError, match=r"path|manifest"):
            dispatch.read_agent_profile(path, relative_to=source)
    with pytest.raises(ValueError, match="unknown"):
        dispatch.read_agent_profile("CLAUDE.md", relative_to="unknown.md")


def test_dispatch_module_loads_by_path_without_application_imports():
    path = Path(__file__).resolve().parents[4] / "server/agent_runtime/hermes_native_dispatch.py"
    spec = importlib.util.spec_from_file_location("standalone_arcreel_dispatch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.NativeAgentDispatcher.__module__ == "standalone_arcreel_dispatch"
