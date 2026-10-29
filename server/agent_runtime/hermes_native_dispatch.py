"""Thin ArcReel profile dispatch over Hermes' native synchronous child agents.

This module is loaded by path in the isolated ACP process. It uses only the
standard library until the native child implementation is requested.
"""

from __future__ import annotations

import copy
import importlib
import posixpath
import re
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

_SCHEMA = "arcreel-hermes-native-profile/v1"
_REPORT_SCHEMA = "arcreel-native-agent-report/v1"
_SHA = re.compile(r"[a-f0-9]{64}")
AGENT_TYPES = frozenset(
    {
        "create-episode-script",
        "generate-assets",
        "normalize-drama-script",
        "review-footage",
        "review-images",
        "split-narration-segments",
        "split-reference-video-units",
    }
)
_REVIEWS = frozenset({"review-images", "review-footage"})
_READ_TOOLS = frozenset(
    "mcp__arcreel__" + name
    for name in (
        "get_project_content",
        "list_source_files",
        "get_source_text",
        "get_episode_script",
        "get_script_plan_content",
        "list_project_files",
        "read_project_file",
    )
)


def _name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    return str(function.get("name") or "") if isinstance(function, Mapping) else ""


def _native_builder(**kwargs: Any) -> Any:
    module = importlib.import_module("tools.delegate_tool")
    factory = getattr(module, "_build_child_preserving_parent_tools", module._build_child_agent)
    if kwargs["max_iterations"] is None:
        kwargs["max_iterations"] = module._load_config().get("max_iterations", module.DEFAULT_MAX_ITERATIONS)
    return factory(**kwargs)


def _native_context(session_id: str) -> Any:
    return importlib.import_module("agent.delegation_context").delegated_child_context(session_id)


class NativeAgentDispatcher:
    """Use native child execution while retaining the ArcReel role/tool boundary.

    ``configure_child(child, parent_agent=..., agent_type=...)`` must return a
    context manager covering execution and close. Its value is the host's
    per-child vision scope; pending image state and observed native units are
    read before leaving that scope. ``audit`` persists only host metadata and
    the native final report, without interpreting its quality verdict.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        model: str,
        base_url: str,
        configure_child: Callable[..., Any],
        audit: Callable[[dict[str, Any]], None] | None = None,
        child_builder: Callable[..., Any] | None = None,
        child_context: Callable[[str], Any] | None = None,
        max_iterations: int | None = None,
    ) -> None:
        if manifest.get("schema") != _SCHEMA or not isinstance(manifest.get("project"), str):
            raise ValueError("ArcReel native profile manifest is invalid")
        self.manifest = copy.deepcopy(dict(manifest))
        documents = self.manifest.get("documents")
        agents = self.manifest.get("agents")
        if not isinstance(documents, dict) or not isinstance(agents, dict):
            raise ValueError("ArcReel native profile documents and agents are required")
        self.documents: dict[str, Any] = documents
        self.agents: dict[str, Any] = agents
        if frozenset(self.agents) != AGENT_TYPES:
            raise ValueError("ArcReel native profile must contain the seven approved agents")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.read_tool = str(self.manifest.get("read_profile_tool") or "read_agent_profile")
        self.dispatch_tool = str(self.manifest.get("dispatch_tool") or "dispatch_agent")
        self.configure_child, self.audit = configure_child, audit
        self.child_builder = child_builder or _native_builder
        self.child_context = child_context or _native_context
        if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
            raise ValueError("Native child iteration budget is invalid")
        self.max_iterations = max_iterations

    def read_agent_profile(self, path: object, relative_to: str | None = None) -> dict[str, Any]:
        """Read a known profile document from frozen manifest data, never disk."""
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or ":" in path:
            raise ValueError("Native profile document path is invalid")
        if relative_to is not None and relative_to not in self.documents:
            raise ValueError("Native profile base document is unknown")
        if path in self.documents:
            logical = path
        else:
            logical = posixpath.normpath(posixpath.join(posixpath.dirname(relative_to or ""), path))
        if logical not in self.documents:
            raise ValueError("Native profile document is outside the approved manifest")
        document = self.documents[logical]
        if not isinstance(document, dict):
            raise ValueError("Native profile document is invalid")
        return copy.deepcopy(document)

    def _route(self, agent: Any) -> None:
        if (
            getattr(agent, "model", None) != self.model
            or str(getattr(agent, "base_url", "")).rstrip("/") != self.base_url
        ):
            raise RuntimeError("Native agent left the pinned local model route")
        if getattr(agent, "_fallback_chain", None):
            raise RuntimeError("Native ArcReel agents cannot use a fallback provider")

    def _allowed(self, parent: Any, role: Mapping[str, Any]) -> frozenset[str]:
        parent_names = {_name(tool) for tool in parent.tools}
        approved = {
            name for name in parent_names if name.startswith("mcp__arcreel__") or name in {self.read_tool, "read_file"}
        }
        declared = role.get("tools")
        if declared is not None:
            if not isinstance(declared, list) or any(not isinstance(name, str) for name in declared):
                raise ValueError("Native agent tool declaration is invalid")
            if not set(declared).issubset(approved):
                raise RuntimeError("Native role requests tools outside the parent's approved ArcReel surface")
            approved.intersection_update(declared)
        role_name = role.get("name")
        if role_name in _REVIEWS:
            inspect = (
                "mcp__arcreel__inspect_images" if role_name == "review-images" else "mcp__arcreel__inspect_video_units"
            )
            read_only = _READ_TOOLS | {self.read_tool, "read_file", inspect}
            if declared is None or not approved.issubset(read_only):
                raise RuntimeError("ArcReel review agents must retain their read-only access policy")
        if not approved:
            raise RuntimeError("Native role has no approved ArcReel tools")
        return frozenset(approved)

    @staticmethod
    def _restrict(child: Any, allowed: frozenset[str]) -> None:
        child.tools = [tool for tool in child.tools if _name(tool) in allowed]
        child.valid_tool_names = {_name(tool) for tool in child.tools}
        if frozenset(child.valid_tool_names) != allowed:
            raise RuntimeError("Hermes did not retain the native role's approved tool definitions")
        child._skip_mcp_refresh = True
        child._arcreel_native_dispatch_depth = 1
        child._delegate_role = "leaf"

    @staticmethod
    def _inspections(scope: Any) -> list[dict[str, Any]]:
        result = []
        seen: set[tuple[str, int]] = set()
        for row in getattr(scope, "observed_video_units", []):
            if not isinstance(row, Mapping):
                continue
            unit, version, hashes = row.get("unit_id"), row.get("version"), row.get("wire_image_sha256s")
            if (
                not isinstance(unit, str)
                or not unit
                or type(version) is not int
                or version < 1
                or not isinstance(hashes, list)
                or not hashes
                or any(not isinstance(value, str) or _SHA.fullmatch(value) is None for value in hashes)
            ):
                continue
            if (unit, version) not in seen:
                result.append({"unit_id": unit, "version": version, "wire_image_sha256s": list(hashes)})
                seen.add((unit, version))
        return result

    def dispatch_agent(
        self,
        parent_agent: Any,
        agent_type: object,
        prompt: object,
        context: object = None,
    ) -> dict[str, Any]:
        """Wait for one native child; no background task or alternate workflow."""
        if not isinstance(agent_type, str) or agent_type not in AGENT_TYPES or agent_type not in self.agents:
            raise ValueError("Unknown ArcReel native agent")
        if not isinstance(prompt, str) or not prompt.strip() or (context is not None and not isinstance(context, str)):
            raise ValueError("Native agent prompt and context must be text")
        if getattr(parent_agent, "_arcreel_native_dispatch_depth", 0) or getattr(parent_agent, "_delegate_depth", 0):
            raise RuntimeError("Nested native agent dispatch is unavailable")
        self._route(parent_agent)
        role = self.agents[agent_type]
        if (
            not isinstance(role, Mapping)
            or role.get("name") != agent_type
            or not isinstance(role.get("prompt"), str)
            or not isinstance(role.get("source_sha256"), str)
            or _SHA.fullmatch(role["source_sha256"]) is None
        ):
            raise ValueError("Native agent profile is invalid")
        allowed = self._allowed(parent_agent, role)
        parent_session = str(getattr(parent_agent, "session_id", "") or "")
        if not parent_session:
            raise RuntimeError("Native dispatch requires its parent session identity")
        child = self.child_builder(
            task_index=0,
            goal=prompt,
            context=context,
            toolsets=None,
            model=self.model,
            max_iterations=self.max_iterations,
            task_count=1,
            parent_agent=parent_agent,
            role="leaf",
        )
        closed = False
        try:
            lock = getattr(parent_agent, "_active_children_lock", None)
            with lock if lock is not None else nullcontext():
                active = getattr(parent_agent, "_active_children", None)
                if not isinstance(active, list):
                    raise RuntimeError("Native parent does not expose child interrupt ownership")
                if child not in active:
                    active.append(child)
            self._route(child)
            if getattr(child, "provider", None) != getattr(parent_agent, "provider", None):
                raise RuntimeError("Native child changed its provider")
            child.ephemeral_system_prompt = role["prompt"]
            child.skip_context_files, child.load_soul_identity, child.skip_background_review = True, False, True
            child_scope = self.configure_child(child, parent_agent=parent_agent, agent_type=agent_type)
            with child_scope as scope:
                try:
                    self._route(child)
                    self._restrict(child, allowed)
                    child._invalidate_system_prompt()
                    session_id = str(getattr(child, "session_id", "") or "")
                    if not session_id or session_id == parent_session:
                        raise RuntimeError("Native child must have an independent session identity")
                    with self.child_context(session_id):
                        message = prompt + ("\n\n" + context if context is not None else "")
                        native = child.run_conversation(user_message=message, task_id="arcreel-agent-" + uuid4().hex)
                    if not isinstance(native, Mapping):
                        raise RuntimeError("Native child returned an invalid result")
                    report = native.get("final_response")
                    report = report if isinstance(report, str) else ""
                    delivered = not bool(getattr(scope, "pending", True))
                    inspections = self._inspections(scope)
                    observed = (
                        bool(inspections)
                        if agent_type == "review-footage"
                        else (bool(getattr(scope, "observed_images", [])) if agent_type == "review-images" else True)
                    )
                    complete = (
                        native.get("completed") is True
                        and native.get("failed") is not True
                        and not native.get("error")
                        and native.get("interrupted") is not True
                        and bool(report.strip())
                        and delivered
                        and observed
                    )
                    result = {
                        "schema": _REPORT_SCHEMA,
                        "project": self.manifest["project"],
                        "event_id": uuid4().hex,
                        "timestamp": time.time(),
                        "parent_session_id": parent_session,
                        "session_id": session_id,
                        "agent_type": agent_type,
                        "profile_sha256": role["source_sha256"],
                        "model": self.model,
                        "status": "completed" if complete else "incomplete",
                        "report": report,
                        "inspections": inspections,
                        "pixels_delivered": delivered,
                        "native_completed": native.get("completed") is True,
                        "native_failed": native.get("failed") is True or bool(native.get("error")),
                        "native_interrupted": native.get("interrupted") is True,
                    }
                finally:
                    try:
                        child.close()
                    finally:
                        closed = True
            if self.audit is not None:
                self.audit(copy.deepcopy(result))
            return result
        finally:
            # Native construction registers parent ownership before returning.
            # Release that exact child even when setup or vision admission fails.
            lock = getattr(parent_agent, "_active_children_lock", None)
            with lock if lock is not None else nullcontext():
                active = getattr(parent_agent, "_active_children", [])
                if child in active:
                    active.remove(child)
            if not closed:
                child.close()
