"""Hermes-owned Python entry point with an MCP-only project tool surface."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

_HOST_TOOLS = frozenset({"read_agent_profile", "dispatch_agent", "read_file"})


def restrict_agent_tools(agent: Any, definitions: Any, *, native_host: bool = False) -> None:
    """Remove native Hermes tools before the first model request."""
    agent.enabled_toolsets = ["mcp-arcreel", *(["arcreel-host"] if native_host else [])]
    agent.disabled_toolsets = []
    agent.tools = definitions(enabled_toolsets=agent.enabled_toolsets, quiet_mode=True, skip_tool_search_assembly=True)
    agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
    allowed_host = _HOST_TOOLS if native_host else frozenset()
    if not agent.valid_tool_names or any(
        not name.startswith("mcp__arcreel__") and name not in allowed_host for name in agent.valid_tool_names
    ):
        raise RuntimeError("Hermes could not establish the ArcReel-only MCP tool surface")
    agent._skip_mcp_refresh = True
    agent._invalidate_system_prompt()
    logging.getLogger(__name__).info(
        "ArcReel project tool surface ready: %d tools, native profile adapters=%s", len(agent.tools), native_host
    )


def require_local_agent_route(agent: Any, *, model: str, base_url: str) -> None:
    if getattr(agent, "model", None) != model or str(getattr(agent, "base_url", "")).rstrip("/") != base_url.rstrip(
        "/"
    ):
        raise RuntimeError("Hermes selected a route outside the configured local model")


def build_local_model_state(agent: Any, *, model: str, base_url: str, schema: Any) -> Any:
    """Expose only the pinned local route without invoking Hermes' provider inventory.

    The stock picker auto-discovers external CLI credentials before applying its
    explicit-provider filter, so even a private HERMES_HOME is insufficient.
    """
    require_local_agent_route(agent, model=model, base_url=base_url)
    choice = "omlx:" + model
    return schema.SessionModelState(
        available_models=[schema.ModelInfo(model_id=choice, name=model)], current_model_id=choice
    )


def require_project_tool_surface(agent: Any) -> None:
    names = {tool["function"]["name"] for tool in agent.tools}
    if (
        not names
        or names != agent.valid_tool_names
        or any(not name.startswith("mcp__arcreel__") and name not in _HOST_TOOLS for name in names)
    ):
        raise RuntimeError("Hermes project tool surface contains an unapproved native tool")


def install_native_host_tools(
    agent: Any,
    manifest: dict[str, Any],
    *,
    dispatcher_type: Any,
    registry: Any,
    vision: Any,
    checkpoint: Any,
    home: Path,
) -> Any:
    """Project the native Skill/Agent host operations without changing business tools."""
    audit_lock = threading.Lock()

    def audit(row: dict[str, Any]) -> None:
        with audit_lock:
            descriptor = os.open(home / "native-agent-reports.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def configure_child(child: Any, *, parent_agent: Any, agent_type: str) -> Any:
        del agent_type
        require_local_agent_route(
            child,
            model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
            base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
        )
        checkpoint.install_project_checkpoint(
            child, home, importlib.import_module("agent.memory_manager").MemoryManager
        )
        return vision.agent_scope(child.session_id, parent_session_id=parent_agent.session_id)

    dispatcher = dispatcher_type(
        manifest,
        model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
        base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
        configure_child=configure_child,
        audit=audit,
        max_iterations=getattr(agent, "max_iterations", None),
    )
    read_spec = importlib.util.spec_from_file_location(
        "arcreel_hermes_native_read", Path(__file__).with_name("hermes_native_read.py")
    )
    if read_spec is None or read_spec.loader is None:
        raise RuntimeError("Native Hermes tool-result reader is unavailable")
    read_module = importlib.util.module_from_spec(read_spec)
    read_spec.loader.exec_module(read_module)
    spill_reader = read_module.NativeSpillReader(home)

    def read_profile(arguments: dict[str, Any], **_kwargs: Any) -> str:
        if set(arguments) - {"path", "relative_to"} or not isinstance(arguments.get("path"), str):
            raise ValueError("Read a native profile document by its path")
        return json.dumps(dispatcher.read_agent_profile(**arguments), ensure_ascii=False)

    def dispatch(arguments: dict[str, Any], **_kwargs: Any) -> str:
        if set(arguments) - {"agent_type", "prompt", "context"}:
            raise ValueError("Native dispatch accepts only agent_type, prompt and context")
        return json.dumps(dispatcher.dispatch_agent(agent, **arguments), ensure_ascii=False)

    def read_cache(arguments: dict[str, Any], **kwargs: Any) -> str:
        if set(arguments) - {"path", "offset", "limit"}:
            raise ValueError("Read a saved tool result by path, offset and limit")
        return spill_reader.read_file(**arguments, task_id=kwargs.get("task_id", "default"))

    schemas = [
        (
            "read_agent_profile",
            "读取当前模式的 ArcReel 原生 skill、reference 或 agent 原文；宿主 Skill/Read 适配，只读。",
            {
                "path": {"type": "string", "description": "原生文档逻辑路径，如 .claude/skills/edit-video/SKILL.md"},
                "relative_to": {"type": "string", "description": "解析相对链接的原生文档路径，可省略"},
            },
            ["path"],
            read_profile,
        ),
        (
            "dispatch_agent",
            "按 ArcReel 原生定义派发独立本地 Hermes 子智能体；等待原生报告，执行原有 tools/skills 权限。",
            {
                "agent_type": {"type": "string", "enum": sorted(manifest["agents"])},
                "prompt": {"type": "string", "description": "原生派发任务，传项目、文件路径、ID 与要求；不传原文全文"},
                "context": {"type": "string", "description": "必要的任务上下文，可省略"},
            },
            ["agent_type", "prompt"],
            dispatch,
        ),
        (
            "read_file",
            "分页读取当前 Hermes 会话 cache/spillover 中保存的工具结果；仅此缓存可读，项目数据仍经 ArcReel MCP。",
            {
                "path": {"type": "string", "description": "工具结果提示中的当前会话缓存文件路径"},
                "offset": {"type": "integer", "minimum": 1, "description": "读取视图的起始行，默认 1"},
                "limit": {"type": "integer", "minimum": 1, "description": "读取行数，默认沿用原生读取上限"},
            },
            ["path"],
            read_cache,
        ),
    ]
    for name, description, properties, required, handler in schemas:
        registry.register(
            name=name,
            toolset="arcreel-host",
            description=description,
            schema={
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
            handler=handler,
            max_result_size_chars=float("inf") if name in {"read_agent_profile", "read_file"} else None,
            override=name == "read_file",
        )
    return dispatcher


def configure_project_instructions(agent: Any, prompt: str) -> None:
    """ACP does not consume the CLI's ephemeral prompt environment variable."""
    if not prompt.strip():
        raise RuntimeError("Hermes project instructions are missing")
    agent.ephemeral_system_prompt = prompt
    agent.skip_context_files = True
    agent.load_soul_identity = False
    agent.skip_background_review = True
    agent._invalidate_system_prompt()


def require_project_prompt(text: str) -> None:
    if text.lstrip().startswith("/"):
        raise ValueError("Native Hermes slash commands are unavailable in ArcReel; use project MCP operations")


def install_isolated_env_loader(module: Any = None) -> None:
    """Keep Hermes imports from loading or rewriting the source checkout's .env."""
    loader: Any = module if module is not None else importlib.import_module("hermes_cli.env_loader")
    original = loader.load_hermes_dotenv
    home = Path(os.environ["HERMES_HOME"])

    def load_isolated(*_args: Any, **_kwargs: Any) -> Any:
        return original(hermes_home=home, project_env=None, load_external_secrets=False)

    loader.load_hermes_dotenv = load_isolated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--delete-session")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if not (source / "acp_adapter/session.py").is_file():
        raise RuntimeError("Configured Hermes source does not contain the ACP session adapter")
    sys.path.insert(0, str(source))
    install_isolated_env_loader()

    if args.delete_session is not None:
        if re.fullmatch(r"[A-Za-z0-9_-]{1,200}", args.delete_session) is None:
            raise RuntimeError("Hermes session identity is invalid")
        home = Path(os.environ["HERMES_HOME"]).resolve(strict=True)
        if not (home / ".arcreel-local-hermes").is_file() or home == (Path.home() / ".hermes").resolve():
            raise RuntimeError("Hermes session directory is not owned by ArcReel")
        state = importlib.import_module("hermes_state")
        database = state.SessionDB(db_path=home / "state.db")
        try:
            deleted = database.delete_session(args.delete_session, sessions_dir=home / "sessions")
        finally:
            database.close()
        print(json.dumps({"deleted": bool(deleted)}))
        return

    bridge_spec = importlib.util.spec_from_file_location(
        "arcreel_hermes_vision_bridge", Path(__file__).with_name("hermes_vision_bridge.py")
    )
    if bridge_spec is None or bridge_spec.loader is None:
        raise RuntimeError("ArcReel native vision transport is unavailable")
    bridge = importlib.util.module_from_spec(bridge_spec)
    bridge_spec.loader.exec_module(bridge)
    checkpoint_spec = importlib.util.spec_from_file_location(
        "arcreel_hermes_checkpoint", Path(__file__).with_name("hermes_checkpoint.py")
    )
    if checkpoint_spec is None or checkpoint_spec.loader is None:
        raise RuntimeError("ArcReel native transcript checkpoint is unavailable")
    checkpoint = importlib.util.module_from_spec(checkpoint_spec)
    checkpoint_spec.loader.exec_module(checkpoint)
    public_spec = importlib.util.spec_from_file_location(
        "arcreel_hermes_public_agent", Path(__file__).with_name("hermes_public_agent.py")
    )
    if public_spec is None or public_spec.loader is None:
        raise RuntimeError("Hermes public agent parameter adapter is unavailable")
    public_agent = importlib.util.module_from_spec(public_spec)
    public_spec.loader.exec_module(public_agent)
    native_manifest = None
    dispatcher_type = None
    manifest_path = os.environ.get("ARCREEL_HERMES_NATIVE_PROFILE")
    if manifest_path is not None:
        path = Path(manifest_path)
        if path != Path(os.environ["HERMES_HOME"]) / "native-profile.json" or path.is_symlink():
            raise RuntimeError("Native profile snapshot must belong to the current Hermes home")
        native_manifest = json.loads(path.read_text(encoding="utf-8"))
        dispatch_spec = importlib.util.spec_from_file_location(
            "arcreel_hermes_native_dispatch", Path(__file__).with_name("hermes_native_dispatch.py")
        )
        if dispatch_spec is None or dispatch_spec.loader is None:
            raise RuntimeError("Native Hermes profile dispatch adapter is unavailable")
        dispatch_module = importlib.util.module_from_spec(dispatch_spec)
        dispatch_spec.loader.exec_module(dispatch_module)
        dispatcher_type = dispatch_module.NativeAgentDispatcher
    vision = bridge.VisionTransport(
        Path(os.environ["HERMES_HOME"]),
        base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
        model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
        writer_tools=tuple(native_manifest.get("production_writer_tools", [])) if native_manifest else (),
    )
    error_spec = importlib.util.spec_from_file_location(
        "arcreel_hermes_mcp_errors", Path(__file__).with_name("hermes_mcp_errors.py")
    )
    if error_spec is None or error_spec.loader is None:
        raise RuntimeError("Native MCP error classification adapter is unavailable")
    errors = importlib.util.module_from_spec(error_spec)
    error_spec.loader.exec_module(errors)
    mcp_module = importlib.import_module("tools.mcp_tool")
    errors.install_mcp_business_errors(mcp_module)
    vision.install_mcp(mcp_module)
    auxiliary = importlib.import_module("agent.auxiliary_client")
    vision.install_auxiliary(auxiliary)
    vision.install_compression(importlib.import_module("agent.context_compressor"), auxiliary)
    acp = importlib.import_module("acp")
    entry = importlib.import_module("acp_adapter.entry")
    server = importlib.import_module("acp_adapter.server")
    bridge.install_acp_text_projection(server, importlib.import_module("acp_adapter.events"))
    sessions = importlib.import_module("acp_adapter.session")
    schema = importlib.import_module("acp.schema")
    tools = importlib.import_module("model_tools")

    class ProjectSessionManager(sessions.SessionManager):
        def _make_agent(self, *args: Any, **kwargs: Any) -> Any:
            startup = importlib.import_module("hermes_cli.mcp_startup")
            startup.ensure_mcp_discovery_before_agent_build(
                logger=logging.getLogger(__name__), timeout=30, thread_name="arcreel-mcp-discovery"
            )
            agent = super()._make_agent(*args, **kwargs)
            public_agent.apply_public_agent_parameters(agent)
            # Capture the already-installed inference lease, rather than replacing it.
            vision.install_http(importlib.import_module("httpx"))
            require_local_agent_route(
                agent,
                model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
                base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
            )
            checkpoint.install_project_checkpoint(
                agent, Path(os.environ["HERMES_HOME"]), importlib.import_module("agent.memory_manager").MemoryManager
            )
            configure_project_instructions(
                agent, os.environ["HERMES_EPHEMERAL_SYSTEM_PROMPT"] + bridge.TRUSTED_VISION_NOTICE
            )
            if native_manifest is not None:
                install_native_host_tools(
                    agent,
                    native_manifest,
                    dispatcher_type=dispatcher_type,
                    registry=importlib.import_module("tools.registry").registry,
                    vision=vision,
                    checkpoint=checkpoint,
                    home=Path(os.environ["HERMES_HOME"]),
                )
            restrict_agent_tools(agent, tools.get_tool_definitions, native_host=native_manifest is not None)
            return agent

    class ProjectACPAgent(server.HermesACPAgent):
        def _build_model_state(self, state: Any) -> Any:
            return build_local_model_state(
                state.agent,
                model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
                base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
                schema=schema,
            )

        async def prompt(self, prompt: Any, session_id: str, **kwargs: Any) -> Any:
            require_project_prompt(server._extract_text(prompt))
            state = self.session_manager.get_session(session_id)
            if state is None:
                raise RuntimeError("Hermes project session is missing")
            require_local_agent_route(
                state.agent,
                model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
                base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
            )
            require_project_tool_surface(state.agent)
            vision.begin_turn(session_id)
            return await super().prompt(prompt, session_id, **kwargs)

    entry._setup_logging()
    entry._load_env()
    asyncio.run(acp.run_agent(ProjectACPAgent(session_manager=ProjectSessionManager()), use_unstable_protocol=True))


if __name__ == "__main__":
    main()
