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
from pathlib import Path
from typing import Any


def restrict_agent_tools(agent: Any, definitions: Any) -> None:
    """Remove native Hermes tools before the first model request."""
    agent.enabled_toolsets = ["mcp-arcreel"]
    agent.disabled_toolsets = []
    agent.tools = definitions(enabled_toolsets=agent.enabled_toolsets, quiet_mode=True, skip_tool_search_assembly=True)
    agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
    if not agent.valid_tool_names or any(not name.startswith("mcp__arcreel__") for name in agent.valid_tool_names):
        raise RuntimeError("Hermes could not establish the ArcReel-only MCP tool surface")
    agent._skip_mcp_refresh = True
    agent._invalidate_system_prompt()
    logging.getLogger(__name__).info(
        "ArcReel project tool surface ready: %d MCP tools, 0 native tools", len(agent.tools)
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
    if not names or names != agent.valid_tool_names or any(not name.startswith("mcp__arcreel__") for name in names):
        raise RuntimeError("Hermes project tool surface contains an unapproved native tool")


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
    vision = bridge.VisionTransport(
        Path(os.environ["HERMES_HOME"]),
        base_url=os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"],
        model=os.environ["ARCREEL_HERMES_EXPECTED_MODEL"],
    )
    vision.install_mcp(importlib.import_module("tools.mcp_tool"))
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
            restrict_agent_tools(agent, tools.get_tool_definitions)
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
