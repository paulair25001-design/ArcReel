"""Local Agent transport contracts, using an ACP process boundary substitute."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import LocalProcessResult
from server.agent_runtime.hermes_acp_bootstrap import (
    configure_project_instructions,
    install_isolated_env_loader,
    require_local_agent_route,
    require_project_prompt,
    require_project_tool_surface,
    restrict_agent_tools,
)
from server.agent_runtime.hermes_client import (
    HermesAgentOptions,
    HermesSDKClient,
    delete_hermes_session,
    recorded_hermes_home,
)
from server.agent_runtime.sdk_frames import parse_frame


class _Writer:
    def __init__(self, process: _ACPProcess) -> None:
        self.process = process

    def write(self, data: bytes) -> None:
        self.process.request(json.loads(data))

    async def drain(self) -> None:
        pass


class _ACPProcess:
    def __init__(self, *, defer_prompt: bool = False) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdin = _Writer(self)
        self.returncode: int | None = None
        self.stopped = asyncio.Event()
        self.prompt_started = asyncio.Event()
        self.defer_prompt = defer_prompt
        self.requests: list[dict[str, Any]] = []
        self.pending_prompt: int | None = None

    def emit(self, payload: dict[str, Any]) -> None:
        self.stdout.feed_data((json.dumps(payload) + "\n").encode())

    def answer(self, request_id: int, payload: dict[str, Any]) -> None:
        self.emit({"jsonrpc": "2.0", "id": request_id, "result": payload})

    def update(self, kind: str, **payload: Any) -> None:
        self.emit(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "hermes-session", "update": {"sessionUpdate": kind, **payload}},
            }
        )

    def request(self, request: dict[str, Any]) -> None:
        self.requests.append(request)
        method = request["method"]
        if method == "initialize":
            self.answer(request["id"], {"protocolVersion": 1})
        elif method == "session/new":
            self.answer(request["id"], {"sessionId": "hermes-session"})
        elif method == "session/load":
            self.update("agent_message_chunk", content={"type": "text", "text": "old history"})
            self.answer(request["id"], {"models": {}})
        elif method == "session/prompt":
            self.prompt_started.set()
            self.pending_prompt = request["id"]
            if not self.defer_prompt:
                self.update("agent_message_chunk", content={"type": "text", "text": "本地"})
                self.update("agent_message_chunk", content={"type": "text", "text": "回复"})
                self.update(
                    "tool_call",
                    toolCallId="tool-1",
                    title="mcp__arcreel__get_workflow_plan",
                    rawInput={"project": "demo"},
                )
                self.update(
                    "tool_call_update",
                    toolCallId="tool-1",
                    status="completed",
                    rawOutput=None,
                    content=[{"type": "content", "content": {"type": "text", "text": '{"next_action": "none"}'}}],
                )
                self.answer(request["id"], {"stopReason": "end_turn"})
        elif method == "session/cancel" and self.pending_prompt is not None:
            self.answer(self.pending_prompt, {"stopReason": "cancelled"})

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.stopped.set()

    def kill(self) -> None:
        self.terminate()

    async def wait(self) -> int:
        await self.stopped.wait()
        return 0


def _options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, resume: str | None = None) -> HermesAgentOptions:
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    config = replace(
        get_local_runtime_config(),
        hermes_home=tmp_path / "hermes",
        hermes_mcp_token="arc-private-test",
        hermes_api_key="omlx-private-test",
    )
    cwd = tmp_path / "project"
    cwd.mkdir()
    return HermesAgentOptions(
        config=config, cwd=cwd, home=config.hermes_home, project_name="demo", system_prompt="local only", resume=resume
    )


async def _prompt(text: str, identity: str):
    yield {"type": "user", "uuid": identity, "message": {"role": "user", "content": text}}


async def _until_idle(client: HermesSDKClient) -> list[dict[str, Any]]:
    frames = []
    async for frame in client.receive_messages():
        frames.append(frame)
        if (
            frame.get("type") == "system"
            and frame.get("subtype") == "session_state_changed"
            and frame.get("state") == "idle"
        ):
            return frames
    raise AssertionError("Hermes session ended before idle")


async def test_local_agent_preserves_stream_tool_and_session_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    options = _options(tmp_path, monkeypatch)
    process = _ACPProcess()
    launch: dict[str, Any] = {}
    diagnostics: list[str] = []
    stderr_seen = asyncio.Event()

    def capture_stderr(line: str):
        diagnostics.append(line)
        stderr_seen.set()

    options = replace(options, stderr=capture_stderr)
    process.stderr.feed_data(b"Connection failed for arc-private-test and omlx-private-test\n")

    async def spawn(*args: Any, **kwargs: Any):
        launch.update(argv=args, **kwargs)
        return process

    async with HermesSDKClient(options, process_factory=spawn) as client:
        await client.query(_prompt("开始", "user-1"))
        frames = await _until_idle(client)
        await stderr_seen.wait()

    assert frames[0]["session_id"] == "hermes-session"
    assert all(parse_frame(frame) is not None for frame in frames)
    assistant = [frame for frame in frames if frame.get("type") == "assistant"]
    assert assistant[0]["message"]["content"] == [{"type": "text", "text": "本地回复"}]
    assert assistant[1]["message"]["content"][0]["name"] == "mcp__arcreel__get_workflow_plan"
    tool_result = next(
        frame for frame in frames if frame.get("type") == "user" and isinstance(frame["message"]["content"], list)
    )
    assert json.loads(tool_result["message"]["content"][0]["content"]) == {"next_action": "none"}
    assert next(frame for frame in frames if frame.get("type") == "result")["is_error"] is False
    assert launch["argv"][0] == options.config.hermes_python
    assert "arc-private-test" not in str(launch["argv"])
    assert "omlx-private-test" not in str(launch["argv"])
    assert diagnostics == ["Connection failed for [redacted] and [redacted]"]
    assert launch["env"]["HERMES_HOME"] == str(options.home)
    assert process.requests[-1]["params"]["prompt"] == [{"type": "text", "text": "开始"}]
    assert recorded_hermes_home(options.config, "hermes-session", options.user_id) == options.home


async def test_session_binding_survives_runtime_directory_relocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    options = _options(tmp_path, monkeypatch)
    process = _ACPProcess()

    async def spawn(*_args: Any, **_kwargs: Any):
        return process

    async with HermesSDKClient(options, process_factory=spawn):
        pass
    moved = tmp_path / "moved-runtime"
    shutil.move(str(options.config.hermes_home), moved)
    config = replace(options.config, hermes_home=moved)

    assert recorded_hermes_home(config, "hermes-session", options.user_id) == moved
    with pytest.raises(RuntimeError, match="does not belong"):
        recorded_hermes_home(config, "hermes-session", "another-user")


async def test_queued_message_withdrawal_does_not_submit_another_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    options = _options(tmp_path, monkeypatch)
    process = _ACPProcess(defer_prompt=True)

    async def spawn(*_args: Any, **_kwargs: Any):
        return process

    async with HermesSDKClient(options, process_factory=spawn) as client:
        await client.query(_prompt("first", "user-1"))
        await process.prompt_started.wait()
        await client.query(_prompt("second", "user-2"))
        withdrawn = await client._query._send_control_request(
            {"subtype": "cancel_async_message", "message_uuid": "user-2"}
        )
        await client.interrupt()
        frames = await _until_idle(client)

    assert withdrawn == {"cancelled": True}
    assert len([request for request in process.requests if request["method"] == "session/prompt"]) == 1
    result = next(frame for frame in frames if frame.get("type") == "result")
    assert result["stop_reason"] == "interrupt"
    assert any(frame.get("command_uuid") == "user-2" and frame.get("state") == "cancelled" for frame in frames)


async def test_resume_uses_same_hermes_session_without_replaying_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    options = _options(tmp_path, monkeypatch, resume="hermes-session")
    process = _ACPProcess()

    async def spawn(*_args: Any, **_kwargs: Any):
        return process

    async with HermesSDKClient(options, process_factory=spawn) as client:
        await client.query(_prompt("continue", "user-1"))
        frames = await _until_idle(client)

    assert [request["method"] for request in process.requests[:2]] == ["initialize", "session/load"]
    assert all("old history" not in json.dumps(frame) for frame in frames)
    assert next(frame for frame in frames if frame.get("subtype") == "init")["session_id"] == "hermes-session"


def test_mcp_only_agent_rejects_a_surface_with_native_tools():
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp"],
        disabled_toolsets=[],
        tools=[],
        valid_tool_names=set(),
        _invalidate_system_prompt=lambda: None,
    )

    def definitions(**_kwargs: Any):
        return [{"function": {"name": "terminal"}}, {"function": {"name": "mcp__arcreel__read_project_file"}}]

    with pytest.raises(RuntimeError, match="ArcReel-only"):
        restrict_agent_tools(agent, definitions)


def test_mcp_only_agent_exposes_only_authenticated_project_tools():
    agent = SimpleNamespace(
        enabled_toolsets=["hermes-acp"],
        disabled_toolsets=[],
        tools=[],
        valid_tool_names=set(),
        _invalidate_system_prompt=lambda: None,
    )

    def definitions(**kwargs: Any):
        if kwargs["enabled_toolsets"] != ["mcp-arcreel"]:
            return [{"function": {"name": "terminal"}}]
        return [{"function": {"name": "mcp__arcreel__read_project_file"}}]

    restrict_agent_tools(agent, definitions)

    assert agent.enabled_toolsets == ["mcp-arcreel"]
    assert agent.valid_tool_names == {"mcp__arcreel__read_project_file"}


def test_restored_session_cannot_select_a_cloud_route():
    agent = SimpleNamespace(model="cloud-model", base_url="https://api.example.com/v1")

    with pytest.raises(RuntimeError, match="outside the configured local model"):
        require_local_agent_route(agent, model="local-qwen", base_url="http://127.0.0.1:8000/v1")


def test_mcp_catalog_stays_visible_when_hermes_defers_tool_schemas():
    catalog = [{"function": {"name": f"mcp__arcreel__tool_{index}"}} for index in range(62)]
    agent = SimpleNamespace(_invalidate_system_prompt=lambda: None)

    def definitions(*, enabled_toolsets, quiet_mode, skip_tool_search_assembly=False):
        assert enabled_toolsets == ["mcp-arcreel"]
        assert quiet_mode is True
        if skip_tool_search_assembly:
            return catalog
        return [{"function": {"name": name}} for name in ("tool_search", "tool_describe", "tool_call")]

    restrict_agent_tools(agent, definitions)
    require_project_tool_surface(agent)

    assert len(agent.tools) == 62
    assert agent.valid_tool_names == {tool["function"]["name"] for tool in catalog}
    assert agent._skip_mcp_refresh is True


def test_native_tools_added_after_session_creation_are_rejected():
    agent = SimpleNamespace(
        tools=[{"function": {"name": "terminal"}}],
        valid_tool_names={"terminal"},
    )

    with pytest.raises(RuntimeError, match="unapproved native tool"):
        require_project_tool_surface(agent)


def test_acp_receives_project_instructions_without_loading_legacy_profiles():
    agent = SimpleNamespace(_invalidate_system_prompt=lambda: None)

    configure_project_instructions(agent, "Current project is demo; local inference only")

    assert agent.ephemeral_system_prompt == "Current project is demo; local inference only"
    assert agent.skip_context_files is True
    assert agent.load_soul_identity is False
    assert agent.skip_background_review is True


def test_native_slash_commands_cannot_bypass_project_tool_routing():
    with pytest.raises(ValueError, match="slash commands are unavailable"):
        require_project_prompt("  /model cloud-provider/model")
    require_project_prompt("Please discuss the /model command as ordinary text")


def test_hermes_env_loader_is_limited_to_the_isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    observed = []

    def loader(**kwargs: Any):
        observed.append(kwargs)
        return [tmp_path / ".env"]

    module = SimpleNamespace(load_hermes_dotenv=loader)
    install_isolated_env_loader(module)

    assert module.load_hermes_dotenv(project_env=Path("/foreign/.env")) == [tmp_path / ".env"]
    assert observed == [{"hermes_home": tmp_path, "project_env": None, "load_external_secrets": False}]


async def test_deletion_uses_owned_hermes_database_and_validates_worker_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    options = _options(tmp_path, monkeypatch)
    process = _ACPProcess()

    async def spawn(*_args: Any, **_kwargs: Any):
        return process

    async with HermesSDKClient(options, process_factory=spawn):
        pass
    (options.home / "state.db").write_bytes(b"owned-test-database")
    launch = {}

    async def runner(argv, **kwargs):
        launch.update(argv=argv, **kwargs)
        return LocalProcessResult(0, b'{"deleted": true}', b"")

    await delete_hermes_session(
        options.config, project_cwd=options.cwd, user_id=options.user_id, session_id="hermes-session", runner=runner
    )

    assert launch["argv"][-2:] == ["--delete-session", "hermes-session"]
    assert launch["env"]["HERMES_HOME"] == str(options.home)
    assert "arc-private-test" not in str(launch["argv"])

    async def failed_runner(*_args, **_kwargs):
        return LocalProcessResult(0, b'{"deleted": "not-confirmed"}', b"")

    with pytest.raises(RuntimeError, match="did not confirm"):
        await delete_hermes_session(
            options.config,
            project_cwd=options.cwd,
            user_id=options.user_id,
            session_id="hermes-session",
            runner=failed_runner,
        )
    assert (options.home / "state.db").read_bytes() == b"owned-test-database"
