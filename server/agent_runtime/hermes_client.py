"""Local Hermes ACP process adapted to the Agent session actor's frame contract."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from lib.db.base import DEFAULT_USER_ID
from lib.infra.async_thread import run_sync_transaction
from lib.infra.json_io import atomic_write_json
from lib.infra.logging_utils import redact_diagnostic_text
from lib.local_runtime.config import LocalRuntimeConfig
from lib.local_runtime.hermes_process import (
    LocalProcessResult,
    build_hermes_environment,
    prepare_hermes_home,
    run_local_process,
)

_MAX_FRAME_BYTES = 32 * 1024 * 1024
_END = object()


def _session_identity(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value) is None:
        raise RuntimeError("Hermes returned an invalid session identity")
    return value


def _binding_path(config: LocalRuntimeConfig, session_id: str) -> Path:
    return config.hermes_home / "bindings" / f"{_session_identity(session_id)}.json"


def recorded_hermes_home(config: LocalRuntimeConfig, session_id: str, user_id: str) -> Path | None:
    """Resolve a session's owned directory independently of the project's absolute path."""
    path = _binding_path(config, session_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "arcreel-hermes-session/v1"
        or payload.get("user_id") != user_id
    ):
        raise RuntimeError("Hermes session binding does not belong to this user")
    relative = payload.get("home")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("Hermes session binding contains an invalid home")
    home = config.hermes_home / relative
    if not home.resolve().is_relative_to(config.hermes_home.resolve()):
        raise RuntimeError("Hermes session binding is outside ArcReel's runtime directory")
    return home


def _record_session_home(config: LocalRuntimeConfig, session_id: str, home: Path, user_id: str) -> None:
    root = config.hermes_home.resolve()
    relative = home.resolve().relative_to(root)
    path = _binding_path(config, session_id)
    existing = recorded_hermes_home(config, session_id, user_id)
    if existing is not None and existing.resolve() != home.resolve():
        raise RuntimeError("Hermes returned an identity already bound to another session directory")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_json(path, {"schema": "arcreel-hermes-session/v1", "home": relative.as_posix(), "user_id": user_id})
    path.chmod(0o600)


def _has_session_database(config: LocalRuntimeConfig, home: Path) -> bool:
    target = home.resolve()
    if not target.is_relative_to(config.hermes_home.resolve()) or target == (Path.home() / ".hermes").resolve():
        raise RuntimeError("Hermes session home is outside ArcReel's runtime directory")
    if not (target / ".arcreel-local-hermes").is_file():
        if (target / "state.db").is_file():
            raise RuntimeError("Hermes history is not owned by ArcReel; the session was retained")
        return False
    return (target / "state.db").is_file()


async def delete_hermes_session(
    config: LocalRuntimeConfig,
    *,
    project_cwd: Path,
    user_id: str,
    session_id: str,
    runner: Callable[..., Awaitable[LocalProcessResult]] = run_local_process,
) -> None:
    """Delete Hermes history through its own database API, without model inference."""
    _session_identity(session_id)
    home = await asyncio.to_thread(recorded_hermes_home, config, session_id, user_id)
    if home is None:
        return
    if not await asyncio.to_thread(_has_session_database, config, home):
        return
    result = await runner(
        [
            config.hermes_python,
            str(Path(__file__).with_name("hermes_acp_bootstrap.py")),
            "--source",
            str(config.hermes_source),
            "--delete-session",
            session_id,
        ],
        env=build_hermes_environment(config, home),
        cwd=project_cwd,
        timeout_seconds=min(config.hermes_timeout_seconds, 60),
    )
    if result.returncode != 0:
        raise RuntimeError("Hermes session history could not be deleted; the ArcReel session was retained")
    try:
        receipt = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("Hermes session deletion returned an invalid receipt") from exc
    if not isinstance(receipt, dict) or not isinstance(receipt.get("deleted"), bool):
        raise RuntimeError("Hermes session deletion did not confirm the requested operation")


@dataclass(frozen=True)
class HermesAgentOptions:
    config: LocalRuntimeConfig
    cwd: Path
    home: Path
    project_name: str
    system_prompt: str
    native_profile: dict[str, Any] | None = None
    user_id: str = DEFAULT_USER_ID
    resume: str | None = None
    session_id: str | None = None
    env: Any = None
    stderr: Callable[[str], None] | None = None


def _content_blocks(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("Hermes user content must be text or content blocks")
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Hermes user content contains an invalid block")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            source = block.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                raise ValueError("Hermes images require explicit base64 content")
            if not isinstance(source.get("data"), str) or not isinstance(source.get("media_type"), str):
                raise ValueError("Hermes image content or media type is missing")
            blocks.append({"type": "image", "data": source["data"], "mimeType": source["media_type"]})
        else:
            raise ValueError("Hermes user content contains an unsupported block")
    return blocks


def _tool_result_content(update: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Keep ACP's rendered text/images when Hermes omits structured raw output."""
    raw = update.get("rawOutput")
    if raw is not None:
        return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    result: list[dict[str, Any]] = []
    for item in update.get("content") or []:
        block = item.get("content") if isinstance(item, dict) else None
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            result.append({"type": "text", "text": block["text"]})
        elif isinstance(block, dict) and block.get("type") == "image" and isinstance(block.get("data"), str):
            result.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": block["data"],
                        "media_type": block.get("mimeType", "image/png"),
                    },
                }
            )
        elif item is not None:
            result.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
    if all(block["type"] == "text" for block in result):
        return "\n".join(block["text"] for block in result)
    return result


class HermesSDKClient:
    """One isolated ACP process, with message scheduling owned by this adapter.

    ACP notifications are translated to the existing frame transport, allowing
    the actor, event log and browser to retain their session contract. All tool
    execution remains inside Hermes and ArcReel's authenticated MCP handlers.
    """

    def __init__(
        self,
        options: HermesAgentOptions,
        *,
        process_factory: Callable[..., Awaitable[Any]] = asyncio.create_subprocess_exec,
    ) -> None:
        self.options = options
        self._process_factory = process_factory
        self._process: Any = None
        self._query = self
        self._frames: asyncio.Queue[Any] = asyncio.Queue()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_request_id = 0
        self._write_lock = asyncio.Lock()
        self._messages: deque[dict[str, Any]] = deque()
        self._message_ready = asyncio.Event()
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._session_id: str | None = None
        self._loading = False
        self._closing = False
        self._active = False
        self._text = ""
        self._message_id: str | None = None
        self._tool_names: dict[str, str] = {}

    async def __aenter__(self) -> HermesSDKClient:
        config = self.options.config
        if not config.hermes_mcp_token:
            raise RuntimeError("Local Hermes requires the internal ArcReel MCP credential")
        await run_sync_transaction(
            prepare_hermes_home, config, self.options.home, cwd=self.options.cwd, agent_tools=True
        )
        if self.options.native_profile is not None:
            await run_sync_transaction(
                atomic_write_json, self.options.home / "native-profile.json", self.options.native_profile
            )
        env = build_hermes_environment(config, self.options.home, cwd=self.options.cwd)
        env["PYTHONPATH"] = str(config.hermes_source)
        env["HERMES_EPHEMERAL_SYSTEM_PROMPT"] = self.options.system_prompt
        env["ARCREEL_HERMES_EXPECTED_MODEL"] = config.hermes_model
        env["ARCREEL_HERMES_EXPECTED_BASE_URL"] = config.hermes_base_url
        if self.options.native_profile is not None:
            env["ARCREEL_HERMES_NATIVE_PROFILE"] = str(self.options.home / "native-profile.json")
        bootstrap = Path(__file__).with_name("hermes_acp_bootstrap.py")
        self._process = await self._process_factory(
            config.hermes_python,
            str(bootstrap),
            "--source",
            str(config.hermes_source),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.options.cwd),
            env=env,
            limit=_MAX_FRAME_BYTES,
        )
        self._reader = asyncio.create_task(self._read_stdout(), name="hermes-acp-output")
        self._stderr_reader = asyncio.create_task(self._read_stderr(), name="hermes-acp-stderr")
        try:
            await self._request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
                    "clientInfo": {"name": "arcreel", "version": "local-hermes"},
                },
            )
            params: dict[str, Any] = {"cwd": str(self.options.cwd), "mcpServers": []}
            if self.options.resume:
                self._session_id = _session_identity(self.options.resume)
                self._loading = True
                result = await self._request("session/load", {**params, "sessionId": self._session_id})
                self._loading = False
                if not result:
                    raise RuntimeError("Hermes could not load the recorded session")
            else:
                if self.options.session_id:
                    raise RuntimeError("Hermes does not support creating a session with a supplied SDK identity")
                result = await self._request("session/new", params)
                session_id = result.get("sessionId")
                self._session_id = _session_identity(session_id)
                await run_sync_transaction(
                    _record_session_home, config, self._session_id, self.options.home, self.options.user_id
                )
            self._emit(
                {"type": "system", "subtype": "init", "session_id": self._session_id, "model": config.hermes_model}
            )
            self._worker = asyncio.create_task(self._run_messages(), name="hermes-acp-prompts")
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_exc: Any) -> None:
        self._closing = True
        tasks = [task for task in (self._worker, self._reader, self._stderr_reader) if task is not None]
        for task in tasks:
            task.cancel()
        if self._process is not None and self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()
                await self._process.wait()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._fail_pending(RuntimeError("Hermes ACP process closed"))
        self._frames.put_nowait(_END)

    async def query(self, prompt: AsyncIterator[dict[str, Any]], session_id: str = "default") -> None:
        del session_id
        async for message in prompt:
            if not isinstance(message.get("uuid"), str):
                raise ValueError("Hermes messages require a stable command UUID")
            self._emit_lifecycle(message["uuid"], "queued")
            if message.get("priority") == "now":
                self._messages.appendleft(message)
                await self.interrupt()
            else:
                self._messages.append(message)
            self._message_ready.set()

    async def interrupt(self) -> None:
        if self._active and self._session_id:
            await self._write({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": self._session_id}})

    async def receive_messages(self) -> AsyncIterator[dict[str, Any]]:
        while (frame := await self._frames.get()) is not _END:
            if isinstance(frame, BaseException):
                raise frame
            yield frame

    async def _send_control_request(self, request: dict[str, Any]) -> dict[str, bool]:
        if request.get("subtype") != "cancel_async_message":
            raise ValueError("Hermes does not support this SDK control request")
        message_uuid = request.get("message_uuid")
        for message in self._messages:
            if message.get("uuid") == message_uuid:
                self._messages.remove(message)
                self._emit_lifecycle(str(message_uuid), "cancelled")
                return {"cancelled": True}
        return {"cancelled": False}

    async def _write(self, message: dict[str, Any]) -> None:
        async with self._write_lock:
            if self._process is None or self._process.stdin is None:
                raise RuntimeError("Hermes ACP stdin is unavailable")
            self._process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            await self._process.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_request_id += 1
        request_id = self._next_request_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, timeout=self.options.config.hermes_timeout_seconds)
        finally:
            self._pending.pop(request_id, None)

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    async def _read_stdout(self) -> None:
        try:
            while line := await self._process.stdout.readline():
                try:
                    message = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError("Hermes ACP returned an invalid protocol frame") from exc
                if not isinstance(message, dict):
                    raise RuntimeError("Hermes ACP returned a non-object protocol frame")
                if "method" in message:
                    if "id" in message:
                        await self._reject_client_request(message)
                    elif message.get("method") == "session/update" and not self._loading:
                        self._notification(message.get("params"))
                    continue
                request_id = message.get("id")
                future = self._pending.get(request_id) if isinstance(request_id, int) else None
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message["error"]
                    detail = (
                        error.get("message", "Hermes ACP request failed")
                        if isinstance(error, dict)
                        else "Hermes ACP request failed"
                    )
                    future.set_exception(RuntimeError(self._diagnostic(detail)))
                else:
                    result = message.get("result")
                    future.set_result(result if isinstance(result, dict) else {})
            if not self._closing:
                raise RuntimeError("Hermes ACP process ended before the session was closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(exc)
            self._frames.put_nowait(exc)

    async def _read_stderr(self) -> None:
        while line := await self._process.stderr.readline():
            if self.options.stderr is not None:
                self.options.stderr(self._diagnostic(line.decode(errors="replace").rstrip()))

    def _diagnostic(self, value: object) -> str:
        text = redact_diagnostic_text(value)
        for secret in (self.options.config.hermes_mcp_token, self.options.config.hermes_api_key):
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    async def _reject_client_request(self, message: dict[str, Any]) -> None:
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if message.get("method") == "session/request_permission":
            response["result"] = {"outcome": {"outcome": "cancelled"}}
        else:
            response["error"] = {"code": -32601, "message": "ArcReel permits project operations through MCP only"}
        await self._write(response)

    async def _run_messages(self) -> None:
        while True:
            await self._message_ready.wait()
            while self._messages:
                message = self._messages.popleft()
                command_uuid = message["uuid"]
                self._active = True
                self._emit_lifecycle(command_uuid, "started")
                self._emit({"type": "system", "subtype": "session_state_changed", "state": "running"})
                self._emit({**message, "session_id": self._session_id})
                started = time.monotonic()
                try:
                    content = message.get("message", {}).get("content")
                    result = await self._request(
                        "session/prompt", {"sessionId": self._session_id, "prompt": _content_blocks(content)}
                    )
                    reason = result.get("stopReason", "end_turn")
                    self._flush_text()
                    self._result(reason, started)
                except TimeoutError:
                    await self.interrupt()
                    self._flush_text()
                    self._result(
                        "error", started, error="Local Hermes timed out; inspect existing tasks before continuing"
                    )
                    while self._messages:
                        self._emit_lifecycle(self._messages.popleft()["uuid"], "discarded")
                    self._frames.put_nowait(RuntimeError("Local Hermes prompt timed out"))
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._flush_text()
                    self._result("error", started, error=self._diagnostic(exc))
                finally:
                    self._active = False
                    self._emit_lifecycle(command_uuid, "completed")
            self._message_ready.clear()
            self._emit({"type": "system", "subtype": "session_state_changed", "state": "idle"})

    def _notification(self, params: object) -> None:
        if not isinstance(params, dict) or params.get("sessionId") != self._session_id:
            return
        update = params.get("update")
        if not isinstance(update, dict):
            return
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content")
            if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
                self._text_delta(content["text"])
        elif kind == "tool_call":
            self._flush_text()
            tool_id = str(update.get("toolCallId") or uuid4().hex)
            name = str(update.get("title") or "Hermes tool")
            self._tool_names[tool_id] = name
            raw_input = update.get("rawInput")
            self._assistant(
                [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": raw_input if isinstance(raw_input, dict) else {},
                    }
                ]
            )
        elif kind == "tool_call_update" and update.get("status") in {"completed", "failed"}:
            tool_id = str(update.get("toolCallId") or "")
            if tool_id in self._tool_names:
                self._tool_names.pop(tool_id, None)
                self._emit(
                    {
                        "type": "user",
                        "uuid": uuid4().hex,
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": _tool_result_content(update),
                                    "is_error": update.get("status") == "failed",
                                }
                            ],
                        },
                    }
                )

    def _text_delta(self, text: str) -> None:
        if self._message_id is None:
            self._message_id = uuid4().hex
            self._stream({"type": "message_start", "message": {"id": self._message_id}})
            self._stream({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        self._text += text
        self._stream({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})

    def _flush_text(self) -> None:
        if self._message_id is not None:
            self._stream({"type": "content_block_stop", "index": 0})
            self._stream({"type": "message_stop"})
            self._assistant([{"type": "text", "text": self._text}], message_id=self._message_id)
        self._text = ""
        self._message_id = None

    def _assistant(self, content: list[dict[str, Any]], *, message_id: str | None = None) -> None:
        identity = message_id or uuid4().hex
        self._emit(
            {
                "type": "assistant",
                "uuid": identity,
                "session_id": self._session_id,
                "parent_tool_use_id": None,
                "message": {
                    "id": identity,
                    "role": "assistant",
                    "model": self.options.config.hermes_model,
                    "content": content,
                },
            }
        )

    def _stream(self, event: dict[str, Any]) -> None:
        self._emit(
            {
                "type": "stream_event",
                "uuid": uuid4().hex,
                "session_id": self._session_id,
                "parent_tool_use_id": None,
                "event": event,
            }
        )

    def _result(self, reason: object, started: float, *, error: str | None = None) -> None:
        cancelled = reason == "cancelled"
        is_error = reason not in {"end_turn", "cancelled"}
        self._emit(
            {
                "type": "result",
                "subtype": "error_during_execution" if is_error else "success",
                "session_id": self._session_id,
                "uuid": uuid4().hex,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "duration_api_ms": 0,
                "num_turns": 1,
                "is_error": is_error,
                "stop_reason": "interrupt" if cancelled else str(reason),
                "result": error or "",
                "usage": {},
                "total_cost_usd": 0,
            }
        )

    def _emit_lifecycle(self, command_uuid: str, state: str) -> None:
        self._emit(
            {"type": "command_lifecycle", "command_uuid": command_uuid, "state": state, "session_id": self._session_id}
        )

    def _emit(self, frame: dict[str, Any]) -> None:
        self._frames.put_nowait(frame)
