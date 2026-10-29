"""Isolated local Hermes configuration and bounded subprocess execution."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from lib.infra.json_io import atomic_write_json
from lib.local_runtime.config import RESOURCE_QUEUE_HELPER, LocalRuntimeConfig

LOCAL_HERMES_PROVIDER = "omlx"
MAX_PROCESS_OUTPUT_BYTES = 8 * 1024 * 1024
_HOME_MARKER = ".arcreel-local-hermes"
_ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR")
_PUBLIC_MODEL_FIELDS = ("context_length", "max_tokens", "temperature", "top_p", "top_k", "min_p", "reasoning_effort")
_PUBLIC_AGENT_FIELDS = ("max_turns", "reasoning_effort", "local_stream_stale_timeout")
_PUBLIC_COMPRESSION_FIELDS = (
    "enabled",
    "checkpoint_required",
    "progress_notices",
    "threshold",
    "threshold_tokens",
    "target_ratio",
    "tail_mode",
    "protect_first_n",
    "protect_last_n",
    "min_tail_user_messages",
    "max_attempts",
    "context_timeout_seconds",
    "context_total_ceiling_seconds",
    "in_place",
    "idle_compact_after_seconds",
    "proactive_prune_tokens",
    "proactive_prune_min_result_chars",
    "proactive_prune_min_reclaim_tokens",
)
_PUBLIC_BUDGET_MAPS = ("model_headroom_tokens", "model_threshold_tokens", "model_thresholds")


def _public_extra_body(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "thinking_budget",
        )
        if type(value.get(key)) in (int, float)
    }
    result.update({key: value[key] for key in ("enable_thinking", "preserve_thinking") if type(value.get(key)) is bool})
    template = value.get("chat_template_kwargs")
    if isinstance(template, dict):
        selected = {
            key: template[key] for key in ("enable_thinking", "preserve_thinking") if type(template.get(key)) is bool
        }
        effort = template.get("reasoning_effort")
        if isinstance(effort, str) and effort in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            selected["reasoning_effort"] = effort
        if type(template.get("thinking_budget")) is int:
            selected["thinking_budget"] = template["thinking_budget"]
        if selected:
            result["chat_template_kwargs"] = selected
    return result


def read_public_hermes_parameters(path: Path | None = None) -> dict[str, Any]:
    """Read only shared inference knobs, excluding identities, routes and credentials."""
    source = path if path is not None else Path.home() / ".hermes/config.yaml"
    if not source.is_file():
        return {}
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Public Hermes configuration must be a mapping")
    result: dict[str, Any] = {}
    for section, fields in (
        ("model", _PUBLIC_MODEL_FIELDS),
        ("agent", _PUBLIC_AGENT_FIELDS),
        ("compression", _PUBLIC_COMPRESSION_FIELDS),
    ):
        values = loaded.get(section)
        if isinstance(values, dict):
            result[section] = {key: values[key] for key in fields if type(values.get(key)) in (str, int, float, bool)}
    compression = loaded.get("compression")
    if isinstance(compression, dict):
        for key in _PUBLIC_BUDGET_MAPS:
            value = compression.get(key)
            if isinstance(value, dict):
                result.setdefault("compression", {})[key] = {
                    name: budget
                    for name, budget in value.items()
                    if isinstance(name, str) and type(budget) in (int, float)
                }
        micro = compression.get("micro_compact")
        if isinstance(micro, dict):
            result.setdefault("compression", {})["micro_compact"] = {
                key: value
                for key, value in micro.items()
                if key in {"enabled", "threshold_tokens", "target_tokens", "protect_last_n", "max_tokens"}
                and type(value) in (int, float, bool)
            }
    auxiliary = loaded.get("auxiliary")
    summary = auxiliary.get("compression") if isinstance(auxiliary, dict) else None
    if isinstance(summary, dict):
        selected = {
            key: summary[key]
            for key in ("reasoning_effort", "max_output_tokens", "context_length", "timeout")
            if type(summary.get(key)) in (str, int, float, bool)
        }
        extra = _public_extra_body(summary.get("extra_body"))
        if extra:
            selected["extra_body"] = extra
        result["auxiliary_compression"] = selected
    providers = loaded.get("providers")
    local_provider = providers.get(LOCAL_HERMES_PROVIDER) if isinstance(providers, dict) else None
    if isinstance(local_provider, dict):
        provider: dict[str, Any] = {}
        extra = _public_extra_body(local_provider.get("extra_body"))
        if extra:
            provider["extra_body"] = extra
        capabilities = local_provider.get("capabilities")
        if isinstance(capabilities, dict) and type(capabilities.get("reasoning")) is bool:
            provider["capabilities"] = {"reasoning": capabilities["reasoning"]}
        result["provider"] = provider
    return copy.deepcopy(result)


class LocalProcessError(RuntimeError):
    """A local process failed without falling back to another engine."""


class LocalProcess(Protocol):
    @property
    def pid(self) -> int | None: ...

    @property
    def returncode(self) -> int | None: ...

    @property
    def stdout(self) -> asyncio.StreamReader | None: ...

    @property
    def stderr(self) -> asyncio.StreamReader | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


ProcessSpawner = Callable[..., Awaitable[LocalProcess]]


@dataclass(frozen=True, slots=True)
class LocalProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def build_hermes_environment(
    config: LocalRuntimeConfig,
    home: Path,
    cwd: Path | None = None,
    mcp: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Preserve OS essentials and pass only the explicitly selected local route."""
    del cwd, mcp
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env.update(
        HERMES_HOME=str(home),
        HERMES_INFERENCE_PROVIDER=LOCAL_HERMES_PROVIDER,
        HERMES_INFERENCE_MODEL=config.hermes_model,
        OPENAI_API_KEY=config.hermes_api_key,
        PYTHONPATH=str(config.hermes_source),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        ARCREEL_RESOURCE_ROOT=str(config.resource_root.resolve()),
        ARCREEL_LOCAL_INFERENCE_URL=config.hermes_base_url,
        ARCREEL_RESOURCE_TIMEOUT_SECONDS=str(config.hermes_timeout_seconds),
    )
    return env


def hermes_environment(config: LocalRuntimeConfig, home: Path) -> dict[str, str]:
    """Environment used by the persistent Hermes ACP client."""
    return build_hermes_environment(config, home)


def hermes_argv(config: LocalRuntimeConfig, *args: str) -> tuple[str, ...]:
    return (config.hermes_executable, *args)


def prepare_hermes_home(
    config: LocalRuntimeConfig,
    home: Path,
    *,
    agent_tools: bool = False,
    cwd: Path | None = None,
    mcp: Mapping[str, Any] | None = None,
    max_output_tokens: int | None = None,
    system_prompt: str | None = None,
    public_config_path: Path | None = None,
) -> None:
    """Write a private ArcReel-owned profile without copying user credentials."""
    del cwd
    if not config.hermes_resource_queue_helper.is_file():
        raise ValueError("Configured local Hermes resource queue helper is missing")
    if config.hermes_resource_queue_helper.resolve() != RESOURCE_QUEUE_HELPER:
        raise ValueError("Local Hermes requires ArcReel's project-owned resource helper")
    root = config.hermes_home.resolve()
    target = home.resolve()
    if not home.is_absolute() or not target.is_relative_to(root):
        raise ValueError("Hermes home must be inside the configured ArcReel runtime directory")
    if target == (Path.home() / ".hermes").resolve():
        raise ValueError("ArcReel cannot overwrite the user's Hermes home")
    marker = target / _HOME_MARKER
    if target.exists() and not marker.exists() and any(target.iterdir()):
        raise ValueError("Hermes runtime directory is not owned by ArcReel")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker.write_text("arcreel-local-hermes/v1\n", encoding="utf-8")
    public = read_public_hermes_parameters(public_config_path)
    local_route = {"provider": LOCAL_HERMES_PROVIDER, "model": config.hermes_model, "base_url": config.hermes_base_url}
    servers: dict[str, Any] = {}
    if agent_tools:
        if mcp is not None:
            servers = dict(mcp)
        elif config.hermes_mcp_token:
            servers = {
                "arcreel": {
                    "url": config.hermes_mcp_url,
                    "headers": {"Authorization": f"Bearer {config.hermes_mcp_token}"},
                    "timeout": config.hermes_timeout_seconds,
                }
            }
        else:
            raise ValueError("Hermes Agent requires an explicit ArcReel MCP token")
    agent: dict[str, Any] = {**public.get("agent", {}), "verbose": False, "image_input_mode": "native"}
    if not agent_tools:
        agent["max_turns"] = 2
    if system_prompt:
        agent["system_prompt"] = system_prompt
    payload = {
        "model": {
            **public.get("model", {}),
            "default": config.hermes_model,
            "provider": LOCAL_HERMES_PROVIDER,
            "base_url": config.hermes_base_url,
            "supports_vision": True,
            **({"max_tokens": max_output_tokens} if max_output_tokens is not None else {}),
        },
        "providers": {
            LOCAL_HERMES_PROVIDER: {
                **public.get("provider", {}),
                "name": "oMLX",
                "base_url": config.hermes_base_url,
                "api_key": config.hermes_api_key,
                "resource_queue_helper": str(config.hermes_resource_queue_helper),
                "api_mode": "chat_completions",
                "default_model": config.hermes_model,
                "capabilities": {"vision": True, "tools": True, **public.get("provider", {}).get("capabilities", {})},
            }
        },
        "agent": agent,
        "auxiliary": {
            "vision": dict(local_route),
            "compression": {**public.get("auxiliary_compression", {}), **local_route},
            "title_generation": {"enabled": False},
        },
        "fallback_providers": [],
        "fallback_model": [],
        # The ACP adapter exposes the full authenticated project catalog and
        # deliberately excludes Hermes' native deferred-tool bridge.
        "tools": {"tool_search": {"enabled": "off"}},
        "platform_toolsets": {"cli": [], "acp": ["arcreel"] if agent_tools else []},
        "mcp_servers": servers,
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "stt": {"enabled": False},
        "tts": {"provider": "local_handoff"},
        "compression": {
            **public.get("compression", {}),
            **({"enabled": False} if not agent_tools else {}),
        },
    }
    atomic_write_json(target / "config.yaml", payload)
    (target / "config.yaml").chmod(0o600)
    env_file = target / ".env"
    env_file.write_text("OPENAI_API_KEY=" + json.dumps(config.hermes_api_key) + "\n", encoding="utf-8")
    env_file.chmod(0o600)


async def _read_bounded(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    size = 0
    while chunk := await stream.read(65536):
        size += len(chunk)
        if size > limit:
            raise LocalProcessError("Local process exceeded its output limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _signal_process(proc: LocalProcess, *, force: bool) -> None:
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix" and proc.pid is not None:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            proc.kill()
        else:
            proc.terminate()


async def _reap(proc: LocalProcess, grace: float) -> None:
    if proc.returncode is not None and proc.pid is None:
        return
    _signal_process(proc, force=False)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except TimeoutError:
        _signal_process(proc, force=True)
        await proc.wait()
    finally:
        # The wrapper may exit before a synthesis child; the process group remains ours.
        if os.name == "posix" and proc.pid is not None:
            _signal_process(proc, force=True)


async def _reap_to_completion(proc: LocalProcess, grace: float) -> None:
    task = asyncio.create_task(_reap(proc, grace))
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def run_local_process(
    args: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
    spawn: ProcessSpawner | None = None,
    output_limit_bytes: int = MAX_PROCESS_OUTPUT_BYTES,
    terminate_grace_seconds: float = 2.0,
) -> LocalProcessResult:
    """Execute without a shell; timeout/cancellation reaps the owned process group."""
    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": dict(env),
        "cwd": cwd,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    spawning = asyncio.ensure_future((spawn or asyncio.create_subprocess_exec)(*args, **kwargs))
    try:
        proc = await asyncio.shield(spawning)
    except asyncio.CancelledError:
        # Process creation can finish after the caller cancels; retain ownership
        # until it either fails or gives us a process that can be reaped.
        while not spawning.done():
            try:
                await asyncio.shield(spawning)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not spawning.cancelled() and spawning.exception() is None:
            await _reap_to_completion(spawning.result(), terminate_grace_seconds)
        raise
    readers = [
        asyncio.create_task(_read_bounded(proc.stdout, output_limit_bytes)),
        asyncio.create_task(_read_bounded(proc.stderr, output_limit_bytes)),
    ]
    waiter = asyncio.create_task(proc.wait())
    try:
        await asyncio.wait_for(
            asyncio.gather(readers[0], readers[1], waiter),
            timeout=timeout_seconds,
        )
    except BaseException:
        await _reap_to_completion(proc, terminate_grace_seconds)
        for task in [*readers, waiter]:
            task.cancel()
        await asyncio.gather(readers[0], readers[1], waiter, return_exceptions=True)
        raise
    return LocalProcessResult(returncode=waiter.result(), stdout=readers[0].result(), stderr=readers[1].result())


def decode_process_json(result: LocalProcessResult) -> dict[str, Any]:
    """Read a worker envelope without exposing child output in error messages."""
    if result.returncode:
        raise LocalProcessError(f"Local worker exited with status {result.returncode}; no fallback was attempted")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeError, ValueError) as exc:
        raise LocalProcessError("Local worker did not return a JSON result") from exc
    if not isinstance(payload, dict):
        raise LocalProcessError("Local worker returned an invalid result envelope")
    return payload
