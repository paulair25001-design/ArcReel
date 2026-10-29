"""Local adapters keep identity, structured output and offline speech evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from lib.backends.audio_backends.base import AudioCapability, AudioSynthesisRequest
from lib.backends.text_backends.base import (
    ImageInput,
    StructuredOutputExhaustedError,
    TextGenerationRequest,
    TextOutputTruncatedError,
)
from lib.local_runtime.config import LocalRuntimeConfig, get_local_runtime_config
from lib.local_runtime.hermes_process import (
    LocalProcessError,
    build_hermes_environment,
    prepare_hermes_home,
    run_local_process,
)
from lib.local_runtime.hermes_text import HermesTextBackend
from lib.local_runtime.hermes_text_worker import _run
from lib.local_runtime.local_audio import LOCAL_NARRATOR_VOICE, LocalAudioBackend


class _Process:
    def __init__(self, output: bytes = b"", *, stderr: bytes = b"", hanging: bool = False) -> None:
        self.pid = None
        self.returncode = None if hanging else 0
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(output)
        self.stderr.feed_data(stderr)
        self.signals: list[str] = []
        self.waiting = asyncio.Event()
        self._done = asyncio.Event()
        if not hanging:
            self._finish()

    def _finish(self) -> None:
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._done.set()

    def terminate(self) -> None:
        self.signals.append("terminate")
        self.returncode = -15
        self._finish()

    def kill(self) -> None:
        self.signals.append("kill")
        self.returncode = -9
        self._finish()

    async def wait(self) -> int:
        self.waiting.set()
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode


class _TextSpawner:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.requests: list[dict[str, Any]] = []
        self.environments: list[dict[str, str]] = []

    def _record(self, args: tuple[str, ...], env: dict[str, str]) -> None:
        self.requests.append(json.loads(Path(args[-1]).read_text(encoding="utf-8")))
        self.environments.append(env)

    async def __call__(self, *args: str, **kwargs: Any) -> _Process:
        await asyncio.to_thread(self._record, args, kwargs["env"])
        return _Process(json.dumps(self.result).encode())


@pytest.fixture
def local_config(tmp_path: Path) -> LocalRuntimeConfig:
    model = tmp_path / "qwen-tts"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_tts","tts_model_type":"voice_design"}')
    return replace(
        get_local_runtime_config(),
        hermes_home=tmp_path / "hermes",
        hermes_api_key="local-auth-unit-only",
        resource_root=tmp_path / "resources",
        tts_model=model,
        tts_receipts_dir=tmp_path / "tts-receipts",
    )


def _text_result(config: LocalRuntimeConfig, text: str, **changes: Any) -> dict[str, Any]:
    return {
        "provider": "omlx",
        "model": config.hermes_model,
        "base_url": config.hermes_base_url,
        "completed": True,
        "text": text,
        "input_tokens": 100,
        "output_tokens": 20,
        **changes,
    }


class _Answer(BaseModel):
    count: int


async def test_hermes_validates_structured_output_and_preserves_local_identity(
    local_config: LocalRuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leave-parent")
    spawn = _TextSpawner(_text_result(local_config, '```json\n{"count":3}\n```'))
    backend = HermesTextBackend(local_config, spawn=spawn)
    result = await backend.generate(TextGenerationRequest(prompt="数一数", response_schema=_Answer))

    assert json.loads(result.text) == {"count": 3}
    assert (result.provider, result.model, result.input_tokens) == ("local-hermes", local_config.hermes_model, 100)
    assert "JSON" in spawn.requests[0]["prompt"]
    assert spawn.requests[0]["response_schema"] == _Answer.model_json_schema()
    assert "ANTHROPIC_API_KEY" not in spawn.environments[0]
    assert spawn.environments[0]["OPENAI_API_KEY"] == local_config.hermes_api_key
    assert local_config.hermes_api_key not in json.dumps(spawn.requests)
    home = Path(spawn.environments[0]["HERMES_HOME"])
    profile = json.loads((home / "config.yaml").read_text())
    assert profile["fallback_providers"] == []
    assert profile["mcp_servers"] == {}
    assert profile["providers"]["omlx"]["resource_queue_helper"] == str(local_config.hermes_resource_queue_helper)
    assert profile["providers"]["omlx"]["api_key"] == local_config.hermes_api_key
    assert (home / "config.yaml").stat().st_mode & 0o777 == 0o600
    assert (home / ".env").stat().st_mode & 0o777 == 0o600


async def test_dict_schema_is_enforced_without_silent_retry(local_config: LocalRuntimeConfig) -> None:
    spawn = _TextSpawner(_text_result(local_config, '{"count":"wrong"}'))
    backend = HermesTextBackend(local_config, spawn=spawn)
    with pytest.raises(StructuredOutputExhaustedError):
        await backend.generate(
            TextGenerationRequest(
                prompt="数一数", response_schema={"type": "object", "properties": {"count": {"type": "integer"}}}
            )
        )
    assert len(spawn.requests) == 1


@pytest.mark.parametrize(
    "changes", [{"model": "other-model"}, {"partial": True}, {"interrupted": True}, {"failed": True}]
)
async def test_worker_identity_or_incomplete_result_is_rejected(
    local_config: LocalRuntimeConfig, changes: dict[str, Any]
) -> None:
    spawn = _TextSpawner(_text_result(local_config, "半段回复", **changes))
    with pytest.raises(LocalProcessError):
        await HermesTextBackend(local_config, spawn=spawn).generate(TextGenerationRequest(prompt="正文"))


async def test_worker_failure_does_not_expose_local_endpoint_credentials(local_config: LocalRuntimeConfig) -> None:
    async def spawn(*args: str, **kwargs: Any) -> _Process:
        process = _Process(local_config.hermes_api_key.encode())
        process.returncode = 1
        return process

    with pytest.raises(LocalProcessError) as caught:
        await HermesTextBackend(local_config, spawn=spawn).generate(TextGenerationRequest(prompt="原始输入"))
    assert local_config.hermes_api_key not in str(caught.value)
    assert "status 1" in str(caught.value)


async def test_invalid_inner_quotes_are_rejected_without_repair(local_config: LocalRuntimeConfig) -> None:
    schema = {
        "type": "object",
        "properties": {"novel_text": {"type": "string"}},
        "required": ["novel_text"],
        "additionalProperties": False,
    }
    spawn = _TextSpawner(_text_result(local_config, '{"novel_text":"他说："别走"。"}'))
    with pytest.raises(StructuredOutputExhaustedError):
        await HermesTextBackend(local_config, spawn=spawn).generate(
            TextGenerationRequest(prompt="只返回原schema", response_schema=schema)
        )
    assert len(spawn.requests) == 1
    assert spawn.requests[0]["response_schema"] == schema


async def test_structured_truncation_is_hard_error_but_free_text_is_marked(local_config: LocalRuntimeConfig) -> None:
    spawn = _TextSpawner(_text_result(local_config, '{"count":', completed=False, finish_reason="length"))
    backend = HermesTextBackend(local_config, spawn=spawn)
    with pytest.raises(TextOutputTruncatedError):
        await backend.generate(TextGenerationRequest(prompt="数一数", response_schema=_Answer))
    result = await backend.generate(TextGenerationRequest(prompt="写正文"))
    assert result.truncated is True
    assert result.text == '{"count":'


async def test_remote_images_are_refused_before_a_process_is_started(local_config: LocalRuntimeConfig) -> None:
    spawn = _TextSpawner(_text_result(local_config, "不会被读取"))
    with pytest.raises(ValueError, match="local image"):
        await HermesTextBackend(local_config, spawn=spawn).generate(
            TextGenerationRequest(prompt="看图", images=[ImageInput(url="https://example.com/image.png")])
        )
    assert not spawn.requests


async def test_each_hermes_invocation_has_fresh_isolated_home(local_config: LocalRuntimeConfig) -> None:
    spawn = _TextSpawner(_text_result(local_config, "完成"))
    backend = HermesTextBackend(local_config, spawn=spawn)
    await backend.generate(TextGenerationRequest(prompt="创作"))
    await backend.generate(TextGenerationRequest(prompt="独立审核"))
    assert spawn.environments[0]["HERMES_HOME"] != spawn.environments[1]["HERMES_HOME"]


def test_profile_does_not_overwrite_user_files(local_config: LocalRuntimeConfig) -> None:
    home = local_config.hermes_home / "foreign"
    home.mkdir(parents=True)
    original = home / "config.yaml"
    original.write_text("original")
    with pytest.raises(ValueError, match="not owned"):
        prepare_hermes_home(local_config, home)
    assert original.read_text() == "original"


def test_missing_queue_guard_fails_before_a_profile_is_created(local_config: LocalRuntimeConfig) -> None:
    home = local_config.hermes_home / "no-guard"
    config = replace(local_config, hermes_resource_queue_helper=home / "missing.py")
    with pytest.raises(ValueError, match="resource queue helper is missing"):
        prepare_hermes_home(config, home)
    assert not home.exists()


def test_agent_profile_contains_only_explicit_mcp_and_local_auxiliary(local_config: LocalRuntimeConfig) -> None:
    config = replace(local_config, hermes_mcp_token="arc-private-token")
    home = config.hermes_home / "session"
    prepare_hermes_home(config, home, agent_tools=True)
    profile = json.loads((home / "config.yaml").read_text())
    assert set(profile["mcp_servers"]) == {"arcreel"}
    assert profile["tools"]["tool_search"]["enabled"] == "off"
    assert profile["auxiliary"]["vision"]["base_url"] == config.hermes_base_url
    assert profile["auxiliary"]["compression"]["provider"] == "omlx"
    assert build_hermes_environment(config, home)["HERMES_HOME"] == str(home)


async def test_cancelling_local_process_terminates_and_reaps() -> None:
    process = _Process(hanging=True)

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        return process

    task = asyncio.create_task(
        run_local_process(["local-worker"], env={}, cwd=Path("/"), timeout_seconds=30, spawn=spawn)
    )
    await process.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.signals == ["terminate"]
    assert process.returncode == -15


async def test_cancellation_during_spawn_still_reaps_the_created_process() -> None:
    process = _Process(hanging=True)
    entered = asyncio.Event()
    finish_spawning = asyncio.Event()

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        entered.set()
        await finish_spawning.wait()
        return process

    task = asyncio.create_task(
        run_local_process(["local-worker"], env={}, cwd=Path("/"), timeout_seconds=30, spawn=spawn)
    )
    await entered.wait()
    task.cancel()
    finish_spawning.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.signals == ["terminate"]
    assert process.returncode == -15


async def test_local_process_output_limit_is_enforced() -> None:
    process = _Process(b"x" * 100)

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        return process

    with pytest.raises(LocalProcessError, match="output limit"):
        await run_local_process(
            ["local-worker"], env={}, cwd=Path("/"), timeout_seconds=30, spawn=spawn, output_limit_bytes=20
        )


async def test_local_process_timeout_reaps_the_worker() -> None:
    process = _Process(hanging=True)

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        return process

    with pytest.raises(TimeoutError):
        await run_local_process(["local-worker"], env={}, cwd=Path("/"), timeout_seconds=0, spawn=spawn)
    assert process.returncode == -15
    assert process.signals == ["terminate"]


class _AudioSpawner:
    def __init__(self, *, truncate: bool = False) -> None:
        self.truncate = truncate
        self.texts: list[str] = []
        self.environments: list[dict[str, str]] = []

    def _write_audio(self, args: tuple[str, ...], env: dict[str, str]) -> None:
        options = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
        output = Path(options["output"])
        text = options["text"]
        self.texts.append(text)
        self.environments.append(env)
        with wave.open(str(output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x00\x00" * 240)
        if self.truncate:
            output.write_bytes(output.read_bytes()[:-8])
        receipt = {
            "status": "AUDIO_GENERATED_LISTENING_NOT_VERIFIED",
            "variant": "voice_design",
            "output": str(output),
            "samples_per_channel": 240,
            "sample_rate": 24000,
            "network": "denied_by_macos_sandbox",
            "paid_voice_called": False,
            "model_downloaded": False,
            "worker": "arcreel-offline-qwen3-tts/v1",
            "network_sandbox_proof": {"blocked_by_os": True, "external_connections_attempted": 0},
            "resource_protection": {
                "root": options["resource_root"],
                "exclusive_local_inference": True,
                "serialized_tts": True,
            },
            "model": options["model"],
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "text": text,
            "instructions": "\n".join((options["voice_description"], options["instructions"])),
        }
        output.with_suffix(".voice-receipt.json").write_text(json.dumps(receipt))

    async def __call__(self, *args: str, **kwargs: Any) -> _Process:
        await asyncio.to_thread(self._write_audio, args, kwargs["env"])
        return _Process()


async def test_local_tts_passes_verbatim_text_voice_and_delivery_instructions(
    local_config: LocalRuntimeConfig, tmp_path: Path
) -> None:
    spawn = _AudioSpawner()
    backend = LocalAudioBackend(local_config, spawn=spawn)
    request = AudioSynthesisRequest(
        text="他说：‘别走。’\n我还欠你一句话。",
        output_path=tmp_path / "take.wav",
        voice=LOCAL_NARRATOR_VOICE,
        voice_description="成年中文男声，中低音",
        instructions="像挽留熟人，克制地强调‘别走’。",
    )
    result = await backend.synthesize(request)
    assert result.output_path == request.output_path
    assert result.provider == "local-tts"
    assert spawn.texts == [request.text]
    assert "OPENAI_API_KEY" not in spawn.environments[0]
    assert result.provenance is not None
    archived_receipt = Path(result.provenance["receipt_path"])
    receipt_bytes = await asyncio.to_thread(archived_receipt.read_bytes)
    receipt = json.loads(receipt_bytes)
    assert result.provenance["receipt"] == receipt
    assert result.provenance["receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert receipt["output"] == str(request.output_path)
    assert not request.output_path.with_suffix(".voice-receipt.json").exists()
    assert receipt["instructions"] == "成年中文男声，中低音\n像挽留熟人，克制地强调‘别走’。"
    assert AudioCapability.SPEECH_SPEED not in backend.capabilities
    remaining = await asyncio.to_thread(lambda: list(tmp_path.glob("*.speech.json")))
    assert not remaining


async def test_failed_tts_keeps_exit_status_when_diagnostic_storage_is_unavailable(
    local_config: LocalRuntimeConfig, tmp_path: Path
) -> None:
    blocked = tmp_path / "not-a-directory"
    await asyncio.to_thread(blocked.write_text, "existing file")

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        process = _Process(stderr=local_config.hermes_api_key.encode())
        process.returncode = 2
        return process

    with pytest.raises(LocalProcessError, match="status 2; private failure diagnostics could not be saved") as caught:
        await LocalAudioBackend(replace(local_config, tts_receipts_dir=blocked), spawn=spawn).synthesize(
            AudioSynthesisRequest(text="私有原词", output_path=tmp_path / "take.wav", voice=LOCAL_NARRATOR_VOICE)
        )
    assert local_config.hermes_api_key not in str(caught.value)
    assert await asyncio.to_thread(blocked.read_text) == "existing file"


async def test_local_tts_rejects_incomplete_pcm(local_config: LocalRuntimeConfig, tmp_path: Path) -> None:
    with pytest.raises(LocalProcessError, match="incomplete PCM"):
        await LocalAudioBackend(local_config, spawn=_AudioSpawner(truncate=True)).synthesize(
            AudioSynthesisRequest(text="原词", output_path=tmp_path / "take.wav", voice=LOCAL_NARRATOR_VOICE)
        )


async def test_local_tts_does_not_fake_a_speed_parameter(local_config: LocalRuntimeConfig, tmp_path: Path) -> None:
    spawn = _AudioSpawner()
    with pytest.raises(ValueError, match="exact speech-speed"):
        await LocalAudioBackend(local_config, spawn=spawn).synthesize(
            AudioSynthesisRequest(
                text="原词", output_path=tmp_path / "take.wav", voice=LOCAL_NARRATOR_VOICE, speed=1.25
            )
        )
    assert not spawn.texts


@pytest.mark.parametrize(
    ("guard", "reason_code"),
    [
        ("LOCAL_TTS: TTS/视频资源正在使用，保持当前任务并等待，不停止其它服务。", "resource_busy"),
        ("tts_macos.py: error: unrecognized arguments: --unexpected", "invalid_arguments"),
        ("Unknown native engine failure", "unclassified_exit"),
    ],
)
async def test_failed_tts_preserves_private_redacted_diagnostics_without_retry(
    local_config: LocalRuntimeConfig,
    tmp_path: Path,
    guard: str,
    reason_code: str,
) -> None:
    request = AudioSynthesisRequest(
        text="私有旁白原词\n第二行不要外泄",
        output_path=tmp_path / "failed-take.wav",
        voice=LOCAL_NARRATOR_VOICE,
        voice_description="私有声线描述",
        instructions="私有演法要求",
    )
    calls: list[tuple[str, ...]] = []

    async def spawn(*args: str, **kwargs: Any) -> _Process:
        calls.append(args)
        stdout = f"Instruct: {request.voice_description}\n{request.instructions}\nText: {request.text}\n"
        stderr = f"{guard}\n{json.dumps(request.text)}\nBearer PRIVATE_BEARER\nAPI_KEY=PRIVATE_OTHER_KEY\n"
        stderr += local_config.hermes_api_key
        process = _Process(stdout.encode(), stderr=stderr.encode())
        process.returncode = 2
        return process

    with pytest.raises(LocalProcessError) as caught:
        await LocalAudioBackend(local_config, spawn=spawn).synthesize(request)
    files = await asyncio.to_thread(lambda: list(local_config.tts_receipts_dir.glob("failures/*/failure.json")))
    assert len(files) == 1
    receipt = json.loads(await asyncio.to_thread(files[0].read_text))
    assert receipt["reason_code"] == reason_code
    assert receipt["returncode"] == 2
    assert receipt["fallback_attempted"] is False
    assert str(files[0]) in str(caught.value)
    assert receipt["safe_reason"] in str(caught.value)
    assert len(calls) == 1
    assert not await asyncio.to_thread(request.output_path.exists)
    for path in [files[0], Path(receipt["stdout"]["path"]), Path(receipt["stderr"]["path"])]:
        content = await asyncio.to_thread(path.read_text)
        assert all(
            value not in content
            for value in (
                request.text,
                *request.text.splitlines(),
                request.voice_description,
                request.instructions,
                local_config.hermes_api_key,
                "PRIVATE_BEARER",
                "PRIVATE_OTHER_KEY",
            )
        )
        info = await asyncio.to_thread(path.stat)
        assert info.st_mode & 0o777 == 0o600
    for value in (request.text, request.voice_description, request.instructions, local_config.hermes_api_key):
        assert value not in str(caught.value)
    remaining = await asyncio.to_thread(lambda: list(tmp_path.glob("*.speech.json")))
    assert not remaining


@pytest.mark.parametrize("schema_mode", ["explicit", "legacy", "free"])
def test_worker_uses_fresh_tool_free_hermes_and_native_images(
    local_config: LocalRuntimeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema_mode: str
) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"local-image-pixels")
    instances: list[Any] = []
    dotenv_scopes: list[dict[str, Any]] = []

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.tools = ["must-be-cleared"]
            self.valid_tool_names = {"terminal"}
            self.closed = False
            instances.append(self)

        def run_conversation(self, **kwargs: Any) -> dict[str, Any]:
            self.conversation = kwargs
            payload = {"model": self.model, "messages": [{"role": "user", "content": "私有原词"}]}
            overrides = self.kwargs["request_overrides"] or {}
            payload.update({key: value for key, value in overrides.items() if key != "extra_body"})
            payload.update(overrides.get("extra_body", {}))
            with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
                client.post(local_config.hermes_base_url + "/chat/completions", json=payload)
            return {
                "final_response": "已审核",
                "completed": True,
                "messages": [{"role": "assistant", "finish_reason": "stop"}],
                "input_tokens": 12,
            }

        def close(self) -> None:
            self.closed = True

    provider = ModuleType("hermes_cli.runtime_provider")
    provider.resolve_runtime_provider = lambda **kwargs: {
        "base_url": local_config.hermes_base_url,
        "provider": "custom",
        "api_key": local_config.hermes_api_key,
    }
    agents = ModuleType("run_agent")
    agents.AIAgent = Agent
    loader = ModuleType("hermes_cli.env_loader")

    def dotenv(**kwargs: Any) -> list[Path]:
        dotenv_scopes.append(kwargs)
        return []

    loader.load_hermes_dotenv = dotenv
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", provider)
    monkeypatch.setitem(sys.modules, "run_agent", agents)
    monkeypatch.setitem(sys.modules, "hermes_cli.env_loader", loader)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    schema = {
        "type": "object",
        "properties": {"novel_text": {"type": "string", "minLength": 1}},
        "required": ["novel_text"],
        "additionalProperties": False,
    }
    request: dict[str, Any] = {
        "source": str(local_config.hermes_source),
        "model": local_config.hermes_model,
        "base_url": local_config.hermes_base_url,
        "prompt": "只看这张原图",
        "images": [str(image)],
        "max_output_tokens": 65536,
    }
    if schema_mode == "legacy":
        request["prompt"] += "\n\nReturn only JSON matching this schema, without Markdown fences:\nnot-the-final-schema"
        request["prompt"] += "\n\nReturn only JSON matching this schema, without Markdown fences:\n" + json.dumps(
            schema
        )
    else:
        request["response_schema"] = schema if schema_mode == "explicit" else None
        if schema_mode == "free":
            request["prompt"] += "\n\nReturn only JSON matching this schema, without Markdown fences:\n" + json.dumps(
                schema
            )
    audit_path = tmp_path / "wire-audit.json"
    result = _run(request, audit_path=audit_path)
    loader.load_hermes_dotenv(project_env="/external/secret.env", load_external_secrets=True)
    assert result["text"] == "已审核"
    assert result["finish_reason"] == "stop"
    agent = instances[0]
    assert agent.closed is True
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent.kwargs["fallback_model"] is None
    assert agent.kwargs["api_key"] == local_config.hermes_api_key
    assert local_config.hermes_api_key not in json.dumps(result)
    assert agent.kwargs["skip_memory"] is True
    expected_overrides = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "arcreel_response", "strict": True, "schema": schema},
        },
        "extra_body": {"structured_outputs": {"json": schema}},
    }
    assert agent.kwargs["request_overrides"] == (None if schema_mode == "free" else expected_overrides)
    audit = json.loads(audit_path.read_text())
    assert audit["requests"][0]["accepted_for_send"] is True
    assert audit["requests"][0]["response_format_schema_matches"] is (None if schema_mode == "free" else True)
    assert agent.conversation["conversation_history"] == []
    assert agent.conversation["user_message"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert dotenv_scopes == [
        {"hermes_home": str(tmp_path / "home"), "project_env": None, "load_external_secrets": False}
    ]
