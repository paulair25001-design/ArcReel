"""Free offline Qwen3-TTS using ArcReel's own protected macOS worker."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lib.backends.audio_backends.base import (
    AudioCapability,
    AudioSynthesisRequest,
    AudioSynthesisResult,
    VoiceOption,
)
from lib.infra.async_thread import run_noninterruptible_async, run_noninterruptible_sync
from lib.infra.json_io import atomic_write_bytes
from lib.local_runtime.config import TTS_MODEL, TTS_PROVIDER, TTS_WORKER, LocalRuntimeConfig, get_local_runtime_config
from lib.local_runtime.hermes_process import LocalProcessError, LocalProcessResult, ProcessSpawner, run_local_process
from lib.speech.narration_config import (
    DEFAULT_INSTRUCTIONS as DEFAULT_INSTRUCTIONS,
)
from lib.speech.narration_config import (
    DEFAULT_VOICE_DESCRIPTION as DEFAULT_VOICE_DESCRIPTION,
)
from lib.speech.narration_config import (
    LOCAL_NARRATOR_VOICE as LOCAL_NARRATOR_VOICE,
)


def _safe_failure_reason(stderr: bytes) -> tuple[str, str]:
    diagnostic = stderr.decode("utf-8", errors="replace")
    guards = (
        (
            "resource_busy",
            "resource_busy",
            "ArcReel local inference resources are busy; wait without stopping other services",
        ),
        (
            "TTS/视频资源正在使用",
            "resource_busy",
            "TTS/video resource guard is busy; wait without stopping other services",
        ),
        ("现有资源锁目录缺失", "resource_lock_directory_missing", "the protected resource lock directory is missing"),
        ("输出已存在", "output_exists", "the requested output already exists; existing takes were preserved"),
        (
            "需要已存在的绝对本地路径",
            "local_path_missing",
            "a required installed local model or runtime path is missing",
        ),
        ("缺少已安装的 mlx_audio", "local_runtime_missing", "the installed local runtime does not contain mlx_audio"),
        ("error: unrecognized arguments:", "invalid_arguments", "the installed wrapper rejected its command arguments"),
        ("完整 WAV", "complete_wav_missing", "the local generator did not produce the required complete WAV"),
    )
    return next(
        ((code, reason) for marker, code, reason in guards if marker in diagnostic),
        ("unclassified_exit", "the local wrapper or engine failed; inspect the private diagnostic receipt"),
    )


def _redact_failure_output(output: bytes, private_values: tuple[str, ...]) -> bytes:
    diagnostic = re.sub(r"\x1b\[[0-9;]*m", "", output.decode("utf-8", errors="replace"))
    variants: set[str] = set()
    for value in private_values:
        for part in (value, *value.splitlines()):
            if part:
                variants.update((part, repr(part)[1:-1]))
                variants.update(json.dumps(part, ensure_ascii=ascii_only)[1:-1] for ascii_only in (False, True))
    for value in sorted(variants, key=len, reverse=True):
        diagnostic = diagnostic.replace(value, "[REDACTED]")
    diagnostic = re.sub(r"(?i)(\bBearer\s+)[^\s\"']+", r"\1[REDACTED]", diagnostic)
    diagnostic = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+",
        r"\1[REDACTED]",
        diagnostic,
    )
    return diagnostic.encode("utf-8")


def _record_failed_run(
    config: LocalRuntimeConfig,
    process: LocalProcessResult,
    args: tuple[str, ...],
    request: AudioSynthesisRequest,
    description: str,
    instructions: str,
) -> tuple[Path, str]:
    """Keep bounded, redacted child diagnostics outside disposable staging."""
    reason_code, safe_reason = _safe_failure_reason(process.stderr)
    folder = config.tts_receipts_dir / "failures" / uuid4().hex
    folder.mkdir(parents=True, mode=0o700)
    private = (request.text, description, instructions, config.hermes_api_key)
    summary = list(args[:3])
    for option, value in zip(args[3::2], args[4::2], strict=True):
        summary.extend((option, value if option in {"--model", "--python", "--output"} else "[REDACTED]"))
    receipt: dict[str, Any] = {
        "schema": "arcreel-local-tts-failure/v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "returncode": process.returncode,
        "reason_code": reason_code,
        "safe_reason": safe_reason,
        "argv_summary": summary,
        "argv_sha256": hashlib.sha256(json.dumps(args, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "output_path": str(request.output_path),
        "fallback_attempted": False,
        "redaction": "input text, voice description, instructions and credential values removed",
    }
    for name, raw in (("stdout", process.stdout), ("stderr", process.stderr)):
        path = folder / f"{name}.txt"
        sanitized = _redact_failure_output(raw, private)
        atomic_write_bytes(path, sanitized)
        path.chmod(0o600)
        receipt[name] = {
            "path": str(path),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "redacted_sha256": hashlib.sha256(sanitized).hexdigest(),
            "original_bytes": len(raw),
        }
    receipt_path = folder / "failure.json"
    atomic_write_bytes(receipt_path, (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    receipt_path.chmod(0o600)
    return receipt_path, safe_reason


def _verify_wav(
    output: Path, *, text: str, instructions: str, model: Path, receipts_dir: Path, resource_root: Path
) -> dict[str, Any]:
    """Verify complete PCM and the wrapper's offline execution receipt."""
    with wave.open(str(output), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()
        frame_size = audio.getnchannels() * audio.getsampwidth()
        if frames <= 0 or audio.getframerate() <= 0 or frame_size <= 0:
            raise LocalProcessError("Local TTS returned empty audio")
        byte_count = 0
        while chunk := audio.readframes(65536):
            byte_count += len(chunk)
        if byte_count != frames * frame_size:
            raise LocalProcessError("Local TTS returned incomplete PCM")
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    original_receipt = output.with_suffix(".voice-receipt.json")
    receipt_bytes = original_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    network = receipt.get("network_sandbox_proof", {})
    protection = receipt.get("resource_protection", {})
    if (
        receipt.get("status") != "AUDIO_GENERATED_LISTENING_NOT_VERIFIED"
        or receipt.get("variant") != "voice_design"
        or receipt.get("network") != "denied_by_macos_sandbox"
        or receipt.get("paid_voice_called") is not False
        or receipt.get("worker") != "arcreel-offline-qwen3-tts/v1"
        or network.get("blocked_by_os") is not True
        or network.get("external_connections_attempted") != 0
        or protection.get("root") != str(resource_root.resolve())
        or protection.get("exclusive_local_inference") is not True
        or protection.get("serialized_tts") is not True
        or receipt.get("model_downloaded") is not False
        or receipt.get("sha256") != digest
        or receipt.get("samples_per_channel") != frames
        or receipt.get("sample_rate") != sample_rate
        or Path(str(receipt.get("output", ""))).resolve() != output.resolve()
        or receipt.get("text") != text
        or receipt.get("instructions") != instructions
        or Path(str(receipt.get("model", ""))).resolve() != model.resolve()
    ):
        raise LocalProcessError("Local TTS receipt did not match the requested offline synthesis")
    receipts_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt_path = receipts_dir / f"{uuid4().hex}.voice-receipt.json"
    atomic_write_bytes(receipt_path, receipt_bytes)
    receipt_path.chmod(0o600)
    original_receipt.unlink()  # hard-delete: temp -- exact receipt retained in the durable runtime archive
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt": receipt,
    }


class LocalAudioBackend:
    """VoiceDesign descriptions and per-take instructions remain explicit inputs."""

    def __init__(self, config: LocalRuntimeConfig | None = None, *, spawn: ProcessSpawner | None = None) -> None:
        self._config = config or get_local_runtime_config()
        self._spawn = spawn

    @property
    def name(self) -> str:
        return TTS_PROVIDER

    @property
    def model(self) -> str:
        return TTS_MODEL

    @property
    def capabilities(self) -> set[AudioCapability]:
        return {AudioCapability.TEXT_TO_SPEECH}

    def list_voices(self) -> list[VoiceOption]:
        return [VoiceOption(id=LOCAL_NARRATOR_VOICE, label="本地旁白")]

    def _prepare(self, request: AudioSynthesisRequest) -> tuple[Path, str, str]:
        if not request.text.strip():
            raise ValueError("Local TTS requires non-empty text")
        if request.speed is not None:
            raise ValueError("Local VoiceDesign does not support an exact speech-speed parameter")
        if request.voice != LOCAL_NARRATOR_VOICE and not request.voice_description:
            raise ValueError("Select a local voice or provide its voice description")
        output = request.output_path
        if not output.is_absolute() or output.suffix.lower() != ".wav":
            raise ValueError("Local TTS output must be an absolute WAV path")
        if output.exists() or output.with_suffix(".voice-receipt.json").exists():
            raise ValueError("Local TTS output already exists")
        if not self._config.tts_script.is_file():
            raise ValueError("Configured local TTS wrapper is missing")
        if self._config.tts_script.resolve() != TTS_WORKER:
            raise ValueError("Local speech requires ArcReel's project-owned worker")
        model_config = json.loads((self._config.tts_model / "config.json").read_text(encoding="utf-8"))
        if model_config.get("model_type") != "qwen3_tts" or model_config.get("tts_model_type") != "voice_design":
            raise ValueError("Local TTS requires the configured Qwen3-TTS VoiceDesign model")
        description = request.voice_description or DEFAULT_VOICE_DESCRIPTION
        instructions = request.instructions or DEFAULT_INSTRUCTIONS
        output.parent.mkdir(parents=True, exist_ok=True)
        text_path = output.with_name(f".{output.stem}.{uuid4().hex}.speech.json")
        payload = {
            "schema": "arcreel-offline-qwen3-tts/v1",
            "text": request.text,
            "voice_description": description,
            "instructions": instructions,
            "language": request.language_type,
            "model": str(self._config.tts_model.resolve()),
            "python": self._config.tts_python,
            "site_packages": str(self._config.tts_site_packages.resolve()),
            "output": str(output),
            "resource_root": str(self._config.resource_root.resolve()),
            "max_tokens": 1200,
            "timeout_seconds": self._config.tts_timeout_seconds,
        }
        atomic_write_bytes(text_path, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        text_path.chmod(0o600)
        return text_path, description, instructions

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        preparing = asyncio.create_task(asyncio.to_thread(self._prepare, request))
        try:
            text_path, description, instructions = await asyncio.shield(preparing)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                prepared = await run_noninterruptible_async(preparing)
                await run_noninterruptible_sync(
                    prepared[0].unlink, missing_ok=True
                )  # hard-delete: temp -- private TTS input copy
            raise
        args = (
            self._config.tts_python,
            "-sB",
            str(self._config.tts_script),
            "--request",
            str(text_path),
        )
        # The project worker applies a deny-network sandbox and ArcReel-owned resource locks.
        env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        try:
            process = await run_local_process(
                args,
                env=env,
                cwd=request.output_path.parent,
                timeout_seconds=self._config.tts_timeout_seconds,
                spawn=self._spawn,
            )
            if process.returncode:
                try:
                    receipt_path, safe_reason = await run_noninterruptible_sync(
                        _record_failed_run, self._config, process, args, request, description, instructions
                    )
                except OSError:
                    raise LocalProcessError(
                        f"Local TTS exited with status {process.returncode}; private failure diagnostics could not be saved; "
                        "no fallback was attempted"
                    ) from None
                raise LocalProcessError(
                    f"Local TTS exited with status {process.returncode}: {safe_reason}; "
                    f"diagnostics: {receipt_path}; no fallback was attempted"
                )
            provenance = await run_noninterruptible_sync(
                _verify_wav,
                request.output_path,
                text=request.text,
                instructions="\n".join((description, instructions)),
                model=self._config.tts_model,
                receipts_dir=self._config.tts_receipts_dir,
                resource_root=self._config.resource_root,
            )
        finally:
            await run_noninterruptible_sync(
                text_path.unlink, missing_ok=True
            )  # hard-delete: temp -- private TTS input copy
        return AudioSynthesisResult(
            provider=self.name,
            model=self.model,
            text=request.text,
            output_path=request.output_path,
            provenance=provenance,
        )
