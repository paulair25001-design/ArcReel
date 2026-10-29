"""ArcReel-owned offline Qwen3-TTS worker; installed engines are not workflows.

The outer process owns ArcReel's resource leases. Only its sandboxed child
imports the installed MLX engine. Voice text is passed in a private JSON file,
never in argv or normal stdout. Imports and --plan do not load a model.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

NETWORK_PROFILE = "(version 1) (allow default) (deny network*)"


def _local(value: object, *, directory: bool = False) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError("required installed local model/runtime path is missing")
    return path.resolve()


def validate_request(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("speech request must be a private local file")
    request = json.loads(path.read_bytes())
    if request.get("schema") != "arcreel-offline-qwen3-tts/v1":
        raise ValueError("unexpected project speech request schema")
    if not isinstance(request.get("text"), str) or not request["text"].strip():
        raise ValueError("speech text is empty")
    for field in ("voice_description", "instructions", "language"):
        if not isinstance(request.get(field), str) or not request[field].strip():
            raise ValueError("speech description, instructions and language are required")
    model = _local(request["model"], directory=True)
    config = json.loads((model / "config.json").read_bytes())
    if config.get("model_type") != "qwen3_tts" or config.get("tts_model_type") != "voice_design":
        raise ValueError("only the installed local Qwen3-TTS VoiceDesign model is supported")
    _local(request["python"])
    site = _local(request["site_packages"], directory=True)
    _local(site / "mlx_audio/tts/generate.py")
    output = Path(request["output"])
    if not output.is_absolute() or output.suffix.lower() != ".wav":
        raise ValueError("speech output must be an absolute WAV path")
    if output.exists() or output.with_suffix(".voice-receipt.json").exists():
        raise ValueError("speech output already exists; preserve the existing take")
    resource_root = Path(request["resource_root"])
    if not resource_root.is_absolute():
        raise ValueError("ArcReel's resource directory must be absolute")
    if type(request.get("max_tokens")) is not int or request["max_tokens"] <= 0:
        raise ValueError("speech max_tokens must be positive")
    if not isinstance(request.get("timeout_seconds"), (int, float)) or request["timeout_seconds"] <= 0:
        raise ValueError("speech timeout must be positive")
    return request


def offline_environment(request: dict[str, Any]) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TMPDIR": tempfile.gettempdir(),
        "PYTHONPATH": str(request["site_packages"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def _sanitized_vendor_log(raw: str, request: dict[str, Any]) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    values = (request["text"], request["voice_description"], request["instructions"])
    for value in sorted(values, key=len, reverse=True):
        for part in (value, *value.splitlines()):
            if part:
                for variant in (part, repr(part)[1:-1], json.dumps(part)[1:-1]):
                    text = text.replace(variant, "[REDACTED]")
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|access[_-]?token|secret|password)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+",
        r"\1[REDACTED]",
        text,
    )
    return text[-4000:]


def _network_proof() -> dict[str, Any]:
    """A loopback-only negative control must be denied by the OS sandbox."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1)
            probe.connect(("127.0.0.1", 9))
    except OSError as error:
        if error.errno in {errno.EPERM, errno.EACCES}:
            return {
                "blocked_by_os": True,
                "errno": error.errno,
                "target": "loopback-discard-port",
                "external_connections_attempted": 0,
            }
    raise ValueError("network sandbox could not be independently verified")


def synthesize(request: dict[str, Any]) -> dict[str, Any]:
    """Invoke only the independent engine API inside the network sandbox."""
    if os.environ.get("ARCREEL_TTS_SANDBOX_CHILD") != "1":
        raise ValueError("speech synthesis requires the project network sandbox")
    network = _network_proof()
    logs = io.StringIO()
    try:
        with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
            engine = importlib.import_module("mlx_audio.tts.generate")

            output = Path(request["output"])
            engine.generate_audio(
                text=request["text"],
                model=request["model"],
                max_tokens=request["max_tokens"],
                voice=None,
                instruct="\n".join((request["voice_description"], request["instructions"])),
                lang_code=request["language"],
                output_path=str(output.parent),
                file_prefix=output.stem,
                audio_format="wav",
                join_audio=True,
                play=False,
                stream=False,
                verbose=False,
                ref_audio=None,
                ref_text=None,
                stt_model=None,
            )
        if not Path(request["output"]).is_file():
            raise ValueError("engine did not produce the required complete WAV")
        return network
    except Exception:
        print(_sanitized_vendor_log(logs.getvalue(), request), file=sys.stderr)
        raise


def execution_receipt(request: dict[str, Any], network_proof: dict[str, Any]) -> dict[str, Any]:
    if network_proof.get("blocked_by_os") is not True or network_proof.get("external_connections_attempted") != 0:
        raise ValueError("speech receipt requires verified offline sandbox evidence")
    output = Path(request["output"])
    output.chmod(0o600)
    with wave.open(str(output), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        channels, width = audio.getnchannels(), audio.getsampwidth()
        if audio.getcomptype() != "NONE" or frames <= 0 or rate <= 0:
            raise ValueError("engine returned empty or unsupported PCM")
        if len(audio.readframes(frames)) != frames * channels * width:
            raise ValueError("engine returned incomplete PCM")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt = {
        "status": "AUDIO_GENERATED_LISTENING_NOT_VERIFIED",
        "output": str(output),
        "sha256": digest,
        "samples_per_channel": frames,
        "sample_rate": rate,
        "duration_seconds": frames / rate,
        "channels": channels,
        "sample_width": width,
        "model": request["model"],
        "variant": "voice_design",
        "python": request["python"],
        "site_packages": request["site_packages"],
        "text": request["text"],
        "instructions": "\n".join((request["voice_description"], request["instructions"])),
        "network": "denied_by_macos_sandbox",
        "network_sandbox_proof": network_proof,
        "model_downloaded": False,
        "paid_voice_called": False,
        "naturalness": "NOT_LISTENED",
        "worker": "arcreel-offline-qwen3-tts/v1",
        "resource_protection": {
            "root": request["resource_root"],
            "exclusive_local_inference": True,
            "serialized_tts": True,
        },
    }
    target = output.with_suffix(".voice-receipt.json")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--synthesize", action="store_true")
    args = parser.parse_args()
    try:
        request = validate_request(args.request)
        if args.plan:
            print(
                json.dumps(
                    {
                        "status": "PLAN_ONLY_NO_SYNTHESIS",
                        "worker": "arcreel-offline-qwen3-tts/v1",
                        "offline": True,
                        "paid_voice": False,
                        "voice_text_in_argv": False,
                    }
                )
            )
            return 0
        if args.synthesize:
            print(json.dumps(synthesize(request)))
            return 0
        if sys.platform != "darwin":
            raise ValueError("the installed local MLX speech engine requires macOS")
        sandbox = _local("/usr/bin/sandbox-exec")
        # This stdlib-only worker runs by file in the offline engine's Python;
        # its sibling resource_queue is project code, not an installed package.
        speech_lease = importlib.import_module("resource_queue").speech_lease

        env = offline_environment(request)
        env["ARCREEL_TTS_SANDBOX_CHILD"] = "1"
        command = [
            str(sandbox),
            "-p",
            NETWORK_PROFILE,
            request["python"],
            "-sB",
            str(Path(__file__).resolve()),
            "--request",
            str(args.request),
            "--synthesize",
        ]
        with speech_lease(root=Path(request["resource_root"]), timeout_seconds=request["timeout_seconds"]) as lease_fds:
            # The compute child retains the same open lock descriptions if
            # the outer worker is terminated before it can reap that child.
            result = subprocess.run(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                timeout=request["timeout_seconds"],
                pass_fds=lease_fds,
                capture_output=True,
            )
            if result.returncode:
                sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
                print("LOCAL_TTS: offline engine failed; no fallback was attempted", file=sys.stderr)
                return result.returncode
            receipt = execution_receipt(request, json.loads(result.stdout))
        print(json.dumps({key: receipt[key] for key in ("status", "output", "sha256", "duration_seconds", "worker")}))
        return 0
    except Exception as error:
        # Input/voice values and vendor exception strings are never exposed.
        if type(error).__name__ == "ResourceAdmissionTimeout":
            code = "resource_busy"
        elif isinstance(error, subprocess.TimeoutExpired):
            code = "engine_timeout"
        elif isinstance(error, TimeoutError):
            code = "local_timeout"
        elif isinstance(error, ValueError):
            code = "local_input_or_complete_pcm_guard"
        else:
            code = "local_runtime_failure"
        print("LOCAL_TTS: " + code + "; " + type(error).__name__ + "; no fallback was attempted", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
