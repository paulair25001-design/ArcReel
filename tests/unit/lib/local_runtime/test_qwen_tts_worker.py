"""Offline project worker tests substitute the engine/process boundary only."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import wave
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lib.local_runtime import qwen_tts_worker as worker


@pytest.fixture(autouse=True)
def deny_network_in_fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: Any) -> None:
        raise PermissionError(errno.EPERM, "unit substitute for macOS sandbox denial")

    monkeypatch.setattr(socket.socket, "connect", denied)


@pytest.fixture
def speech_request(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_tts","tts_model_type":"voice_design"}')
    site = tmp_path / "site"
    engine = site / "mlx_audio/tts/generate.py"
    engine.parent.mkdir(parents=True)
    engine.write_text("# Never imported; engine is replaced only in unit calls.\n")
    request = {
        "schema": "arcreel-offline-qwen3-tts/v1",
        "model": str(model),
        "python": sys.executable,
        "site_packages": str(site),
        "text": "私有原词\n完整保留",
        "voice_description": "成熟而温暖的私有声线",
        "instructions": "克制地解释，不念稿",
        "language": "Chinese",
        "max_tokens": 1200,
        "timeout_seconds": 1.0,
        "output": str(tmp_path / "take.wav"),
        "resource_root": str(tmp_path / "resources"),
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request, ensure_ascii=False))
    path.chmod(0o600)
    return path, request


def _wav(path: Path) -> bytes:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\x01\x00" * 240)
    return path.read_bytes()


def test_plan_does_not_import_the_model_or_expose_voice_input(
    speech_request: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, request = speech_request
    monkeypatch.setattr(sys, "argv", [worker.__file__, "--request", str(path), "--plan"])
    assert worker.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "PLAN_ONLY_NO_SYNTHESIS"
    assert output.err == ""
    assert request["text"] not in output.out
    assert request["voice_description"] not in output.out
    assert not Path(request["output"]).exists()
    assert not Path(request["resource_root"]).exists()


def test_native_voice_design_api_keeps_pcm_and_suppresses_private_engine_stdout(
    speech_request: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, request = speech_request
    calls = []
    engine = ModuleType("mlx_audio.tts.generate")

    def generate_audio(**kwargs: Any) -> None:
        calls.append(kwargs)
        print(kwargs["text"], kwargs["instruct"])
        _wav(Path(request["output"]))

    engine.generate_audio = generate_audio
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", engine)
    monkeypatch.setenv("ARCREEL_TTS_SANDBOX_CHILD", "1")
    network = worker.synthesize(request)
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert len(calls) == 1
    assert calls[0]["text"] == request["text"]
    assert calls[0]["instruct"] == request["voice_description"] + "\n" + request["instructions"]
    assert calls[0]["voice"] is None
    assert calls[0]["ref_audio"] is None
    assert calls[0]["stt_model"] is None
    raw = Path(request["output"]).read_bytes()
    receipt = worker.execution_receipt(request, network)
    sidecar = Path(request["output"]).with_suffix(".voice-receipt.json")
    assert sidecar.stat().st_mode & 0o777 == 0o600
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["samples_per_channel"] == 240
    assert receipt["text"] == request["text"]
    assert receipt["paid_voice_called"] is False
    assert receipt["model_downloaded"] is False
    assert receipt["worker"] == "arcreel-offline-qwen3-tts/v1"


def test_project_worker_holds_exclusive_resource_guards_and_sandboxes_the_child(
    speech_request: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, request = speech_request
    monkeypatch.syspath_prepend(str(Path(worker.__file__).parent))
    monkeypatch.setattr(sys, "argv", [worker.__file__, "--request", str(path)])
    monkeypatch.setenv("OPENAI_API_KEY", "MUST_NOT_LEAVE_PARENT")
    calls = []

    def engine_process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        assert command[0] == "/usr/bin/sandbox-exec"
        assert "(deny network*)" in command[2]
        assert command[-1] == "--synthesize"
        assert str(Path(worker.__file__).resolve()) in command
        assert request["text"] not in command
        assert request["voice_description"] not in command
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert len(kwargs["pass_fds"]) == 2
        assert all(os.fstat(fd).st_size == 0 for fd in kwargs["pass_fds"])
        with (
            (Path(request["resource_root"]) / "mac-inference.lock").open("r") as stream,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        _wav(Path(request["output"]))
        return subprocess.CompletedProcess(
            command, 0, stdout=b'{"blocked_by_os":true,"external_connections_attempted":0}', stderr=b""
        )

    monkeypatch.setattr(subprocess, "run", engine_process)
    assert worker.main() == 0
    captured = capsys.readouterr()
    assert len(calls) == 1
    assert json.loads(captured.out)["worker"] == "arcreel-offline-qwen3-tts/v1"
    assert request["text"] not in captured.out
    assert request["instructions"] not in captured.out
    assert "MUST_NOT_LEAVE_PARENT" not in captured.out


def test_vendor_failure_output_is_redacted_without_fallback(
    speech_request: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, request = speech_request
    engine = ModuleType("mlx_audio.tts.generate")

    def generate_audio(**kwargs: Any) -> None:
        print(kwargs["text"], kwargs["instruct"], "api_key=PRIVATE_KEY Bearer PRIVATE_BEARER")
        raise RuntimeError("vendor-message:" + request["text"])

    engine.generate_audio = generate_audio
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", engine)
    monkeypatch.setenv("ARCREEL_TTS_SANDBOX_CHILD", "1")
    monkeypatch.setattr(sys, "argv", [worker.__file__, "--request", str(path), "--synthesize"])
    assert worker.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    for private in (
        *request["text"].splitlines(),
        request["voice_description"],
        request["instructions"],
        "PRIVATE_KEY",
        "PRIVATE_BEARER",
    ):
        assert private not in captured.err
    assert not Path(request["output"]).exists()


def test_worker_refuses_existing_take_and_does_not_rewrite_it(speech_request: tuple[Path, dict[str, Any]]) -> None:
    path, request = speech_request
    output = Path(request["output"])
    original = _wav(output)
    with pytest.raises(ValueError, match="already exists"):
        worker.validate_request(path)
    assert output.read_bytes() == original
