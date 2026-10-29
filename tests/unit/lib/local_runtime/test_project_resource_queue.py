"""Project-only resource exclusion and the installed Hermes HTTP hook."""

from __future__ import annotations

import fcntl
import threading
from pathlib import Path

import httpx
import pytest

from lib.local_runtime.config import LocalOnlyPolicyError, get_local_runtime_config
from lib.local_runtime.resource_queue import file_lease, inference_lease, install_httpx_inference_barrier, speech_lease


@pytest.mark.parametrize("external_setting", ["ARCREEL_TTS_SCRIPT", "ARCREEL_HERMES_RESOURCE_QUEUE_HELPER"])
def test_configuration_uses_only_project_owned_workers(monkeypatch: pytest.MonkeyPatch, external_setting: str) -> None:
    for name in ("ARCREEL_HERMES_RESOURCE_QUEUE_HELPER", "ARCREEL_TTS_SCRIPT"):
        monkeypatch.delenv(name, raising=False)
    config = get_local_runtime_config()
    own = Path(__file__).resolve().parents[4] / "lib/local_runtime"
    assert config.hermes_resource_queue_helper == own / "resource_queue.py"
    assert config.tts_script == own / "qwen_tts_worker.py"
    assert "runtime/resources" in str(config.resource_root)
    monkeypatch.setenv(external_setting, "/tmp/external-workflow.py")
    with pytest.raises(LocalOnlyPolicyError, match="project-owned"):
        get_local_runtime_config()


def test_speech_wait_is_bounded_and_keeps_inference_lock_owned(tmp_path: Path) -> None:
    with inference_lease(root=tmp_path, shared=True):
        with (
            pytest.raises(TimeoutError, match="resource admission"),
            speech_lease(root=tmp_path, timeout_seconds=0.03),
        ):
            raise AssertionError("TTS cannot overlap local inference")
        with (tmp_path / "mac-inference.lock").open("r") as stream, pytest.raises(BlockingIOError):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (
        speech_lease(root=tmp_path, timeout_seconds=1),
        (tmp_path / "mac-inference.lock").open("r") as stream,
        pytest.raises(BlockingIOError),
    ):
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_resource_wait_can_cancel_before_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "held.lock"
    with (
        file_lease(path),
        pytest.raises(InterruptedError, match="cancelled before dispatch"),
        file_lease(path, stop=lambda: True),
    ):
        raise AssertionError("cancelled work cannot dispatch")


def test_waiting_speech_writer_precedes_later_inference_reader(tmp_path: Path) -> None:
    events: list[str] = []
    writer_entered = threading.Event()
    release_writer = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        try:
            with speech_lease(root=tmp_path, timeout_seconds=3):
                events.append("speech")
                writer_entered.set()
                assert release_writer.wait(2)
        except Exception as error:
            errors.append(error)

    def reader() -> None:
        try:
            with inference_lease(root=tmp_path, timeout_seconds=3):
                events.append("reader")
        except Exception as error:
            errors.append(error)

    # The reader is started only after the writer already owns its turnstile.
    with inference_lease(root=tmp_path):
        thread = threading.Thread(target=writer)
        thread.start()
        turn = tmp_path / "mac-inference-turn.lock"
        with turn.open("r") as stream:
            for _ in range(100):
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(stream, fcntl.LOCK_UN)
                except BlockingIOError:
                    break
                writer_entered.wait(0.01)
            else:
                raise AssertionError("writer did not own turnstile")
        follower = threading.Thread(target=reader)
        follower.start()
    assert writer_entered.wait(2)
    assert events == ["speech"]
    release_writer.set()
    thread.join(3)
    follower.join(3)
    assert not thread.is_alive()
    assert not follower.is_alive()
    assert errors == []
    assert events == ["speech", "reader"]


def test_streamed_hermes_response_holds_lease_until_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCREEL_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("ARCREEL_LOCAL_INFERENCE_URL", "http://127.0.0.1:8000/v1")
    original = httpx.Client.send
    monkeypatch.setattr(httpx.Client, "send", original)
    install_httpx_inference_barrier()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"chunk"))) as client:
        response = client.send(
            client.build_request("POST", "http://127.0.0.1:8000/v1/chat/completions", json={}), stream=True
        )
        with (tmp_path / "mac-inference.lock").open("r") as stream, pytest.raises(BlockingIOError):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response.close()
        with speech_lease(root=tmp_path, timeout_seconds=0.1):
            pass


def test_hermes_hook_blocks_external_inference_and_redirects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCREEL_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.setenv("ARCREEL_LOCAL_INFERENCE_URL", "http://127.0.0.1:8000/v1")
    original = httpx.Client.send
    monkeypatch.setattr(httpx.Client, "send", original)
    install_httpx_inference_barrier()
    received: list[str] = []

    def receive(request: httpx.Request) -> httpx.Response:
        received.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://example.invalid/v1/chat/completions"})

    with httpx.Client(transport=httpx.MockTransport(receive), follow_redirects=True) as client:
        with pytest.raises(ValueError, match="cannot leave"):
            client.post("https://example.invalid/v1/chat/completions", json={})
        response = client.post("http://127.0.0.1:8000/v1/chat/completions", json={})
    assert response.status_code == 307
    assert received == ["http://127.0.0.1:8000/v1/chat/completions"]
