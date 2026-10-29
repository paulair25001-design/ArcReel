"""ArcReel's process-scoped local inference and speech resource locks.

This module is also loaded by the independent Hermes interpreter by file path;
it uses only the standard library until the HTTP hook is explicitly installed.
No model/service action, project orchestration, or external workflow is owned.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import os
import threading
import time
import weakref
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ResourceAdmissionTimeout(TimeoutError):
    """A protected resource was not acquired before dispatch."""


def _root(value: Path | None = None) -> Path:
    configured = str(value) if value is not None else os.environ.get("ARCREEL_RESOURCE_ROOT", "")
    path = Path(configured)
    if not configured or not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("ArcReel requires its own absolute resource directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextlib.contextmanager
def file_lease(
    path: Path, *, deadline: float | None = None, shared: bool = False, stop: Callable[[], bool] | None = None
) -> Generator[int, None, None]:
    """Wait without changing or unlinking the existing lock file."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    acquired = False
    try:
        while True:
            if stop is not None and stop():
                raise InterruptedError("ArcReel resource wait cancelled before dispatch")
            if deadline is not None and time.monotonic() >= deadline:
                raise ResourceAdmissionTimeout("ArcReel resource admission timed out before dispatch")
            try:
                fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.05)
        yield fd
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def inference_lease(
    *,
    root: Path | None = None,
    shared: bool = True,
    timeout_seconds: float = 1800,
    stop: Callable[[], bool] | None = None,
) -> Generator[int, None, None]:
    """A turnstile prevents later readers from starving a waiting TTS writer."""
    folder = _root(root)
    deadline = time.monotonic() + timeout_seconds
    with file_lease(folder / "mac-inference-turn.lock", deadline=deadline, stop=stop):
        lease = file_lease(folder / "mac-inference.lock", shared=shared, deadline=deadline, stop=stop)
        inference_fd = lease.__enter__()
    try:
        yield inference_fd
    finally:
        lease.__exit__(None, None, None)


@contextlib.contextmanager
def speech_lease(*, root: Path, timeout_seconds: float) -> Generator[tuple[int, int], None, None]:
    folder = _root(root)
    deadline = time.monotonic() + timeout_seconds
    with (
        file_lease(folder / "tts.lock", deadline=deadline) as tts_fd,
        inference_lease(
            root=folder, shared=False, timeout_seconds=max(0.0, deadline - time.monotonic())
        ) as inference_fd,
    ):
        yield tts_fd, inference_fd


def install_httpx_inference_barrier() -> None:
    """Hold a read lease through HTTP response consumption, never a tool turn."""
    httpx = importlib.import_module("httpx")

    if getattr(httpx.Client.send, "_arcreel_inference_barrier", False):
        return
    expected = urlsplit(os.environ.get("ARCREEL_LOCAL_INFERENCE_URL", ""))
    if expected.scheme not in {"http", "https"} or expected.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("ArcReel's inference barrier requires a configured loopback endpoint")
    folder = _root()
    timeout_seconds = float(os.environ.get("ARCREEL_RESOURCE_TIMEOUT_SECONDS", "1800"))
    original_send = httpx.Client.send
    inference_paths = {
        expected.path.rstrip("/") + suffix for suffix in ("/chat/completions", "/responses", "/embeddings")
    }

    def send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        url = request.url
        if request.method != "POST" or url.path not in inference_paths:
            return original_send(client, request, *args, **kwargs)
        actual_port = url.port or (443 if url.scheme == "https" else 80)
        expected_port = expected.port or (443 if expected.scheme == "https" else 80)
        if url.scheme != expected.scheme or url.host != expected.hostname or actual_port != expected_port or url.query:
            raise ValueError("ArcReel inference cannot leave its configured local endpoint")
        lease = inference_lease(root=folder, timeout_seconds=timeout_seconds)
        lease.__enter__()
        kwargs["follow_redirects"] = False
        try:
            response = original_send(client, request, *args, **kwargs)
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        if not kwargs.get("stream", False):
            lease.__exit__(None, None, None)
            return response
        release = weakref.finalize(response, lease.__exit__, None, None, None)
        close = response.close
        mutex = threading.Lock()

        def close_response() -> None:
            with mutex:
                try:
                    close()
                finally:
                    release()

        response.close = close_response
        return response

    send.__dict__["_arcreel_inference_barrier"] = True
    httpx.Client.send = send
