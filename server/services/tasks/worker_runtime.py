"""The application's native worker, shared by HTTP and host-neutral tools."""

from __future__ import annotations

from typing import Any, Protocol


class ArtifactDownloadWorker(Protocol):
    async def read_video_poll_timeout_seconds(self) -> int: ...

    async def retry_artifact_download(self, task: dict[str, Any], *, poll_timeout_seconds: int) -> None: ...


_worker: ArtifactDownloadWorker | None = None


def bind_generation_worker(worker: ArtifactDownloadWorker | None) -> None:
    """Bind only the running application's worker; shutdown clears the binding."""
    global _worker
    _worker = worker


def get_generation_worker() -> ArtifactDownloadWorker | None:
    return _worker
