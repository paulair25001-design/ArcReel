"""Resume an accepted provider job through the existing native download lane."""

from __future__ import annotations

from typing import Any

from lib.db.base import DEFAULT_USER_ID
from lib.generation.generation_queue import GenerationQueue
from server.services.tasks.worker_runtime import ArtifactDownloadWorker


async def retry_artifact_download(
    task_id: str,
    *,
    queue: GenerationQueue,
    worker: ArtifactDownloadWorker | None,
    project_name: str | None = None,
    user_id: str = DEFAULT_USER_ID,
) -> dict[str, Any]:
    """Keep ownership and admission checks ahead of the irreversible state change."""
    if worker is None:
        raise ValueError("Native generation worker is unavailable")
    if project_name is not None:
        task = await queue.get_task(task_id)
        if task is None or task.get("project_name") != project_name or task.get("user_id", DEFAULT_USER_ID) != user_id:
            raise ValueError("Task does not belong to this project and user")
    # Reading configuration can fail. Do it before changing failed -> running.
    poll_timeout_seconds = await worker.read_video_poll_timeout_seconds()
    task = await queue.retry_artifact_download(task_id)
    await worker.retry_artifact_download(task, poll_timeout_seconds=poll_timeout_seconds)
    return task
