"""The tool and HTTP route share one accepted-job download admission service."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from lib.db.base import DEFAULT_USER_ID
from server.media_tools.production import RetryArtifactDownloadRequest
from server.media_tools.production import retry_artifact_download as retry_tool
from server.services.tasks.artifact_download_retry import retry_artifact_download
from server.tool_runtime import CallerContext, ProjectScope, ToolRequest


class DownloadQueue:
    def __init__(self):
        self.task = {
            "task_id": "original",
            "project_name": "demo",
            "user_id": DEFAULT_USER_ID,
            "status": "failed",
            "provider_id": "l20",
            "provider_job_id": "accepted-original",
            "submitted_base_url": "http://private-runtime",
        }

    async def get_task(self, task_id):
        return copy.deepcopy(self.task) if task_id == self.task["task_id"] else None

    async def retry_artifact_download(self, task_id):
        if task_id != self.task["task_id"] or self.task["status"] != "failed":
            raise ValueError("not retryable")
        self.task["status"] = "running"
        return copy.deepcopy(self.task)


class DownloadWorker:
    def __init__(self, *, config_error=False):
        self.config_error = config_error
        self.recovered = None

    async def read_video_poll_timeout_seconds(self):
        if self.config_error:
            raise RuntimeError("config unavailable")
        return 1234

    async def retry_artifact_download(self, task, *, poll_timeout_seconds):
        self.recovered = {"task": task, "timeout": poll_timeout_seconds}


async def test_tool_resumes_the_same_task_and_job_without_exposing_endpoint(tmp_path):
    queue, worker = DownloadQueue(), DownloadWorker()
    outcome = await retry_tool(
        ToolRequest(RetryArtifactDownloadRequest(task_id="original")),
        ProjectScope("demo", tmp_path),
        CallerContext(DEFAULT_USER_ID, "mcp"),
        SimpleNamespace(queue=queue, generation_worker=worker),
    )
    assert outcome.problem is None
    assert outcome.value == {
        "task_id": "original",
        "status": "running",
        "provider_id": "l20",
        "provider_job_id": "accepted-original",
    }
    assert worker.recovered["timeout"] == 1234
    assert worker.recovered["task"]["task_id"] == "original"
    assert worker.recovered["task"]["provider_job_id"] == "accepted-original"
    assert queue.task["status"] == "running"


@pytest.mark.parametrize("owner", ["foreign-project", "foreign-user"])
async def test_foreign_scope_is_refused_without_transition(owner):
    queue, worker = DownloadQueue(), DownloadWorker()
    kwargs = {"project_name": "other"} if owner == "foreign-project" else {"project_name": "demo", "user_id": "other"}
    with pytest.raises(ValueError, match="does not belong"):
        await retry_artifact_download("original", queue=queue, worker=worker, **kwargs)
    assert queue.task["status"] == "failed"
    assert worker.recovered is None


async def test_configuration_failure_and_missing_worker_leave_task_retryable():
    queue = DownloadQueue()
    with pytest.raises(RuntimeError, match="config unavailable"):
        await retry_artifact_download(
            "original", queue=queue, worker=DownloadWorker(config_error=True), project_name="demo"
        )
    assert queue.task["status"] == "failed"
    with pytest.raises(ValueError, match="unavailable"):
        await retry_artifact_download("original", queue=queue, worker=None, project_name="demo")
    assert queue.task["status"] == "failed"
