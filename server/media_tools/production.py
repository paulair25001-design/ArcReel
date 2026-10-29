"""Host-neutral production facts, image selection and accepted-job recovery."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from lib.infra.api_errors import ApiError
from lib.script.script_editor import ScriptEditError
from server.media_tools.context import tool_error, tool_problem
from server.media_tools.image_edits import EditableResourceType
from server.services.currency.resource_versions import restore_resource_version
from server.services.project.production_status import ProductionEpisodeNotFound, read_episode_production_status
from server.services.tasks.artifact_download_retry import retry_artifact_download as retry_native_artifact_download
from server.services.tasks.image_edit_tasks import edit_version_resource_type, resolve_current_image_pointer
from server.services.tasks.worker_runtime import get_generation_worker
from server.tool_runtime import CallerContext, ProjectScope, Services, ToolOutcome, ToolRequest


class EpisodeProductionStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_id: StrictInt = Field(ge=1, description="要读取制作事实的正式集号")


class SelectImageVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    resource_type: EditableResourceType = Field(description="图像种类，与inspect_images和edit_images一致")
    resource_id: str = Field(min_length=1, description="资产名称或分镜精确 ID；衍生资产用本体名/衍生名")
    version: StrictInt = Field(ge=1, description="项目原生版本历史中的版本号")


class RetryArtifactDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, description="本项目已接受供应商任务但下载失败的原生 task_id")


async def get_episode_production_status(
    request: ToolRequest[EpisodeProductionStatusRequest], scope: ProjectScope, caller: CallerContext, services: Services
) -> ToolOutcome[dict[str, Any]]:
    try:
        return ToolOutcome(
            value=await read_episode_production_status(
                scope.project_name,
                request.value.episode_id,
                user_id=caller.user_id,
                projects=services.projects,
                queue=services.queue,
            )
        )
    except ProductionEpisodeNotFound as exc:
        return tool_problem(str(exc), code="episode_not_found", params={"episode": request.value.episode_id})
    except Exception as exc:
        return tool_error("get_episode_production_status", exc)


def _resolve_image(services: Services, project: str, resource_type: str, resource_id: str) -> str:
    data = services.projects.load_project(project)
    if resource_type != "storyboard":
        return resolve_current_image_pointer(data, resource_type, resource_id, None)[0]
    for script_file in services.projects.list_scripts(project):
        try:
            script = services.projects.load_script_readonly(project, script_file)
            return resolve_current_image_pointer(data, resource_type, resource_id, script)[0]
        except (KeyError, ScriptEditError):
            continue
    raise KeyError("The storyboard does not belong to this project")


async def select_image_version(
    request: ToolRequest[SelectImageVersionRequest], scope: ProjectScope, _caller: CallerContext, services: Services
) -> ToolOutcome[dict[str, Any]]:
    value = request.value
    try:
        resource_id = await asyncio.to_thread(
            _resolve_image, services, scope.project_name, value.resource_type, value.resource_id
        )
        selected = await restore_resource_version(
            scope.project_name,
            edit_version_resource_type(value.resource_type),
            resource_id,
            value.version,
            project_manager=services.projects,
        )
        return ToolOutcome(
            value={
                "resource_type": value.resource_type,
                "resource_id": resource_id,
                "current_version": selected["current_version"],
                "file_path": selected["file_path"],
            }
        )
    except ApiError as exc:
        return tool_problem(str(exc), code=exc.key, params=exc.params)
    except KeyError:
        return tool_problem("The image resource does not belong to this project", code="image_resource_not_found")
    except Exception as exc:
        return tool_error("select_image_version", exc)


async def retry_artifact_download(
    request: ToolRequest[RetryArtifactDownloadRequest], scope: ProjectScope, caller: CallerContext, services: Services
) -> ToolOutcome[dict[str, Any]]:
    try:
        task = await retry_native_artifact_download(
            request.value.task_id,
            queue=services.queue,
            worker=services.generation_worker or get_generation_worker(),
            project_name=scope.project_name,
            user_id=caller.user_id,
        )
        return ToolOutcome(
            value={key: task.get(key) for key in ("task_id", "status", "provider_id", "provider_job_id")}
        )
    except ValueError as exc:
        return tool_problem(str(exc), code="task_retry_download_unavailable")
    except Exception as exc:
        return tool_error("retry_artifact_download", exc)
