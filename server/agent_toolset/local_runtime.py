"""Project operations required by the protected local media route."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lib.local_runtime.config import local_only_enabled
from server.agent_toolset.declaration import BLOCKED, ToolDeclaration
from server.services.project.end_frame import EndFrameError, set_end_frame_from_project_image
from server.tool_runtime import CallerContext, ProjectScope, Services, ToolOutcome, ToolProblem, ToolRequest


class SetStoryboardEndFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    script_file: str = Field(min_length=1, description="镜头所属剧本的文件名，例如 episode_1.json")
    shot_id: str = Field(min_length=1, description="剧本中要绑定目标尾帧的分镜 ID，例如 E1S01")
    source_path: str = Field(min_length=1, description="项目内已有图片的相对路径；按实际镜头意图选择目标尾帧")


class FinishLocalSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


async def set_storyboard_end_frame(
    request: ToolRequest[SetStoryboardEndFrameRequest],
    scope: ProjectScope,
    caller: CallerContext,
    services: Services,
) -> ToolOutcome[dict[str, object]]:
    del caller, services
    value = request.value
    try:
        path = await set_end_frame_from_project_image(
            project_name=scope.project_name,
            script_file=value.script_file,
            shot_id=value.shot_id,
            source_path=value.source_path,
        )
    except EndFrameError as exc:
        return ToolOutcome(problem=ToolProblem(exc.key, str(exc), params=exc.params))
    return ToolOutcome(value={"shot_id": value.shot_id, "end_frame_image": path})


async def finish_local_session(
    request: ToolRequest[FinishLocalSessionRequest],
    scope: ProjectScope,
    caller: CallerContext,
    services: Services,
) -> ToolOutcome[dict[str, object]]:
    del request, caller
    if not local_only_enabled():
        return ToolOutcome(problem=ToolProblem("local_runtime_disabled", "Local media execution is not enabled"))
    stats = await services.queue.get_task_stats(scope.project_name)
    if stats.get("queued", 0) or stats.get("running", 0):
        return ToolOutcome(problem=ToolProblem("local_runtime_busy", "Wait for all project generation tasks to finish"))
    from lib.local_runtime.l20_resource_session import finalize_l20_runtime

    result = await finalize_l20_runtime(project=scope.project_name)
    return ToolOutcome(value={"project": scope.project_name, "restoration": result})


LOCAL_RUNTIME_TOOLS = (
    ToolDeclaration(
        name="set_storyboard_end_frame",
        description="按实际镜头内容选择项目内已有图片，复制为该分镜的目标尾帧快照。L20 FastH3 生成前须同时有首帧和目标尾帧，不自动复制首帧冒充目标尾帧。",
        request_model=SetStoryboardEndFrameRequest,
        migration=BLOCKED,
        domain_key="end_frame",
        handler=set_storyboard_end_frame,
    ),
    ToolDeclaration(
        name="finish_local_session",
        description="本项目图片和视频全部完成，或用户停止制作后，精确结束本项目拥有的 L20 会话并恢复原服务。有排队或运行中的生成任务时拒绝；不能在每个镜头之间调用。",
        request_model=FinishLocalSessionRequest,
        migration=BLOCKED,
        domain_key="local_session",
        handler=finish_local_session,
    ),
)
