"""Read or stop a user-authorized native production objective."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from server.agent_runtime.autonomous_production import get_autonomous_production, public_goal
from server.agent_runtime.autonomy_store import AutonomyStore
from server.agent_toolset.declaration import Exempt, ToolDeclaration
from server.tool_runtime import CallerContext, ProjectScope, Services, ToolOutcome, ToolProblem, ToolRequest


class GoalControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$", description="本轮自动目标上下文中的 run_id")


async def _owned(run_id: str, project: str, user_id: str):
    record = await AutonomyStore().get(run_id, user_id)
    return record if record is not None and record["project_name"] == project else None


async def read_goal(
    request: ToolRequest[GoalControlRequest], scope: ProjectScope, caller: CallerContext, services: Services
):
    del services
    goal = await _owned(request.value.run_id, scope.project_name, caller.user_id)
    if goal is None:
        return ToolOutcome(problem=ToolProblem("autonomy_run_not_found", "The production objective was not found"))
    return ToolOutcome(value=public_goal(goal))


async def stop_goal(
    request: ToolRequest[GoalControlRequest], scope: ProjectScope, caller: CallerContext, services: Services
):
    del services
    goal = await _owned(request.value.run_id, scope.project_name, caller.user_id)
    if goal is None:
        return ToolOutcome(problem=ToolProblem("autonomy_run_not_found", "The production objective was not found"))
    stopped = await AutonomyStore().stop(goal["id"], caller.user_id)
    service = get_autonomous_production()
    if service is not None:
        service.notify()
    return ToolOutcome(value=public_goal(stopped))


AUTONOMY_CONTROL_TOOLS = (
    ToolDeclaration(
        name="get_autonomous_production_goal",
        description="读取本项目已授权自动制作目标的状态；单轮完成不代表目标已交付。",
        request_model=GoalControlRequest,
        migration=Exempt("只读：自动目标存于运行时数据库，项目迁移失败时仍须可查询。"),
        domain_key="autonomous_goal",
        handler=read_goal,
    ),
    ToolDeclaration(
        name="stop_autonomous_production",
        description="用户要求停止自动制作时调用。保存停止意图，已提交媒体任务继续收尾，原生服务等待队列清空并恢复模型；不删除已有内容。",
        request_model=GoalControlRequest,
        migration=Exempt("仅停止已授权自动目标；项目迁移失败时仍须能停止并归还资源。"),
        domain_key="autonomous_goal",
        handler=stop_goal,
    ),
)
