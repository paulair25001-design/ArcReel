"""One explicit user authorization starts native, local Hermes production."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from lib.infra.api_errors import BadRequestError, ConflictError, NotFoundError, ServiceUnavailableError
from lib.local_runtime.config import local_only_enabled
from server.agent_runtime.autonomous_production import get_autonomous_production, public_goal
from server.agent_runtime.autonomy_store import AutonomyStore
from server.auth import CurrentUser

router = APIRouter()


class StartAutonomousProduction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: int = Field(ge=1)
    objective: str = Field(min_length=1, max_length=12000)
    client_key: str = Field(min_length=1, max_length=128)

    @field_validator("objective", "client_key")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace characters")
        return value.strip()


class RetryAutonomousProduction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: StrictInt = Field(ge=0)


def _service():
    if not local_only_enabled():
        raise BadRequestError("autonomy_requires_local")
    result = get_autonomous_production()
    if result is None:
        raise ServiceUnavailableError("autonomy_unavailable")
    return result


@router.post("/projects/{project_name}/autonomous-production")
async def start_production(project_name: str, body: StartAutonomousProduction, user: CurrentUser):
    try:
        goal = await _service().create(project_name, body.episode_id, body.objective.strip(), user.id, body.client_key)
    except FileNotFoundError as error:
        raise NotFoundError("project_not_found", name=project_name) from error
    except ValueError as error:
        raise ConflictError("autonomy_goal_conflict") from error
    return {"goal": public_goal(goal)}


@router.get("/projects/{project_name}/autonomous-production/latest")
async def latest_production(project_name: str, user: CurrentUser):
    goal = await AutonomyStore().get_latest(project_name, user.id)
    return {"goal": public_goal(goal) if goal is not None else None}


async def _owned(project: str, run_id: str, user_id: str):
    try:
        goal = await AutonomyStore().get(run_id, user_id)
    except ValueError as error:
        raise NotFoundError("autonomy_run_not_found") from error
    if goal is None or goal["project_name"] != project:
        raise NotFoundError("autonomy_run_not_found")
    return goal


@router.get("/projects/{project_name}/autonomous-production/{run_id}")
async def get_production(project_name: str, run_id: str, user: CurrentUser):
    return {"goal": public_goal(await _owned(project_name, run_id, user.id))}


@router.post("/projects/{project_name}/autonomous-production/{run_id}/stop")
async def stop_production(project_name: str, run_id: str, user: CurrentUser):
    await _owned(project_name, run_id, user.id)
    goal = await _service().stop(run_id, user.id)
    return {"goal": public_goal(goal)}


@router.post("/projects/{project_name}/autonomous-production/{run_id}/retry")
async def retry_production(project_name: str, run_id: str, body: RetryAutonomousProduction, user: CurrentUser):
    await _owned(project_name, run_id, user.id)
    try:
        goal = await _service().retry(run_id, user.id, body.expected_revision)
    except ValueError as error:
        raise ConflictError("autonomy_goal_conflict") from error
    return {"goal": public_goal(goal)}
