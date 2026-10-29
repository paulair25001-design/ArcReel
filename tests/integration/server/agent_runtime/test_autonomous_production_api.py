"""Authorization, scope and privacy at the native production-goal HTTP seam."""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from server.agent_runtime.autonomy_store import AutonomyStore
from server.auth import CurrentUserInfo, get_current_user
from server.error_handlers import register_error_handlers
from server.routers import autonomous_production as api


@pytest.fixture
async def client(db_factory, monkeypatch):
    store = AutonomyStore(session_factory=db_factory)
    monkeypatch.setattr(api, "AutonomyStore", lambda: store)
    app = FastAPI()
    app.include_router(api.router, prefix="/api/v1")
    register_error_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: CurrentUserInfo(id="owner", sub="owner")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http, store


async def test_reads_hide_other_users_projects_and_internal_delivery(client):
    http, store = client
    goal = await store.create("one", 2, "Finish episode", "owner", "authorization")
    await store.claim(goal["id"], "driver", 60, user_id="owner")
    await store.update(goal["id"], "driver", {"pending_delivery": {"message": "private"}}, user_id="owner")
    await store.create("one", 1, "Other user's episode", "other", "authorization")
    result = await http.get("/api/v1/projects/one/autonomous-production/latest")
    assert result.status_code == 200
    assert result.json()["goal"]["id"] == goal["id"]
    assert "private" not in result.text
    assert "lease_owner" not in result.text
    assert (await http.get(f"/api/v1/projects/two/autonomous-production/{goal['id']}")).status_code == 404
    assert (await http.post(f"/api/v1/projects/two/autonomous-production/{goal['id']}/stop")).status_code == 404


async def test_nonlocal_runtime_cannot_start_autonomous_generation(client, monkeypatch):
    http, _ = client
    service = AsyncMock()
    monkeypatch.setattr(api, "local_only_enabled", lambda: False)
    monkeypatch.setattr(api, "get_autonomous_production", lambda: service)
    response = await http.post(
        "/api/v1/projects/one/autonomous-production",
        json={"episode_id": 2, "objective": "Finish episode", "client_key": "authorization"},
    )
    assert response.status_code == 400
    service.create.assert_not_called()


@pytest.mark.parametrize("field", ["objective", "client_key"])
async def test_blank_authorization_fields_rejected_before_dispatch(client, monkeypatch, field):
    http, _ = client
    service = AsyncMock()
    monkeypatch.setattr(api, "local_only_enabled", lambda: True)
    monkeypatch.setattr(api, "get_autonomous_production", lambda: service)
    body = {"episode_id": 2, "objective": "Finish episode", "client_key": "authorization", field: "  "}
    response = await http.post("/api/v1/projects/one/autonomous-production", json=body)
    assert response.status_code == 422
    service.create.assert_not_called()
