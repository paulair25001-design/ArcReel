"""Local-only HTTP probes must never dispatch alternate inference handlers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.local_runtime import LocalOnlyProbeMiddleware


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/providers/openai/test",
        "/api/v1/providers/l20/models/discover",
        "/api/v1/custom-providers/test",
        "/api/v1/custom-providers/discover-anthropic",
        "/api/v1/custom-endpoints/trial-runs",
        "/api/v1/agent/test-connection",
        "/api/v1/agent/credentials/1/test",
    ],
)
def test_local_policy_refuses_probe_before_handler(monkeypatch: pytest.MonkeyPatch, path: str):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    app = FastAPI()
    calls: list[str] = []

    @app.post(path)
    async def alternate_inference():
        calls.append(path)
        return {"called": True}

    app.add_middleware(LocalOnlyProbeMiddleware)
    with TestClient(app) as client:
        response = client.post(path, headers={"Accept-Language": "en"})
    assert response.status_code == 403
    assert response.json()["code"] == "local_only_policy"
    assert "Local Hermes" in response.json()["detail"]
    assert calls == []


def test_local_policy_leaves_project_generation_and_read_only_trial_records_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    app = FastAPI()

    @app.post("/api/v1/projects/demo/generate")
    async def local_generation():
        return {"local": True}

    @app.get("/api/v1/custom-endpoints/trial-runs/history")
    async def historical_record():
        return {"status": "complete"}

    app.add_middleware(LocalOnlyProbeMiddleware)
    with TestClient(app) as client:
        assert client.post("/api/v1/projects/demo/generate").json() == {"local": True}
        assert client.get("/api/v1/custom-endpoints/trial-runs/history").json() == {"status": "complete"}
