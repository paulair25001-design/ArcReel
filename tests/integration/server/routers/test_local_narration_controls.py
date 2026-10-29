"""Local voice identity and delivery survive real HTTP and MCP project writes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lib.config.resolver import ConfigResolver
from lib.project.project_manager import ProjectManager
from lib.speech.narration_config import project_tts_settings
from server.remote_mcp import ArcApiKeyVerifier, build_remote_mcp_server, mount_remote_mcp
from server.services.project.workflow_planner import WorkflowPlanner
from server.tool_runtime import Services
from tests.integration.server.routers.projects_router_support import build_projects_client


def _creation(name: str) -> dict[str, object]:
    return {
        "name": name,
        "title": "Local speech controls",
        "generation_mode": "storyboard",
        "content_mode": "narration",
        "narration_delivery": "use_tts",
        "audio_backend": "local-tts/qwen3-tts-voice-design",
        "narration_voice": "local-narrator",
        "narration_speed": None,
        "narration_voice_description": "  成年中文女声，温暖清楚  ",
        "narration_instructions": "  像向朋友解释，重点词稍作停顿。  ",
    }


def test_http_project_voice_controls_create_patch_read_and_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    manager = ProjectManager(tmp_path)
    with build_projects_client(monkeypatch, manager, session_factory=db_factory) as client:
        created = client.post("/api/v1/projects", json=_creation("local-voice"))
        assert created.status_code == 200
        project = client.get("/api/v1/projects/local-voice").json()["project"]
        assert project["narration_voice_description"] == "成年中文女声，温暖清楚"
        assert project["narration_instructions"] == "像向朋友解释，重点词稍作停顿。"

        changed = client.patch(
            "/api/v1/projects/local-voice",
            json={"narration_instructions": "  前半句轻松，结尾自然落句。  "},
        )
        assert changed.status_code == 200
        after_patch = client.get("/api/v1/projects/local-voice").json()["project"]
        settings = project_tts_settings(after_patch)
        assert settings is not None
        assert settings.voice_description == "成年中文女声，温暖清楚"
        assert settings.instructions == "前半句轻松，结尾自然落句。"
        assert manager.load_project("local-voice")["narration_instructions"] == settings.instructions

        refused = client.patch("/api/v1/projects/local-voice", json={"narration_instructions": 17})
        assert refused.status_code == 422
        assert manager.load_project("local-voice")["narration_instructions"] == settings.instructions

        cleared = client.patch(
            "/api/v1/projects/local-voice",
            json={"narration_voice_description": None, "narration_instructions": "  "},
        )
        assert cleared.status_code == 200
        after_clear = client.get("/api/v1/projects/local-voice").json()["project"]
        assert "narration_voice_description" not in after_clear
        assert "narration_instructions" not in after_clear
        restored_defaults = project_tts_settings(after_clear)
        assert restored_defaults is not None
        assert restored_defaults.instructions != settings.instructions


async def test_mcp_schema_and_roundtrip_preserve_local_voice_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    manager = ProjectManager(tmp_path)
    services = Services(
        projects=manager,
        workflow_planner=WorkflowPlanner(manager),
        capabilities=ConfigResolver(db_factory),
    )

    async def verify(token: str):
        return {"sub": "apikey:local-test", "via": "apikey"} if token == "arc-local-test" else None

    server = build_remote_mcp_server(projects=manager, services=services, token_verifier=ArcApiKeyVerifier(verify))
    app = FastAPI()
    mount_remote_mcp(app, server.streamable_http_app())
    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer arc-local-test"},
            follow_redirects=True,
        ) as client,
        streamable_http_client("http://localhost/mcp", http_client=client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        declarations = await session.list_tools()
        creation_schema = next(item.inputSchema for item in declarations.tools if item.name == "create_project")
        assert "narration_voice_description" in creation_schema["properties"]
        assert "narration_instructions" in creation_schema["properties"]
        created = await session.call_tool("create_project", _creation("mcp-voice"))
        assert created.isError is False
        read_back = await session.call_tool("get_project_content", {"project": "mcp-voice"})
        assert read_back.structuredContent is not None
        project = read_back.structuredContent["project_content"]["project"]
        assert project["narration_voice_description"] == "成年中文女声，温暖清楚"
        assert project["narration_instructions"] == "像向朋友解释，重点词稍作停顿。"

        patched = await session.call_tool(
            "patch_project",
            {"project": "mcp-voice", "settings": {"narration_instructions": "  克制地讲述，句尾放低。  "}},
        )
        assert patched.isError is False
        read_back = await session.call_tool("get_project_content", {"project": "mcp-voice"})
        assert read_back.structuredContent is not None
        assert (
            read_back.structuredContent["project_content"]["project"]["narration_instructions"]
            == "克制地讲述，句尾放低。"
        )

        refused = await session.call_tool(
            "patch_project",
            {
                "project": "mcp-voice",
                "settings": {"narration_voice_description": "不应写入", "narration_instructions": []},
            },
        )
        assert refused.isError is True
        persisted = await asyncio.to_thread(manager.load_project, "mcp-voice")
        assert persisted["narration_voice_description"] == "成年中文女声，温暖清楚"

        cleared = await session.call_tool(
            "patch_project",
            {"project": "mcp-voice", "settings": {"narration_voice_description": "  ", "narration_instructions": None}},
        )
        assert cleared.isError is False
        read_back = await session.call_tool("get_project_content", {"project": "mcp-voice"})
        assert read_back.structuredContent is not None
        assert "narration_voice_description" not in read_back.structuredContent["project_content"]["project"]
        assert "narration_instructions" not in read_back.structuredContent["project_content"]["project"]
