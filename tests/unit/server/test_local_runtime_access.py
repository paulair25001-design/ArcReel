"""Project-scoped Hermes MCP credentials cannot access sibling projects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from lib.project.project_manager import ProjectManager
from server.agent_toolset.content_read import GET_PROJECT_CONTENT
from server.agent_toolset.project_entry import CREATE_PROJECT, LIST_PROJECTS
from server.agent_toolset.remote import remote_tool
from server.auth import API_KEY_PREFIX
from server.remote_mcp import ArcApiKeyVerifier
from server.tool_runtime import Services


@pytest.mark.parametrize(
    ("subject", "scopes"),
    [("apikey:internal-hermes-project:demo", ["arcreel", "project:demo"]), ("apikey:normal-user-key", ["arcreel"])],
)
async def test_verifier_preserves_the_authenticated_project_scope(subject: str, scopes: list[str]):
    async def verify(_token: str):
        return {"sub": subject, "via": "apikey"}

    access = await ArcApiKeyVerifier(verify_api_key=verify).verify_token(API_KEY_PREFIX + "test-key")

    assert access is not None
    assert access.scopes == scopes


async def test_project_token_can_read_its_current_project_but_cannot_escape(tmp_path: Path):
    projects = ProjectManager(tmp_path)
    cwd = projects.projects_dir / "demo"
    cwd.mkdir()
    (cwd / "project.json").write_text(
        json.dumps({"title": "Current local project", "content_mode": "narration"}), encoding="utf-8"
    )
    services = Services.defaults(projects)
    access = AccessToken(token=API_KEY_PREFIX + "test", client_id="hermes", scopes=["arcreel", "project:demo"])
    marker = auth_context_var.set(AuthenticatedUser(access))
    try:
        read = remote_tool(GET_PROJECT_CONTENT, projects=projects, services=services)
        own = await read.run({"project": "demo"})
        sibling = await read.run({"project": "sibling"})
        list_result = await remote_tool(LIST_PROJECTS, projects=projects, services=services).run({})
        create_result = await remote_tool(CREATE_PROJECT, projects=projects, services=services).run({"name": "sibling"})
    finally:
        auth_context_var.reset(marker)

    assert own.isError is False
    assert own.structuredContent["project_content"]["project"]["title"] == "Current local project"
    for refused in (sibling, list_result, create_result):
        assert refused.isError is True
        assert refused.structuredContent["problem"]["code"] == "project_scope_violation"
    assert not (projects.projects_dir / "sibling").exists()
