"""Local Agent assembly uses real scoped credentials without cloud configuration."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from lib.db.base import DEFAULT_USER_ID
from lib.infra.data_root_layout import DataRootLayout
from lib.local_runtime.config import get_local_runtime_config
from server.agent_runtime.hermes_client import HermesAgentOptions
from server.agent_runtime.options_assembler import OptionsAssembler, agent_pricing_of
from server.remote_mcp import ArcApiKeyVerifier


def _assembler(data_root: Path, project: str) -> OptionsAssembler:
    cwd = DataRootLayout(data_root).projects_dir / project
    cwd.mkdir(parents=True)

    def no_legacy_policy():
        raise AssertionError("The local Agent must not assemble the Claude sandbox or legacy profile")

    async def no_cloud_credentials():
        raise AssertionError("The local Agent must not read cloud credentials")

    return OptionsAssembler(
        data_root=data_root,
        allowed_tools=[],
        setting_sources=[],
        access_policy_provider=no_legacy_policy,
        max_turns_provider=lambda: None,
        resolve_project_cwd=lambda name: DataRootLayout(data_root).projects_dir / name,
        provider_env_loader=no_cloud_credentials,
    )


async def test_local_options_use_real_current_project_scope_without_cloud_reads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    monkeypatch.setenv("ARCREEL_HERMES_MCP_TOKEN", "unscoped-token-from-environment")
    layout = DataRootLayout.current()
    project = "hermes-options-" + uuid4().hex
    assembler = _assembler(layout.root, project)

    options = await assembler.build(project)

    assert isinstance(options, HermesAgentOptions)
    assert options.cwd == layout.projects_dir / project
    assert options.config.hermes_mcp_token != "unscoped-token-from-environment"
    access = await ArcApiKeyVerifier().verify_token(options.config.hermes_mcp_token)
    assert access is not None
    assert access.scopes == ["arcreel", "project:" + project]
    assert project in options.system_prompt
    assert "get_workflow_plan" in options.system_prompt
    assert "confirm_script_review" in options.system_prompt
    assert "get_video_capabilities" in options.system_prompt
    assert "finish_local_session" not in options.system_prompt
    assert "独立审核须另开" not in options.system_prompt
    assert "FastH3" not in options.system_prompt
    assert options.env == {}
    assert agent_pricing_of(options).multiplier == 0
    assert assembler.build_session_store() is None


async def test_local_resume_requires_an_existing_owned_binding(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    layout = DataRootLayout.current()
    project = "hermes-resume-" + uuid4().hex
    assembler = _assembler(layout.root, project)
    config = get_local_runtime_config()
    home = config.hermes_home / "sessions" / "owned-session"
    home.mkdir(parents=True)
    bindings = config.hermes_home / "bindings"
    bindings.mkdir()
    (bindings / "recorded-session.json").write_text(
        json.dumps(
            {"schema": "arcreel-hermes-session/v1", "home": "sessions/owned-session", "user_id": DEFAULT_USER_ID}
        ),
        encoding="utf-8",
    )

    options = await assembler.build(project, resume_id="recorded-session")

    assert isinstance(options, HermesAgentOptions)
    assert options.resume == "recorded-session"
    assert options.home == home
    access = await ArcApiKeyVerifier().verify_token(options.config.hermes_mcp_token)
    assert access is not None
    assert "project:" + project in access.scopes
    with pytest.raises(RuntimeError, match="binding is missing"):
        await assembler.build(project, resume_id="missing-session")
