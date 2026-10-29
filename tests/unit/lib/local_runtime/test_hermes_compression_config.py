"""Only the private auxiliary summary route receives non-thinking controls."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import prepare_hermes_home


@pytest.mark.parametrize("agent_tools", [False, True])
def test_private_profile_limits_only_compression(tmp_path: Path, agent_tools: bool):
    config = replace(
        get_local_runtime_config(), hermes_home=tmp_path, hermes_api_key="private-test", hermes_mcp_token="mcp-test"
    )
    home = tmp_path / "private"
    prepare_hermes_home(config, home, agent_tools=agent_tools)
    profile = json.loads((home / "config.yaml").read_text())
    assert profile["model"]["max_tokens"] == 65536
    assert profile["model"]["default"] == config.hermes_model
    assert profile["model"]["base_url"] == config.hermes_base_url
    assert profile["providers"]["omlx"]["api_key"] == "private-test"
    assert "extra_body" not in profile["model"]
    assert profile["auxiliary"]["vision"] == {
        "provider": "omlx",
        "model": config.hermes_model,
        "base_url": config.hermes_base_url,
    }
    summary = profile["auxiliary"]["compression"]
    assert summary["provider"] == "omlx"
    assert summary["model"] == config.hermes_model
    assert summary["base_url"] == config.hermes_base_url
    assert summary["reasoning_effort"] == "none"
    assert summary["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert summary["max_output_tokens"] == 8192
    assert profile["compression"] == {
        "enabled": agent_tools,
        "threshold_tokens": 150000,
        "tail_mode": "legacy",
        "context_timeout_seconds": 120,
        "context_total_ceiling_seconds": 600,
    }
