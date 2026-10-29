"""Only public inference knobs enter the private local route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import prepare_hermes_home, read_public_hermes_parameters


def test_public_snapshot_keeps_shared_knobs_and_excludes_credentials_and_workflows(tmp_path: Path):
    public = tmp_path / "public.yaml"
    public.write_text(
        json.dumps(
            {
                "model": {"context_length": 262144, "max_tokens": 65536, "default": "cloud-secret"},
                "agent": {"reasoning_effort": "xhigh", "max_turns": 150, "system_prompt": "old-flow"},
                "compression": {
                    "threshold": 0.8,
                    "tail_mode": "legacy",
                    "context_timeout_seconds": 1800,
                    "context_total_ceiling_seconds": 1800,
                    "model_headroom_tokens": {"Qwen": 65536},
                },
                "providers": {
                    "omlx": {
                        "api_key": "cloud-secret",
                        "base_url": "https://cloud-secret.invalid",
                        "extra_body": {
                            "enable_thinking": True,
                            "top_p": 0.95,
                            "api_key": "cloud-secret",
                            "chat_template_kwargs": {
                                "reasoning_effort": "xhigh",
                                "preserve_thinking": True,
                                "system_prompt": "old-flow",
                            },
                        },
                    }
                },
                "auxiliary": {
                    "compression": {
                        "provider": "cloud-secret",
                        "reasoning_effort": "none",
                        "max_output_tokens": 16384,
                        "timeout": 1800,
                    }
                },
                "skills": ["old-flow"],
                "mcp_servers": {"old-flow": {}},
                "fallback_providers": ["cloud-secret"],
            }
        )
    )
    before = hashlib.sha256(public.read_bytes()).hexdigest()
    parameters = read_public_hermes_parameters(public)
    assert parameters["agent"] == {"reasoning_effort": "xhigh", "max_turns": 150}
    assert parameters["provider"]["extra_body"] == {
        "enable_thinking": True,
        "top_p": 0.95,
        "chat_template_kwargs": {"reasoning_effort": "xhigh", "preserve_thinking": True},
    }
    config = replace(get_local_runtime_config(), hermes_home=tmp_path / "private", hermes_api_key="local-only")
    prepare_hermes_home(config, config.hermes_home / "attempt", public_config_path=public)
    profile = json.loads((config.hermes_home / "attempt/config.yaml").read_text())
    assert profile["model"]["max_tokens"] == 65536
    assert profile["compression"]["model_headroom_tokens"] == {"Qwen": 65536}
    assert profile["compression"]["context_total_ceiling_seconds"] == 1800
    assert profile["auxiliary"]["compression"]["max_output_tokens"] == 16384
    assert profile["auxiliary"]["compression"]["provider"] == "omlx"
    assert "cloud-secret" not in json.dumps(profile)
    assert "old-flow" not in json.dumps(profile)
    assert hashlib.sha256(public.read_bytes()).hexdigest() == before


def test_missing_public_fields_use_native_defaults_without_private_overrides(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("{}")
    config = replace(get_local_runtime_config(), hermes_home=tmp_path / "private", hermes_api_key="local-only")
    prepare_hermes_home(config, config.hermes_home / "attempt", public_config_path=path)
    profile = json.loads((config.hermes_home / "attempt/config.yaml").read_text())
    assert "context_length" not in profile["model"]
    assert "max_tokens" not in profile["model"]
    assert "threshold_tokens" not in profile["compression"]
    assert "context_total_ceiling_seconds" not in profile["compression"]
    assert profile["auxiliary"]["compression"] == {
        "provider": "omlx",
        "model": config.hermes_model,
        "base_url": config.hermes_base_url,
    }
