"""The installed ACP schema can describe the local model without credential discovery."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config


def test_local_model_state_uses_actual_acp_schema_without_provider_inventory(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes Python is unavailable; this test does not install or launch a model")
    bootstrap = Path(__file__).resolve().parents[4] / "server/agent_runtime/hermes_acp_bootstrap.py"
    script = """
import importlib, importlib.util, json, sys, types
calls = {"inventory": 0, "pool": 0}
def forbidden_inventory(*args, **kwargs):
    calls["inventory"] += 1
    raise AssertionError("Provider inventory is not allowed")
def forbidden_pool(*args, **kwargs):
    calls["pool"] += 1
    raise AssertionError("Credential pool discovery is not allowed")
inventory = types.ModuleType("hermes_cli.inventory")
inventory.load_picker_context = forbidden_inventory
inventory.build_models_payload = forbidden_inventory
sys.modules[inventory.__name__] = inventory
pool = types.ModuleType("agent.credential_pool")
pool.load_pool = forbidden_pool
sys.modules[pool.__name__] = pool
schema = importlib.import_module("acp.schema")
spec = importlib.util.spec_from_file_location("arcreel_bootstrap", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
agent = types.SimpleNamespace(model="local-qwen", base_url="http://127.0.0.1:8000/v1")
state = module.build_local_model_state(agent, model="local-qwen", base_url=agent.base_url, schema=schema)
payload = state.model_dump(mode="json", by_alias=True)
restored = schema.SessionModelState.model_validate(payload)
print(json.dumps({"payload": payload, "model_ids": [row.model_id for row in restored.available_models], "calls": calls}))
"""
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(bootstrap)],
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(config.hermes_source)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    receipt = json.loads(result.stdout)

    assert receipt["model_ids"] == ["omlx:local-qwen"]
    assert receipt["payload"]["currentModelId"] == "omlx:local-qwen"
    assert receipt["calls"] == {"inventory": 0, "pool": 0}
