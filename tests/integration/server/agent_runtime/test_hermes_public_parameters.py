"""Shared Hermes knobs reach the real native parser and SDK without network."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import build_hermes_environment, prepare_hermes_home


def test_public_main_and_compression_parameters_reach_native_sdk(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes runtime is unavailable")
    config = replace(
        config,
        hermes_home=tmp_path / "hermes",
        hermes_model="local-qwen",
        hermes_api_key="private-test",
        hermes_base_url="http://127.0.0.1:8000/v1",
        resource_root=tmp_path / "resources",
    )
    public = tmp_path / "public.yaml"
    public.write_text(
        json.dumps(
            {
                "model": {"context_length": 262144, "max_tokens": 65536},
                "agent": {"reasoning_effort": "xhigh", "max_turns": 150, "system_prompt": "old-workflow-do-not-copy"},
                "compression": {
                    "threshold": 0.8,
                    "threshold_tokens": 150000,
                    "target_ratio": 0.2,
                    "tail_mode": "legacy",
                    "protect_last_n": 6,
                    "context_timeout_seconds": 1800,
                    "context_total_ceiling_seconds": 1800,
                    "model_headroom_tokens": {"local-qwen": 65536},
                },
                "providers": {
                    "omlx": {
                        "api_key": "public-secret-do-not-copy",
                        "base_url": "https://wrong.invalid",
                        "resource_queue_helper": "old-factory-helper",
                        "extra_body": {
                            "enable_thinking": True,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "chat_template_kwargs": {"preserve_thinking": True, "reasoning_effort": "xhigh"},
                        },
                    }
                },
                "auxiliary": {
                    "compression": {
                        "provider": "cloud-do-not-copy",
                        "model": "other-model",
                        "reasoning_effort": "none",
                        "max_output_tokens": 16384,
                        "context_length": 262144,
                        "timeout": 1800,
                        "extra_body": {
                            "enable_thinking": False,
                            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
                        },
                    }
                },
                "mcp_servers": {"old-workflow": {}},
                "fallback_providers": ["cloud-do-not-copy"],
            }
        )
    )
    home = config.hermes_home / "attempt"
    prepare_hermes_home(config, home, public_config_path=public)
    profile = (home / "config.yaml").read_text()
    for forbidden in (
        "public-secret-do-not-copy",
        "wrong.invalid",
        "old-factory-helper",
        "old-workflow",
        "cloud-do-not-copy",
        "other-model",
    ):
        assert forbidden not in profile
    script = r"""
import importlib, importlib.util, json, socket, sys
from pathlib import Path
def forbidden(*args, **kwargs):raise AssertionError("This test cannot access the network")
socket.socket.connect=forbidden
spec=importlib.util.spec_from_file_location("bootstrap",sys.argv[1]);bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
bootstrap.install_isolated_env_loader()
from hermes_cli.config import load_config_readonly
profile=load_config_readonly()
import httpx
sent=[]
def reply(request):
    if request.method!="POST":return httpx.Response(404)
    body=json.loads(request.content);sent.append(body)
    return httpx.Response(200,json={"id":"fake","object":"chat.completion","created":0,"model":"local-qwen",
        "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})
httpx.Client._transport_for_url=lambda self,url:httpx.MockTransport(reply)
from run_agent import AIAgent
agent=AIAgent(model="local-qwen",provider="custom",requested_provider="omlx",base_url="http://127.0.0.1:8000/v1",
    api_key="private-test",api_mode="chat_completions",enabled_toolsets=["context_engine"],max_iterations=2,
    quiet_mode=True,skip_context_files=True,skip_memory=True,skip_background_review=True,save_trajectories=False)
try:
    kwargs=agent._build_api_kwargs([{"role":"user","content":"Serialization test."}],[])
    agent.client.chat.completions.create(**kwargs)
finally:agent.close()
aux=importlib.import_module("agent.auxiliary_client")
aux.call_llm(task="compression",messages=[{"role":"user","content":"Serialization test."}],
    main_runtime={"model":"local-qwen","provider":"custom","base_url":"http://127.0.0.1:8000/v1","api_key":"private-test","api_mode":"chat_completions"})
main,summary=sent
assert main["model"]==summary["model"]=="local-qwen"
assert main["enable_thinking"] is True
assert main["chat_template_kwargs"]["reasoning_effort"]=="xhigh"
assert main["chat_template_kwargs"]["preserve_thinking"] is True
assert main["max_tokens"]==65536
assert main["temperature"]==1.0 and main["top_p"]==0.95 and main["top_k"]==20
assert summary["chat_template_kwargs"]["enable_thinking"] is False
assert summary["max_tokens"]==16384
assert profile["compression"]["threshold"]==0.8
assert profile["compression"]["context_total_ceiling_seconds"]==1800
assert not hasattr(bootstrap,"configure_project_compression")
print(json.dumps({"main":{k:main.get(k) for k in ("model","max_tokens","enable_thinking","chat_template_kwargs","temperature","top_p","top_k")},
    "summary":{k:summary.get(k) for k in ("model","max_tokens","chat_template_kwargs")},"shared_deadline":1800}))
"""
    result = subprocess.run(
        [
            config.hermes_python,
            "-c",
            script,
            str(Path(__file__).resolve().parents[4] / "server/agent_runtime/hermes_acp_bootstrap.py"),
        ],
        cwd=tmp_path,
        env=build_hermes_environment(config, home),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["main"]["enable_thinking"] is True
    assert receipt["main"]["max_tokens"] == 65536
    assert receipt["summary"]["max_tokens"] == 16384
    assert receipt["shared_deadline"] == 1800
