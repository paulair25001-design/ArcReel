"""The installed native compression host requires a durable private checkpoint."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import build_hermes_environment, prepare_hermes_home


def test_native_required_checkpoint_precedes_summary_and_survives_failed_compression(tmp_path: Path):
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
                "compression": {"enabled": True, "checkpoint_required": True, "tail_mode": "legacy"},
            }
        )
    )
    home = config.hermes_home / "attempt"
    prepare_hermes_home(config, home, public_config_path=public)
    runtime = Path(__file__).resolve().parents[4] / "server/agent_runtime"
    script = r'''
import copy, hashlib, importlib.util, json, os, socket, sys
from pathlib import Path
def deny_network(*args, **kwargs):raise AssertionError("No real network is permitted")
socket.socket.connect=deny_network
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(sys.argv[1])/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
bootstrap=load("arcreel_bootstrap","hermes_acp_bootstrap.py")
bootstrap.install_isolated_env_loader()
checkpoint=load("arcreel_checkpoint","hermes_checkpoint.py")
from agent.memory_manager import MemoryManager
from agent.conversation_compression import compress_context,CompressionCheckpointUnavailable
from run_agent import AIAgent
import httpx
home=Path(os.environ["HERMES_HOME"])
requests=[]
finish="length"
def respond(request):
    if request.method!="POST":return httpx.Response(404)
    snapshots=list(home.glob("checkpoints/pre-compress/native-session/*.json"))
    assert len(snapshots)==1,"Checkpoint must be durable before the first summary request"
    saved=json.loads(snapshots[0].read_bytes())
    assert saved["messages"]==original
    assert saved["evidence_messages"] and all(row["role"]!="tool" for row in saved["evidence_messages"])
    assert hashlib.sha256(snapshots[0].read_bytes()).hexdigest()==snapshots[0].stem
    body=json.loads(request.content);requests.append(body)
    summary="## Active Task\nContinue the user's native ArcReel task with all accepted artifacts preserved.\n## Progress\nText context archived; no new media submitted. Keep the same project and use native tools."
    if body.get("stream"):
        chunks=[{"id":"summary","object":"chat.completion.chunk","created":0,"model":"local-qwen",
            "choices":[{"index":0,"delta":{"role":"assistant","content":summary},"finish_reason":None}]},
            {"id":"summary","object":"chat.completion.chunk","created":0,"model":"local-qwen",
            "choices":[{"index":0,"delta":{},"finish_reason":finish}]}]
        return httpx.Response(200,headers={"content-type":"text/event-stream"},content=("".join("data: "+json.dumps(row)+"\n\n" for row in chunks)+"data: [DONE]\n\n").encode())
    return httpx.Response(200,json={"id":"summary","object":"chat.completion","created":0,"model":"local-qwen",
        "choices":[{"index":0,"message":{"role":"assistant","content":summary},"finish_reason":finish}]})
httpx.Client._transport_for_url=lambda self,url:httpx.MockTransport(respond)
agent=AIAgent(model="local-qwen",provider="custom",requested_provider="omlx",base_url="http://127.0.0.1:8000/v1",
    api_key="private-test",api_mode="chat_completions",enabled_toolsets=["context_engine"],session_id="native-session",
    max_iterations=2,quiet_mode=True,skip_context_files=True,skip_memory=True,skip_background_review=True,save_trajectories=False)
messages=[{"role":"system","content":"Native checkpoint contract test."}]
for index in range(20):
    messages.extend([{"role":"user","content":("User evidence %d. " % index)*500},
        {"role":"assistant","content":("Assistant context %d. " % index)*500}])
messages.insert(3,{"role":"tool","tool_call_id":"t1","content":[{"type":"text","text":"tool evidence"},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,PRIVATE_TEST_BYTES"}}]})
original=copy.deepcopy(messages)
try:
    assert agent.compression_checkpoint_required is True
    try:
        compress_context(agent,messages,"Native test",approx_tokens=100000,force=True)
    except CompressionCheckpointUnavailable:pass
    else:raise AssertionError("Missing checkpoint must fail closed")
    assert not requests and messages==original
    checkpoint.install_project_checkpoint(agent,home,MemoryManager)
    assert isinstance(agent._memory_manager,MemoryManager)
    assert agent._memory_manager.providers==[]
    assert agent._memory_manager.supports_pre_compress_checkpoint(2)
    assert not agent._memory_manager.supports_pre_compress_checkpoint(3)
    result,_=compress_context(agent,messages,"Native test",approx_tokens=100000,force=True)
    assert requests and result==original and messages==original
    assert agent.context_compressor.compression_count==0
    snapshots=list(home.glob("checkpoints/pre-compress/native-session/*.json"))
    assert len(snapshots)==1
    cold=checkpoint.TranscriptCheckpoint(home,lambda:"native-session")
    assert cold.read_checkpoint("native-session",snapshots[0].stem)["messages"]==original
    finish="stop"
    result,_=compress_context(agent,messages,"Native test",approx_tokens=100000,force=True)
    assert len(result)<len(original) and agent.context_compressor.compression_count>0
    assert cold.read_checkpoint("native-session",snapshots[0].stem)["messages"]==original
    assert all(body["model"]=="local-qwen" for body in requests)
    print(json.dumps({"required":True,"private_snapshot":True,"raw_and_evidence_preserved":True,
        "failed_summary_preserved_history":True,"native_success":True,"summary_requests":len(requests)}))
finally:agent.close()
'''
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(runtime)],
        cwd=tmp_path,
        env=build_hermes_environment(config, home),
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["required"]
    assert receipt["private_snapshot"]
    assert receipt["raw_and_evidence_preserved"]
    assert receipt["failed_summary_preserved_history"]
    assert receipt["native_success"]
    assert receipt["summary_requests"] >= 2
