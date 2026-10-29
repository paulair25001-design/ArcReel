"""Native child construction, execution and multimodal SDK requests without network."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import build_hermes_environment, prepare_hermes_home
from server.agent_runtime.hermes_native_profile import load_hermes_native_profile


def test_installed_native_children_keep_profile_permissions_pixels_and_public_parameters(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes runtime is unavailable; this test never installs or calls a model")
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
                "agent": {"reasoning_effort": "xhigh", "max_turns": 150},
                "compression": {"checkpoint_required": True},
                "providers": {
                    "omlx": {
                        "extra_body": {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "enable_thinking": True,
                            "chat_template_kwargs": {"reasoning_effort": "xhigh", "preserve_thinking": True},
                        }
                    }
                },
            }
        )
    )
    home = config.hermes_home / "attempt"
    prepare_hermes_home(config, home, agent_tools=True, mcp={}, public_config_path=public)
    runtime = Path(__file__).resolve().parents[4] / "server/agent_runtime"
    manifest = load_hermes_native_profile(
        "narration", profile_root=runtime.parents[1] / "agent_runtime_profile"
    ).to_manifest("demo")
    (home / "native-profile.json").write_text(json.dumps(manifest, ensure_ascii=False))
    script = r"""
import asyncio,base64,hashlib,importlib,importlib.util,io,json,os,socket,sys,threading,types
from pathlib import Path
def forbidden(*args,**kwargs):raise AssertionError("Real network is forbidden")
socket.socket.connect=forbidden
home=Path(os.environ["HERMES_HOME"])
os.environ["ARCREEL_HERMES_EXPECTED_MODEL"]="local-qwen"
os.environ["ARCREEL_HERMES_EXPECTED_BASE_URL"]="http://127.0.0.1:8000/v1"
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(sys.argv[1])/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
bootstrap=load("bootstrap","hermes_acp_bootstrap.py");bootstrap.install_isolated_env_loader()
dispatch=load("dispatch","hermes_native_dispatch.py")
checkpoint=load("checkpoint","hermes_checkpoint.py")
bridge=load("bridge","hermes_vision_bridge.py")
manifest=json.loads((home/"native-profile.json").read_text())
import httpx
from PIL import Image
from mcp.types import CallToolResult,TextContent,ImageContent
buffer=io.BytesIO();Image.new("RGB",(4,3),(255,0,0)).save(buffer,format="JPEG",quality=85)
jpeg=buffer.getvalue();encoded=base64.b64encode(jpeg).decode();jpeg_sha=hashlib.sha256(jpeg).hexdigest()
vision=bridge.VisionTransport(home,base_url="http://127.0.0.1:8000/v1",model="local-qwen")
vision.begin_turn("parent-session")
mcp=importlib.import_module("tools.mcp_tool")
async def call_tool(*args,**kwargs):
    return CallToolResult(content=[TextContent(type="text",text="Deterministic native contact sheet."),
        ImageContent(type="image",mimeType="image/jpeg",data=encoded)],
        structuredContent={"inspect_video_units":{"units":[{"unit_id":"S1","version":1,"status":"ok",
            "sheets":[{"image":1,"times":[0,1,2]}]}],"model_review":None}})
server=types.SimpleNamespace(session=types.SimpleNamespace(call_tool=call_tool),_rpc_lock=asyncio.Lock())
mcp._get_connected_server_for_call=lambda name:server
mcp._trust_gate_check=lambda *args:None
mcp._run_on_mcp_loop=lambda factory,**kwargs:asyncio.run(factory())
vision.install_mcp(mcp)
import model_tools
from tools.registry import registry
def schema(name):return {"name":name,"description":"Deterministic fixture tool.","parameters":{"type":"object","properties":{},"additionalProperties":True}}
read_names=manifest["agents"]["review-footage"]["tools"]
for name in read_names:
    if name in {"read_agent_profile","read_file"}:continue
    handler=mcp._make_tool_handler("arcreel","inspect_video_units",30) if name=="mcp__arcreel__inspect_video_units" else lambda args,**kwargs:json.dumps({"fixture":True})
    registry.register(name=name,toolset="mcp-arcreel",schema=schema(name),handler=handler,check_fn=lambda:True)
registry.register(name="mcp__arcreel__generate_videos",toolset="mcp-arcreel",schema=schema("mcp__arcreel__generate_videos"),
    handler=lambda args,**kwargs:(_ for _ in ()).throw(AssertionError("Review child cannot generate media")),check_fn=lambda:True)
sent=[]
mock_errors=[]
mode="review"
def respond(request):
    if request.method!="POST":return httpx.Response(404)
    assert str(request.url)=="http://127.0.0.1:8000/v1/chat/completions"
    body=json.loads(request.content);sent.append(body)
    assert body["model"]=="local-qwen"
    assert body["max_tokens"]==65536
    assert body["temperature"]==1.0 and body["top_p"]==0.95 and body["top_k"]==20
    assert body["enable_thinking"] is True
    assert body["chat_template_kwargs"]["reasoning_effort"]=="xhigh"
    assert body["chat_template_kwargs"]["preserve_thinking"] is True
    names={item["function"]["name"] for item in body.get("tools",[])}
    assert "dispatch_agent" not in names and "delegate_task" not in names
    assert all(name.startswith("mcp__arcreel__") or name in {"read_agent_profile","read_file"} for name in names)
    system="\n".join(str(message.get("content")) for message in body["messages"] if message["role"]=="system")
    assert manifest["documents"]["CLAUDE.md"]["content"] in system
    if mode=="creation":
        assert manifest["documents"][".claude/agents/create-episode-script.md"]["content"].strip() in system
        assert manifest["documents"][".claude/skills/generate-script/SKILL.md"]["content"].strip() in system
        delta={"role":"assistant","content":"Deterministic preload contract report."};finish="stop"
    else:
        assert manifest["documents"][".claude/agents/review-footage.md"]["content"].strip() in system
        assert names==set(read_names)
        assert "mcp__arcreel__generate_videos" not in names
        if len(sent)==1:
            # A model can emit an unavailable tool despite its schema. The real
            # native executor must refuse it without creating another child.
            delta={"role":"assistant","tool_calls":[{"index":0,"id":"invalid-nested","type":"function",
                "function":{"name":"dispatch_agent","arguments":json.dumps({"agent_type":"review-footage","prompt":"invalid nested attempt"})}}]};finish="tool_calls"
        elif len(sent)==2:
            delta={"role":"assistant","tool_calls":[{"index":0,"id":"cache-page","type":"function",
                "function":{"name":"read_file","arguments":json.dumps({"path":str(spill),"offset":2,"limit":1})}},
                {"index":1,"id":"outside-read","type":"function","function":{"name":"read_file",
                "arguments":json.dumps({"path":str(home/"config.yaml")})}}]};finish="tool_calls"
        elif len(sent)==3:
            results={message.get("tool_call_id"):str(message.get("content")) for message in body["messages"] if message.get("role")=="tool"}
            assert "cache second line" in results["cache-page"]
            assert "cache first line" not in results["cache-page"] and "cache third line" not in results["cache-page"]
            assert "error" in results["outside-read"].lower() and "private-test" not in results["outside-read"]
            delta={"role":"assistant","tool_calls":[{"index":0,"id":"inspect-contact","type":"function",
                "function":{"name":"mcp__arcreel__inspect_video_units","arguments":json.dumps({"project":"demo","unit_ids":["S1"]})}}]};finish="tool_calls"
        else:
            assert len(sent)==4
            urls=[part["image_url"]["url"] for message in body["messages"] for part in message.get("content",[]) if isinstance(message.get("content"),list) and isinstance(part,dict) and part.get("type")=="image_url"]
            assert len(urls)==1 and hashlib.sha256(base64.b64decode(urls[0].split(",",1)[1])).hexdigest()==jpeg_sha
            delta={"role":"assistant","content":"Deterministic review report: S1 v1, sampled frames received."};finish="stop"
    if body.get("stream"):
        chunks=[{"id":"fake","object":"chat.completion.chunk","created":0,"model":"local-qwen","choices":[{"index":0,"delta":delta,"finish_reason":None}]},
            {"id":"fake","object":"chat.completion.chunk","created":0,"model":"local-qwen","choices":[{"index":0,"delta":{},"finish_reason":finish}]}]
        return httpx.Response(200,headers={"content-type":"text/event-stream"},content=("".join("data: "+json.dumps(row)+"\n\n" for row in chunks)+"data: [DONE]\n\n").encode())
    message={k:v for k,v in delta.items()};message["role"]="assistant"
    for call in message.get("tool_calls",[]):call.pop("index",None)
    return httpx.Response(200,json={"id":"fake","object":"chat.completion","created":0,"model":"local-qwen",
        "choices":[{"index":0,"message":message,"finish_reason":finish}]})
def reply(request):
    try:return respond(request)
    except BaseException:
        import traceback
        mock_errors.append(traceback.format_exc())
        raise
httpx.Client._transport_for_url=lambda self,url:httpx.MockTransport(reply)
from run_agent import AIAgent
parent=AIAgent(model="local-qwen",provider="custom",requested_provider="omlx",base_url="http://127.0.0.1:8000/v1",
    api_key="private-test",api_mode="chat_completions",enabled_toolsets=["mcp-arcreel"],max_iterations=150,
    session_id="parent-session",quiet_mode=True,skip_context_files=True,skip_memory=True,skip_background_review=True,save_trajectories=False)
try:
    host=bootstrap.install_native_host_tools(parent,manifest,dispatcher_type=dispatch.NativeAgentDispatcher,
        registry=registry,vision=vision,checkpoint=checkpoint,home=home)
    bootstrap.restrict_agent_tools(parent,model_tools.get_tool_definitions,native_host=True)
    assert not parent._fallback_chain
    spill=home/"cache/spillover/native-fixture.txt"
    spill.parent.mkdir(parents=True,exist_ok=True)
    spill.write_text("cache first line\ncache second line\ncache third line\n")
    page=registry.dispatch("read_file",{"path":str(spill),"offset":2,"limit":1})
    assert "cache second line" in str(page)
    assert "cache first line" not in str(page) and "cache third line" not in str(page)
    outside=home.parent/"outside-project.json"
    outside.write_text('{"private":"outside-project-must-not-be-read"}')
    for denied in (home/"config.yaml",outside):
        failure=registry.dispatch("read_file",{"path":str(denied)})
        assert "error" in str(failure).lower()
        assert "private-test" not in str(failure) and "outside-project-must-not-be-read" not in str(failure)
    vision.install_http(httpx)
    result=registry.dispatch("dispatch_agent",{"agent_type":"review-footage","prompt":"Review the native contact sheet for S1."})
    result=json.loads(result) if isinstance(result,str) else result
    assert result["status"]=="completed",{"result":result,"mock_errors":mock_errors,
        "parameters":[{k:body.get(k) for k in ("max_tokens","enable_thinking","temperature","top_p","top_k","chat_template_kwargs")} for body in sent]}
    assert result["native_completed"] and result["pixels_delivered"]
    assert result["parent_session_id"]=="parent-session" and result["session_id"]!="parent-session"
    assert result["inspections"]==[{"unit_id":"S1","version":1,"wire_image_sha256s":[jpeg_sha]}]
    assert encoded not in json.dumps(result) and "data:image" not in json.dumps(result)
    assert not parent._active_children
    assert len(sent)==4
    audit=[json.loads(row) for row in (home/"vision-wire-audit.jsonl").read_text().splitlines()]
    started=[row for row in audit if row["event"]=="agent_started"]
    assert len(started)==1 and started[0]["session_id"]==result["session_id"]
    delivered=[row for row in audit if row["event"]=="observed_video_units"]
    assert delivered and all(row["session_id"]==result["session_id"] for row in delivered)
    assert all(row.get("parent_session_id")=="parent-session" for row in delivered)
    assert not any(row["event"]=="request" and row["session_id"]=="parent-session" for row in audit)
    reports=[json.loads(row) for row in (home/"native-agent-reports.jsonl").read_text().splitlines()]
    assert reports==[result]
    assert "data:image" not in (home/"native-agent-reports.jsonl").read_text()
    mode="creation"
    creation=host.dispatch_agent(parent,"create-episode-script","Verify the preloaded native skill contract.")
    assert creation["status"]=="completed"
    assert creation["session_id"] not in {"parent-session",result["session_id"]}
    assert len(sent)==5
    print(json.dumps({"native_child":True,"independent_sid":True,"public_parameters":True,"read_only_tools":True,
        "nested_tool_refused":True,"preload_skill":True,"contact_sheet_on_child_wire":True,"parent_report_only":True,
        "spill_page_and_denial_on_child_wire":True,"sdk_requests":len(sent)}))
finally:parent.close()
"""
    environment = build_hermes_environment(config, home)
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(runtime)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["native_child"]
    assert receipt["independent_sid"]
    assert receipt["public_parameters"]
    assert receipt["read_only_tools"]
    assert receipt["nested_tool_refused"]
    assert receipt["preload_skill"]
    assert receipt["contact_sheet_on_child_wire"]
    assert receipt["parent_report_only"]
    assert receipt["spill_page_and_denial_on_child_wire"]
    assert receipt["sdk_requests"] == 5
