"""Business MCP failures preserve native transport breakers and vision admission."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import build_hermes_environment, prepare_hermes_home


def test_installed_mcp_business_errors_do_not_trip_transport_or_observe_error_pixels(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes runtime is unavailable; no network or model is used")
    config = replace(
        config,
        hermes_home=tmp_path / "hermes",
        hermes_model="local-qwen",
        hermes_api_key="private-test",
        hermes_base_url="http://127.0.0.1:8000/v1",
        resource_root=tmp_path / "resources",
    )
    public = tmp_path / "public.yaml"
    public.write_text("{}")
    home = config.hermes_home / "attempt"
    prepare_hermes_home(config, home, public_config_path=public)
    runtime = Path(__file__).resolve().parents[4] / "server/agent_runtime"
    script = r'''
import asyncio,base64,concurrent.futures,hashlib,importlib,importlib.util,io,json,os,socket,sys,types
from pathlib import Path
def deny_network(*args,**kwargs):raise AssertionError("This test forbids real network access")
socket.socket.connect=deny_network
home=Path(os.environ["HERMES_HOME"])
def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(sys.argv[1])/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
bootstrap=load("bootstrap","hermes_acp_bootstrap.py");bootstrap.install_isolated_env_loader()
fix=load("business_errors","hermes_mcp_errors.py")
bridge=load("bridge","hermes_vision_bridge.py")
native=importlib.import_module("tools.mcp_tool")
from mcp.types import CallToolResult,TextContent,ImageContent
from PIL import Image
buffer=io.BytesIO();Image.new("RGB",(4,3),(255,0,0)).save(buffer,format="JPEG",quality=85)
jpeg=buffer.getvalue();encoded=base64.b64encode(jpeg).decode();wire_sha=hashlib.sha256(jpeg).hexdigest()
rgb=Image.open(io.BytesIO(jpeg)).convert("RGB")
pixel_sha=hashlib.sha256((str(rgb.width)+"x"+str(rgb.height)+"\0").encode("ascii")+rgb.tobytes()).hexdigest()
problem={"code":"version_not_found","action":"fix_input","params":{"unit_id":"S1","version":99,"available_versions":[1]}}
business={"operation":"inspect_images","problem":problem}
structured={"problem":problem}
phase="business"
remaining=3
calls=[]
async def call_tool(name,arguments):
    global remaining
    calls.append((phase,name))
    if phase=="timeout":raise TimeoutError("deterministic transport timeout")
    if phase=="business" and remaining:
        remaining-=1
        return CallToolResult(isError=True,content=[TextContent(type="text",text=json.dumps(business))],structuredContent=structured)
    if phase=="error_image":
        return CallToolResult(isError=True,content=[TextContent(type="text",text=json.dumps(business)),
            ImageContent(type="image",mimeType="image/jpeg",data=encoded)],structuredContent=structured)
    if phase=="valid_image":
        return CallToolResult(content=[TextContent(type="text",text="native successful image"),
            ImageContent(type="image",mimeType="image/jpeg",data=encoded)],structuredContent={"inspect_images":{
            "resource_type":"storyboard","images":[{"id":"S1","version":1,"image":1,
                "source_sha256":wire_sha,"source_pixel_sha256":pixel_sha,"wire_image_sha256":wire_sha}]}})
    return CallToolResult(content=[TextContent(type="text",text="native successful read")],structuredContent={"read":{"ok":True}})
server=types.SimpleNamespace(session=types.SimpleNamespace(call_tool=call_tool),_rpc_lock=asyncio.Lock())
native._get_connected_server_for_call=lambda name:server
native._trust_gate_check=lambda *args:None
rpc_thread=concurrent.futures.ThreadPoolExecutor(max_workers=1)
native._run_on_mcp_loop=lambda factory,**kwargs:rpc_thread.submit(lambda:asyncio.run(factory())).result(timeout=5)
fix.install_mcp_business_errors(native)
vision=bridge.VisionTransport(home,base_url="http://127.0.0.1:8000/v1",model="local-qwen")
vision.install_mcp(native)
handler=native._make_tool_handler("arcreel","inspect_images",1)
read=native._make_tool_handler("arcreel","get_project_content",1)
for count in (3,4):
    remaining=count
    before=len(calls)
    for index in range(count):
        payload=json.loads(handler({"version":99}))
        assert json.loads(payload["error"])==business
        assert payload["structuredContent"]==structured
        assert set(payload)=={"error","structuredContent"}
        assert native._server_error_counts.get("arcreel",0)==0
    success=json.loads(read({}))
    assert success["structuredContent"]=={"read":{"ok":True}}
    assert "error" not in success
    assert len(calls)==before+count+1

# Both wrappers share the native async call context; an error cannot enqueue
# pixels, even if its typed MCP payload includes a real ImageContent block.
import httpx
from openai import OpenAI
sent=[]
def respond(request):
    body=json.loads(request.content);sent.append(body)
    return httpx.Response(200,json={"id":"fake","object":"chat.completion","created":0,"model":"local-qwen",
        "choices":[{"index":0,"message":{"role":"assistant","content":"deterministic response"},"finish_reason":"stop"}]})
vision.install_http(httpx)
client=OpenAI(base_url="http://127.0.0.1:8000/v1",api_key="private-test",http_client=httpx.Client(transport=httpx.MockTransport(respond)))
with vision.agent_scope("child-session",parent_session_id="parent-session") as scope:
    phase="error_image"
    failure=json.loads(handler({"version":99}))
    assert failure["structuredContent"]==structured
    assert json.loads(failure["error"])==business
    assert not scope.pending and not scope.observed_images
    assert encoded not in json.dumps(failure)
    client.chat.completions.create(model="local-qwen",messages=[{"role":"user","content":"Text-only boundary check."}])
    assert not scope.pending and not scope.observed_images
    audit=[json.loads(line) for line in (home/"vision-wire-audit.jsonl").read_text().splitlines()]
    assert not any(row["event"]=="observed_images" for row in audit)
    assert not any(row["event"]=="mcp_images" for row in audit)

    # A subsequent genuine success must still traverse the original MCP
    # handler and vision wrapper and reach a real SDK-serialized image body.
    phase="valid_image"
    image=handler({"version":1})
    assert image["_multimodal"] is True and scope.pending
    client.chat.completions.create(model="local-qwen",messages=[{"role":"user","content":image["content"]}])
    assert not scope.pending
    assert len(scope.observed_images)==1
    assert scope.observed_images[0]["wire_image_sha256"]==wire_sha
    assert len(sent)==2
    assert isinstance(sent[0]["messages"][0]["content"],str)
    parts=sent[1]["messages"][0]["content"]
    pixels=[part for part in parts if part.get("type")=="image_url"]
    assert len(pixels)==1
    assert hashlib.sha256(base64.b64decode(pixels[0]["image_url"]["url"].split(",",1)[1])).hexdigest()==wire_sha
client.close()

# No successful typed RPC result is available on a real exception. The same
# native server must retain its original three-failure breaker and cooldown.
phase="timeout"
before=len(calls)
for index in range(3):
    failure=json.loads(read({}))
    assert "TimeoutError" in failure["error"]
assert len(calls)==before+3
assert native._server_error_counts["arcreel"]==3
blocked=json.loads(read({}))
assert "unreachable" in blocked["error"] and "3 consecutive" in blocked["error"]
assert len(calls)==before+3
assert "arcreel" in native._server_breaker_opened_at
rpc_thread.shutdown(wait=True)
print(json.dumps({"business_sequences":[3,4],"subsequent_success_reached":True,"business_problem_preserved":True,
    "error_pixels_not_observed":True,"valid_pixels_reached_sdk":True,"transport_three_failure_breaker":True}))
'''
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(runtime)],
        env=build_hermes_environment(config, home),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["business_sequences"] == [3, 4]
    assert receipt["subsequent_success_reached"]
    assert receipt["business_problem_preserved"]
    assert receipt["error_pixels_not_observed"]
    assert receipt["valid_pixels_reached_sdk"]
    assert receipt["transport_three_failure_breaker"]
