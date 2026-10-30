"""Installed Hermes consumes real MCP image blocks before OpenAI serialization."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.hermes_process import prepare_hermes_home


def test_installed_mcp_and_model_image_pipeline_uses_pixels_without_network(tmp_path: Path):
    config = get_local_runtime_config()
    if not Path(config.hermes_python).is_file():
        pytest.skip("Installed Hermes runtime is unavailable; no model is installed by this test")
    config = replace(
        config,
        hermes_home=tmp_path / "hermes",
        hermes_model="local-qwen",
        hermes_base_url="http://127.0.0.1:8000/v1",
        hermes_api_key="private-test",
    )
    home = config.hermes_home / "attempt"
    public = tmp_path / "public.yaml"
    public.write_text(
        json.dumps(
            {
                "model": {"context_length": 262144, "max_tokens": 65536},
                "agent": {"reasoning_effort": "xhigh", "max_turns": 150},
                "compression": {
                    "threshold": 0.8,
                    "threshold_tokens": 150000,
                    "tail_mode": "legacy",
                    "context_timeout_seconds": 1800,
                    "context_total_ceiling_seconds": 1800,
                },
                "auxiliary": {
                    "compression": {
                        "reasoning_effort": "none",
                        "max_output_tokens": 16384,
                        "timeout": 1800,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    }
                },
            }
        )
    )
    prepare_hermes_home(config, home, public_config_path=public)
    runtime = Path(__file__).resolve().parents[4] / "server/agent_runtime"
    script = r"""
import asyncio, base64, hashlib, importlib, importlib.util, io, json, os, socket, sys, types
from pathlib import Path
def no_network(*args, **kwargs):
    raise AssertionError("Network is forbidden in the installed-runtime integration test")
socket.socket.connect = no_network
home = Path(os.environ["HERMES_HOME"])
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(sys.argv[1]) / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
bootstrap = load("arcreel_bootstrap", "hermes_acp_bootstrap.py")
bootstrap.install_isolated_env_loader()
from hermes_cli.config import load_config_readonly
profile = load_config_readonly()
assert profile["model"]["max_tokens"] == 65536
assert "max_summary_tokens" not in profile["compression"]
bridge = load("arcreel_vision", "hermes_vision_bridge.py")
native = importlib.import_module("tools.mcp_tool")
from mcp.types import CallToolResult, TextContent, ImageContent
from PIL import Image
buffer = io.BytesIO(); Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format="JPEG", quality=85)
jpeg = buffer.getvalue()
source = Image.open(io.BytesIO(jpeg)).convert("RGB")
source_pixel_sha = hashlib.sha256((str(source.width) + "x" + str(source.height) + "\0").encode("ascii") + source.tobytes()).hexdigest()

versions = []
async def call_tool(*args, **kwargs):
    versions.append(len(versions) + 1)
    count = 1 if len(versions) == 1 else 8
    return CallToolResult(content=[TextContent(type="text", text="one native image"),
        *(ImageContent(type="image", data=base64.b64encode(jpeg).decode(), mimeType="image/jpeg") for _ in range(count))],
        structuredContent={"inspect_images": {"resource_type": "storyboard", "images": [
            {"id": f"S{index:02d}", "version": versions[-1], "image": index + 1,
             "prompt": "Technical image description. " * 30, "source_sha256": hashlib.sha256(jpeg).hexdigest(), "source_pixel_sha256": source_pixel_sha, "wire_image_sha256": hashlib.sha256(jpeg).hexdigest()} for index in range(count)]}})
server = types.SimpleNamespace(session=types.SimpleNamespace(call_tool=call_tool), _rpc_lock=asyncio.Lock())
native._get_connected_server_for_call = lambda name: server
native._trust_gate_check = lambda *args: None
native._run_on_mcp_loop = lambda factory, **kwargs: asyncio.run(factory())
transport = bridge.VisionTransport(home, base_url="http://127.0.0.1:8000/v1", model="local-qwen")
transport.begin_turn("native-test")
transport.install_mcp(native)
result = native._make_tool_handler("arcreel", "inspect_images", 30)({})
from collections import deque
from acp_adapter import events, server as acp_server
updates = []
events._send_update = lambda conn, session, loop, update: updates.append(update.model_dump(mode="json", by_alias=True))
bridge.install_acp_text_projection(acp_server, events)
callback = acp_server.make_step_cb(None, "native-test", None,
    {"mcp__arcreel__inspect_images": deque(["call-1"])}, {"call-1": {}})
callback(1, [{"name": "mcp__arcreel__inspect_images", "result": result}])
callback_list = acp_server.make_step_cb(None, "native-test", None,
    {"mcp__arcreel__inspect_images": deque(["call-2"])}, {"call-2": {}})
callback_list(2, [{"name": "mcp__arcreel__inspect_images", "result": result["content"]}])
ui_json = json.dumps(updates)
assert updates and "data:image" not in ui_json and base64.b64encode(jpeg).decode() not in ui_json
from run_agent import AIAgent
agent = object.__new__(AIAgent)
agent.model = "local-qwen"
agent.provider = "custom"
agent.base_url = "http://127.0.0.1:8000/v1"
vision = agent._model_supports_vision()
tool_vision = agent._provider_supports_vision_tool_messages()
content = agent._tool_result_content_for_active_model("mcp__arcreel__inspect_images", result)
assert vision and tool_vision and isinstance(content, list)
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.model_metadata import estimate_messages_tokens_rough
canonical = make_tool_result_message("mcp__arcreel__inspect_images", content, "c1")
assert isinstance(canonical["content"], list) and canonical["content"][1]["type"] == "image_url"
larger = [{"type": "text", "text": content[0]["text"]}, {"type": "image_url", "image_url": {
    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg + bytes(200000)).decode()}}]
large_canonical = make_tool_result_message("mcp__arcreel__inspect_images", larger, "c1")
small_estimate = estimate_messages_tokens_rough([canonical])
assert estimate_messages_tokens_rough([large_canonical]) == small_estimate
import httpx
from openai import OpenAI
seen = []
def respond(request):
    body = json.loads(request.content)
    part = next(part for part in body["messages"][-1]["content"] if part.get("type") == "image_url")
    image = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    seen.append({"image_sha256": hashlib.sha256(image).hexdigest(), "role": body["messages"][-1]["role"]})
    return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
        "model": "local-qwen", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop"}]})
transport.install_http(httpx)
with OpenAI(api_key="private-test", base_url=agent.base_url,
    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
    client.chat.completions.create(model=agent.model, messages=[
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "mcp__arcreel__inspect_images", "arguments": "{}"}}]},
        {key: value for key, value in canonical.items() if key in {"role", "content", "tool_call_id", "name"}}])
auxiliary = importlib.import_module("agent.auxiliary_client")
compressor_module = importlib.import_module("agent.context_compressor")
transport.install_auxiliary(auxiliary)
transport.install_compression(compressor_module, auxiliary)
batch_a = native._make_tool_handler("arcreel", "inspect_images", 30)({})
batch_b = native._make_tool_handler("arcreel", "inspect_images", 30)({})
compression_seen = []
summary_finish_reason = "stop"
def summary_response(request):
    body = json.loads(request.content)
    count = sum(1 for message in body["messages"] if isinstance(message.get("content"), list)
        for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url")
    compression_seen.append(count)
    if count == 0:
        assert body["chat_template_kwargs"]["enable_thinking"] == profile["auxiliary"]["compression"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        assert body["max_tokens"] == profile["auxiliary"]["compression"]["max_output_tokens"]
    else:
        assert "chat_template_kwargs" not in body
    summary = "## Active Task\nContinue the user's native production task. Inspect the pending A/B images before deciding actions.\n## Progress\nEarlier text facts summarized; no media submitted."
    if body.get("stream"):
        chunks = [{"id": "summary", "object": "chat.completion.chunk", "created": 0, "model": "local-qwen",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": summary}, "finish_reason": None}]},
            {"id": "summary", "object": "chat.completion.chunk", "created": 0, "model": "local-qwen",
            "choices": [{"index": 0, "delta": {}, "finish_reason": summary_finish_reason}]}]
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
            content="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")
    return httpx.Response(200, json={"id": "summary", "object": "chat.completion", "created": 0,
        "model": "local-qwen", "choices": [{"index": 0, "message": {"role": "assistant",
        "content": summary}, "finish_reason": summary_finish_reason}]})
original_init = httpx.Client.__init__
def mocked_init(self, *args, **kwargs):
    kwargs["transport"] = httpx.MockTransport(summary_response)
    kwargs["mounts"] = {}
    kwargs["trust_env"] = False
    return original_init(self, *args, **kwargs)
httpx.Client.__init__ = mocked_init
history = [{"role": "user", "content": "Continue the native production task."}]
for index in range(18):
    history.extend([{"role": "user", "content": f"Earlier task context {index}: " + "technical task history. " * 1000},
        {"role": "assistant", "content": f"Earlier context {index} recorded."}])
history.extend([{"role": "assistant", "content": None, "tool_calls": [
    {"id": "a", "type": "function", "function": {"name": "mcp__arcreel__inspect_images", "arguments": "{}"}},
    {"id": "b", "type": "function", "function": {"name": "mcp__arcreel__inspect_images", "arguments": "{}"}}]},
    make_tool_result_message("mcp__arcreel__inspect_images", batch_a["content"], "a"),
    make_tool_result_message("mcp__arcreel__inspect_images", batch_b["content"], "b")])
compressor = compressor_module.ContextCompressor(model="local-qwen", provider="custom", base_url=agent.base_url,
    api_key="private-test", config_context_length=262144, quiet_mode=True, protect_first_n=2,
    protect_last_n=4, tail_mode=profile["compression"].get("tail_mode", "lean"), threshold_tokens_cap=5000)
compressed = compressor.compress(history, current_tokens=180000, force=True)
def count_images(messages):
    return sum(1 for message in messages if isinstance(message.get("content"), list)
        for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url")
assert compressor.compression_count > 0 and not compressor._last_compress_aborted
assert compression_seen == [0], compression_seen
assert count_images(compressed) == 16, count_images(compressed)
with OpenAI(api_key="private-test", base_url=agent.base_url, max_retries=0) as client:
    client.chat.completions.create(model=agent.model, messages=[
        {key: value for key, value in message.items() if key in {"role", "content", "tool_calls", "tool_call_id"}}
        for message in compressed])
assert compression_seen == [0, 16], compression_seen
# Delivered images return to the stock newest-tool-only aging policy.
assert count_images(compressor_module._strip_historical_media(compressed)) == 8
good_counts = list(compression_seen)
summary_finish_reason = "length"
truncated = compressor_module.ContextCompressor(model="local-qwen", provider="custom", base_url=agent.base_url,
    api_key="private-test", config_context_length=262144, quiet_mode=True, protect_first_n=2,
    protect_last_n=4, tail_mode=profile["compression"].get("tail_mode", "lean"), threshold_tokens_cap=5000)
truncated.compress(history, current_tokens=180000, force=True)
assert truncated.compression_count == 0 and truncated._last_compress_aborted
assert truncated._previous_summary is None
receipts = [json.loads(line) for line in (home / "vision-wire-audit.jsonl").read_text().splitlines()]
request = next(row for row in receipts if row["event"] == "request")
print(json.dumps({"supports_vision": vision, "supports_tool_images": tool_vision,
    "native_multimodal": result["_multimodal"], "seen": seen,
    "ui_has_no_pixels": "data:image" not in ui_json and base64.b64encode(jpeg).decode() not in ui_json,
    "native_compression_count": compressor.compression_count,
    "compression_then_main_image_counts": good_counts,
    "truncated_summary_not_committed": truncated.compression_count == 0 and truncated._previous_summary is None,
    "canonical_image_token_estimate": small_estimate,
    "image_count": request["image_count"], "required_images_present": request["required_images_present"],
    "accepted_for_send": request["accepted_for_send"]}))
"""
    result = subprocess.run(
        [config.hermes_python, "-c", script, str(runtime)],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "PYTHONPATH": str(config.hermes_source),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_HOME": str(home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["supports_vision"]
    assert receipt["supports_tool_images"]
    assert receipt["native_multimodal"]
    assert receipt["ui_has_no_pixels"]
    assert receipt["image_count"] == 1
    assert receipt["required_images_present"]
    assert receipt["accepted_for_send"]
    assert receipt["native_compression_count"] > 0
    assert receipt["truncated_summary_not_committed"]
    assert receipt["compression_then_main_image_counts"] == [0, 16]
    assert 1500 <= receipt["canonical_image_token_estimate"] < 3000
    assert receipt["seen"] == [
        {"image_sha256": "1057be407cc0d2f2cc32591559baa6f615df8a1d0a3f18f1308eccd1ef754cc2", "role": "user"}
    ]
