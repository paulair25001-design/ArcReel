"""MCP pixels reach serialized local requests without entering UI/audit bodies."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from PIL import Image

from lib.infra.image_utils import image_pixel_sha256
from server.agent_runtime.hermes_vision_bridge import VisionTransport, install_acp_text_projection

_buffer = io.BytesIO()
Image.new("RGB", (2, 2), (255, 0, 0)).save(_buffer, format="JPEG", quality=85)
_IMAGE = _buffer.getvalue()
_PIXEL_SHA = image_pixel_sha256(_IMAGE)


def _mcp(
    transport: VisionTransport,
    *,
    attached: bool = True,
    row_patch: dict[str, Any] | None = None,
    image_count: int = 1,
) -> Any:
    module = SimpleNamespace()

    def factory(server: str, tool: str, timeout: float):
        def handler(*args: Any, **kwargs: Any):
            async def rpc():
                if attached:
                    for _ in range(image_count):
                        module._cache_mcp_image_block(
                            SimpleNamespace(type="image", data=base64.b64encode(_IMAGE).decode(), mimeType="image/jpeg")
                        )
                row = {
                    "id": "S01",
                    "version": 1,
                    "image": 1,
                    "source_sha256": hashlib.sha256(_IMAGE).hexdigest(),
                    "source_pixel_sha256": _PIXEL_SHA,
                    "wire_image_sha256": hashlib.sha256(_IMAGE).hexdigest(),
                    **(row_patch or {}),
                }
                return json.dumps(
                    {
                        "result": "MEDIA:/untrusted/text/is/not/read.png",
                        "structuredContent": {
                            "inspect_images": {
                                "resource_type": "storyboard",
                                "images": [row],
                            }
                        }
                        if tool == "inspect_images"
                        else {
                            "inspect_video_units": {
                                "units": [
                                    {
                                        "unit_id": "S01",
                                        "version": 1,
                                        "status": "ok",
                                        "sheets": [{"image": i + 1} for i in range(image_count)],
                                    }
                                ]
                            }
                        },
                    }
                )

            return module._run_on_mcp_loop(rpc)

        return handler

    module._make_tool_handler = factory
    module._cache_mcp_image_block = lambda block: "MEDIA:/unused/cache.png"
    # A fresh context simulates the dedicated MCP event-loop thread.
    module._run_on_mcp_loop = lambda factory: contextvars.Context().run(asyncio.run, factory())
    transport.install_mcp(module)
    return module


def _receipt(home: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (home / "vision-wire-audit.jsonl").read_text().splitlines()]


def _http_module(guard_calls: list[Any] | None = None) -> Any:
    class LocalClient(httpx.Client):
        def send(self, request: Any, *args: Any, **kwargs: Any):
            if guard_calls is not None:
                guard_calls.append(kwargs.get("follow_redirects"))
            return super().send(request, *args, **kwargs)

    return SimpleNamespace(Client=LocalClient, RequestNotRead=httpx.RequestNotRead)


def test_actual_mcp_pixels_reach_local_wire_and_audit_without_content_leak(
    tmp_path: Path, respx_mock: respx.MockRouter
):
    guard_calls = []
    http = _http_module(guard_calls)
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="local-qwen")
    transport.begin_turn("session-1")
    result = _mcp(transport)._make_tool_handler("arcreel", "inspect_images", 30)({})
    assert result["_multimodal"] is True
    transport.install_http(http)
    route = respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").respond(200, json={"ok": True})
    with http.Client(headers={"Authorization": "Bearer never-record-this"}) as client:
        client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "local-qwen",
                "messages": [{"role": "tool", "content": result["content"]}],
            },
        )
        # Already delivered pixels may be compressed from later requests.
        client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "continue"}],
            },
        )
    assert guard_calls == [False, False]
    sent = json.loads(route.calls[0].request.content)
    assert base64.b64decode(sent["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1]) == _IMAGE
    receipt = _receipt(tmp_path)
    request = next(row for row in receipt if row["event"] == "request")
    assert request["image_count"] == 1
    assert request["required_images_present"] is True
    assert request["images"] == [
        {"role": "tool", "mime_type": "image/jpeg", "bytes": len(_IMAGE), "sha256": hashlib.sha256(_IMAGE).hexdigest()}
    ]
    assert request["body_sha256"] == hashlib.sha256(route.calls[0].request.content).hexdigest()
    observed = [row for row in receipt if row["event"] == "observed_images"]
    assert len(observed) == 1
    assert observed[0]["protocol"] == "arcreel-mcp-vision/v2"
    assert observed[0]["rows"] == [
        {
            "resource_type": "storyboard",
            "id": "S01",
            "version": 1,
            "source_sha256": hashlib.sha256(_IMAGE).hexdigest(),
            "source_pixel_sha256": _PIXEL_SHA,
            "wire_image_sha256": hashlib.sha256(_IMAGE).hexdigest(),
        }
    ]
    response = next(row for row in receipt if row["event"] == "response")
    for field in ("transport_id", "session_id", "request_id", "model", "body_sha256"):
        assert request[field] == response[field] == observed[0][field]
    cold = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="local-qwen")
    assert cold.transport_id != observed[0]["transport_id"]
    assert [row for row in _receipt(tmp_path) if row["event"] == "observed_images"] == observed
    text = (tmp_path / "vision-wire-audit.jsonl").read_text()
    assert "never-record-this" not in text
    assert base64.b64encode(_IMAGE).decode() not in text
    assert (tmp_path / "vision-wire-audit.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["stripped_image", "external_endpoint", "wrong_model"])
def test_missing_pixels_or_wrong_route_fail_before_transport(
    tmp_path: Path, respx_mock: respx.MockRouter, failure: str
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="local-qwen")
    result = _mcp(transport)._make_tool_handler("arcreel", "inspect_images", 30)({})
    transport.install_http(http)
    endpoint = (
        "https://outside.invalid/v1/chat/completions"
        if failure == "external_endpoint"
        else ("http://127.0.0.1:8000/v1/chat/completions")
    )
    with http.Client() as client, pytest.raises(ValueError, match="contract was not preserved"):
        client.post(
            endpoint,
            json={
                "model": "wrong" if failure == "wrong_model" else "local-qwen",
                "messages": [
                    {"role": "tool", "content": "MEDIA:only-text" if failure == "stripped_image" else result["content"]}
                ],
            },
        )
    assert len(respx_mock.calls) == 0
    assert _receipt(tmp_path)[-1]["accepted_for_send"] is False


def test_declared_images_cannot_silently_become_media_text(tmp_path: Path):
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    with pytest.raises(ValueError, match="not captured"):
        _mcp(transport, attached=False)._make_tool_handler("arcreel", "inspect_images", 30)({})


def test_delegated_review_pixels_remain_owned_by_the_child_context(tmp_path: Path, respx_mock: respx.MockRouter):
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    transport.begin_turn("parent")
    mcp = _mcp(transport)
    http = _http_module()
    transport.install_http(http)
    respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").respond(200, json={"ok": True})
    with transport.agent_scope("review-1", parent_session_id="parent"):
        images = mcp._make_tool_handler("arcreel", "inspect_images", 30)({})
        reviewer_context = contextvars.copy_context()
    with http.Client() as client:

        def request(content):
            return client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": content}]},
            )

        assert request("parent reads only a report").status_code == 200
        with transport.agent_scope("review-2", parent_session_id="parent"):
            assert request("independent reviewer").status_code == 200
        with pytest.raises(ValueError, match="contract was not preserved"):
            reviewer_context.run(request, "review-1 cannot omit its pending images")
        assert reviewer_context.run(request, images["content"]).status_code == 200
    rows = _receipt(tmp_path)
    seen = [row for row in rows if row["event"] == "observed_images"]
    assert len(seen) == 1
    assert seen[0]["session_id"] == "review-1"
    assert seen[0]["parent_session_id"] == "parent"
    assert seen[0]["rows"][0]["source_pixel_sha256"] == _PIXEL_SHA
    assert {row["session_id"] for row in rows if row["event"] == "request"} == {"parent", "review-1", "review-2"}


def test_acp_projects_text_only_while_inference_retains_pixels():
    events = SimpleNamespace(build_tool_complete=lambda *args, **kwargs: kwargs["result"])
    server = SimpleNamespace(make_step_cb=lambda: None)
    install_acp_text_projection(server, events)
    result = {"_multimodal": True, "text_summary": "one image attached", "content": [{"data": "private-pixels"}]}
    assert events.build_tool_complete("call-1", "inspect_images", result=result) == "one image attached"
    assert server.build_tool_complete("call-1", "inspect_images", result=result) == "one image attached"
    assert result["content"] == [{"data": "private-pixels"}]


def test_redirects_and_image_stripping_recovery_cannot_bypass_the_wire_contract(
    tmp_path: Path, respx_mock: respx.MockRouter
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    result = _mcp(transport)._make_tool_handler("arcreel", "inspect_images", 30)({})
    transport.install_http(http)
    route = respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").respond(
        307, headers={"Location": "https://outside.invalid/v1/chat/completions"}
    )
    with http.Client(follow_redirects=True) as client:
        response = client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "tool", "content": result["content"]}]},
            follow_redirects=True,
        )
        assert response.status_code == 307
        assert not response.history
        # A failed/redirected request has not consumed the required pixels.
        with pytest.raises(ValueError, match="contract was not preserved"):
            client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "tool", "content": "text fallback"}]},
            )
    assert route.call_count == 1
    assert len(respx_mock.calls) == 1


def test_auxiliary_success_does_not_consume_pending_main_images_or_leak_scope(
    tmp_path: Path, respx_mock: respx.MockRouter
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    transport.begin_turn("same-session")
    result = _mcp(transport)._make_tool_handler("arcreel", "inspect_images", 30)({})
    transport.install_http(http)
    route = respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").respond(200, json={"ok": True})
    with http.Client() as client:
        auxiliary = SimpleNamespace(
            call_llm=lambda **kwargs: client.post(
                "http://127.0.0.1:8000/v1/chat/completions", json={"model": "qwen", "messages": kwargs["messages"]}
            )
        )
        transport.install_auxiliary(auxiliary)
        auxiliary.call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])
        assert not [row for row in _receipt(tmp_path) if row["event"] == "observed_images"]
        transport.begin_turn("same-session")
        with pytest.raises(ValueError, match="contract was not preserved"):
            client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": "text is not observation"}]},
            )
        client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "tool", "content": result["content"]}]},
        )
    assert route.call_count == 2
    requests = [row for row in _receipt(tmp_path) if row["event"] == "request"]
    assert [row["request_kind"] for row in requests] == ["auxiliary", "main", "main"]
    assert [row["accepted_for_send"] for row in requests] == [True, False, True]


@pytest.mark.parametrize(
    ("endpoint", "model"),
    [
        ("https://outside.invalid/v1/chat/completions", "qwen"),
        ("http://127.0.0.1:8000/v1/chat/completions", "different-model"),
    ],
)
def test_native_auxiliary_context_keeps_route_and_model_pinned(
    tmp_path: Path, respx_mock: respx.MockRouter, endpoint: str, model: str
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    transport.install_http(http)
    with http.Client() as client:
        auxiliary = SimpleNamespace(call_llm=lambda: client.post(endpoint, json={"model": model, "messages": []}))
        transport.install_auxiliary(auxiliary)
        with pytest.raises(ValueError, match="contract was not preserved"):
            auxiliary.call_llm()
    assert len(respx_mock.calls) == 0
    assert _receipt(tmp_path)[-1]["request_kind"] == "auxiliary"


def test_same_version_reinspection_after_failed_main_requires_one_real_observation(
    tmp_path: Path, respx_mock: respx.MockRouter
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    transport.begin_turn("same-session")
    mcp = _mcp(transport)
    first = mcp._make_tool_handler("arcreel", "inspect_images", 30)({})
    transport.install_http(http)
    route = respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(500, json={"error": "temporary"}), httpx.Response(200, json={"ok": True})]
    )
    with http.Client() as client:
        failed = client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "tool", "content": first["content"]}]},
        )
        assert failed.status_code == 500
        assert not [row for row in _receipt(tmp_path) if row["event"] == "observed_images"]
        transport.begin_turn("same-session")
        with pytest.raises(ValueError, match="contract was not preserved"):
            client.post("http://127.0.0.1:8000/v1/chat/completions", json={"model": "qwen", "messages": []})
        newest = mcp._make_tool_handler("arcreel", "inspect_images", 30)({})
        client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "tool", "content": newest["content"]}]},
        )
    assert route.call_count == 2
    requests = [row for row in _receipt(tmp_path) if row["event"] == "request"]
    assert [row["pending_observation_count"] for row in requests] == [1, 1, 1]
    assert [row["accepted_for_send"] for row in requests] == [True, False, True]


@pytest.mark.parametrize(
    "row_patch",
    [
        {"source_sha256": None},
        {"source_pixel_sha256": "missing"},
        {"wire_image_sha256": "0" * 64},
        {"image": 2},
    ],
)
def test_invalid_source_provenance_cannot_create_observation(tmp_path: Path, row_patch: dict[str, Any]):
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    with pytest.raises(ValueError, match=r"provenance|JPEG|order"):
        _mcp(transport, row_patch=row_patch)._make_tool_handler("arcreel", "inspect_images", 30)({})
    assert not (tmp_path / "vision-wire-audit.jsonl").exists()


def test_native_video_contact_sheets_reach_model_without_creating_image_version_proof(
    tmp_path: Path, respx_mock: respx.MockRouter
):
    http = _http_module()
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    result = _mcp(transport)._make_tool_handler("arcreel", "inspect_video_units", 30)({})
    transport.install_http(http)
    route = respx_mock.post("http://127.0.0.1:8000/v1/chat/completions").respond(200, json={"ok": True})
    with http.Client() as client:
        client.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "tool", "content": result["content"]}]},
        )
    assert route.call_count == 1
    request = next(row for row in _receipt(tmp_path) if row["event"] == "request")
    assert request["image_count"] == 1
    assert request["required_images_present"] is True
    assert not [row for row in _receipt(tmp_path) if row["event"] == "observed_images"]
    observed = [row for row in _receipt(tmp_path) if row["event"] == "observed_video_units"]
    assert observed[0]["rows"] == [
        {"unit_id": "S01", "version": 1, "wire_image_sha256s": [hashlib.sha256(_IMAGE).hexdigest()]}
    ]


def test_visual_tool_image_limit_is_explicit_and_never_drops_frames(tmp_path: Path):
    transport = VisionTransport(tmp_path, base_url="http://127.0.0.1:8000/v1", model="qwen")
    with pytest.raises(ValueError, match=r"96 image blocks.*smaller batches"):
        _mcp(transport, image_count=97)._make_tool_handler("arcreel", "inspect_video_units", 30)({})
