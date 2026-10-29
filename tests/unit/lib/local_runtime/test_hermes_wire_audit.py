"""Audit actual serialized requests without contacting an inference service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from lib.local_runtime.hermes_text_worker import _audit_wire_requests, _structured_request_overrides


def _request() -> dict[str, Any]:
    return {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "configured-local-model",
        "response_schema": {
            "type": "object",
            "properties": {"novel_text": {"type": "string"}},
            "required": ["novel_text"],
            "additionalProperties": False,
        },
        "prompt": "PRIVATE_PROMPT_DO_NOT_LOG",
    }


def _body(request: dict[str, Any]) -> dict[str, Any]:
    overrides = _structured_request_overrides(request)
    assert overrides is not None
    return {
        "model": request["model"],
        "messages": [{"role": "user", "content": request["prompt"]}],
        "response_format": overrides["response_format"],
        **overrides["extra_body"],
    }


def test_wire_audit_records_serialized_contract_without_private_content(tmp_path: Path) -> None:
    request = _request()
    audit_path = tmp_path / "wire-audit.json"
    received: list[httpx.Request] = []

    def receive(outbound: httpx.Request) -> httpx.Response:
        received.append(outbound)
        return httpx.Response(200, json={"private_response": "RESPONSE_DO_NOT_LOG"})

    before = httpx.Client.send
    with (
        httpx.Client(transport=httpx.MockTransport(receive)) as client,
        _audit_wire_requests(request, audit_path),
    ):
        response = client.post(
            request["base_url"] + "/chat/completions",
            json=_body(request),
            headers={"Authorization": "Bearer KEY_DO_NOT_LOG", "X-Private": "HEADER_DO_NOT_LOG"},
        )
    assert response.status_code == 200
    assert len(received) == 1
    assert httpx.Client.send is before
    serialized = audit_path.read_text()
    assert all(
        secret not in serialized
        for secret in (
            request["prompt"],
            "KEY_DO_NOT_LOG",
            "HEADER_DO_NOT_LOG",
            "RESPONSE_DO_NOT_LOG",
            "novel_text",
        )
    )
    receipt = json.loads(serialized)
    expected_hash = hashlib.sha256(
        json.dumps(
            request["response_schema"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert receipt["expected_schema_sha256"] == expected_hash
    assert receipt["grammar_enforcement"] == "not_observable_from_client"
    assert receipt["requests"][0]["accepted_for_send"] is True
    assert receipt["requests"][0]["response_format_schema_sha256"] == expected_hash
    assert receipt["requests"][0]["structured_outputs_schema_sha256"] == expected_hash
    assert audit_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "change",
    [
        "missing-response-format",
        "missing-structured-outputs",
        "changed-response-schema",
        "mutated-nested-schema",
        "changed-structured-schema",
        "non-strict",
        "wrong-model",
        "wrong-endpoint",
        "remote-endpoint",
        "invalid-json",
    ],
)
def test_lost_wire_constraints_are_rejected_before_transport(tmp_path: Path, change: str) -> None:
    request = _request()
    body = _body(request)
    url = request["base_url"] + "/chat/completions"
    if change == "missing-response-format":
        del body["response_format"]
    elif change == "missing-structured-outputs":
        del body["structured_outputs"]
    elif change == "changed-response-schema":
        body["response_format"]["json_schema"]["schema"] = {"type": "string"}
    elif change == "changed-structured-schema":
        body["structured_outputs"]["json"] = {"type": "string"}
    elif change == "mutated-nested-schema":
        body["structured_outputs"]["json"]["properties"].clear()
    elif change == "non-strict":
        body["response_format"]["json_schema"]["strict"] = False
    elif change == "wrong-model":
        body["model"] = "another-model"
    elif change == "wrong-endpoint":
        url = "http://127.0.0.1:8001/v1/chat/completions"
    elif change == "remote-endpoint":
        url = "https://example.com/v1/chat/completions"
    sends: list[httpx.Request] = []
    audit_path = tmp_path / "wire-audit.json"

    def receive(outbound: httpx.Request) -> httpx.Response:
        sends.append(outbound)
        return httpx.Response(200)

    before = httpx.Client.send
    options = {"content": b"not-json"} if change == "invalid-json" else {"json": body}
    with (
        httpx.Client(transport=httpx.MockTransport(receive)) as client,
        pytest.raises(ValueError, match="outbound inference contract"),
        _audit_wire_requests(request, audit_path),
    ):
        client.post(url, **options)
    assert sends == []
    assert "novel_text" in request["response_schema"]["properties"]
    assert httpx.Client.send is before
    assert json.loads(audit_path.read_text())["requests"][0]["accepted_for_send"] is False


def test_structured_success_without_an_observed_request_is_rejected(tmp_path: Path) -> None:
    request = _request()
    before = httpx.Client.send
    audit_path = tmp_path / "wire-audit.json"
    with (
        pytest.raises(ValueError, match="no accepted outbound contract"),
        _audit_wire_requests(request, audit_path),
    ):
        pass
    assert json.loads(audit_path.read_text())["requests"] == []
    assert httpx.Client.send is before


def test_wire_audit_preserves_existing_resource_send_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request()
    original = httpx.Client.send
    guard_calls: list[str] = []

    def resource_guard(client: httpx.Client, outbound: httpx.Request, **kwargs: Any) -> httpx.Response:
        guard_calls.append("existing-resource-guard")
        return original(client, outbound, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", resource_guard)
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client,
        _audit_wire_requests(request, tmp_path / "wire-audit.json"),
    ):
        client.post(request["base_url"] + "/chat/completions", json=_body(request))
    assert guard_calls == ["existing-resource-guard"]
    assert httpx.Client.send is resource_guard


@pytest.mark.parametrize(
    "redirect_to",
    [
        "http://127.0.0.1:8000/v1/redirected",
        "https://example.invalid/v1/chat/completions",
    ],
)
def test_wire_audit_never_follows_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redirect_to: str,
) -> None:
    request = _request()
    original = httpx.Client.send
    sent_urls: list[str] = []
    guard_options: list[bool] = []

    def receive(outbound: httpx.Request) -> httpx.Response:
        sent_urls.append(str(outbound.url))
        return httpx.Response(307, headers={"Location": redirect_to})

    def resource_guard(client: httpx.Client, outbound: httpx.Request, **kwargs: Any) -> httpx.Response:
        guard_options.append(kwargs["follow_redirects"])
        return original(client, outbound, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", resource_guard)
    audit_path = tmp_path / "wire-audit.json"
    url = request["base_url"] + "/chat/completions"
    with (
        httpx.Client(transport=httpx.MockTransport(receive), follow_redirects=True) as client,
        _audit_wire_requests(request, audit_path),
    ):
        outbound = client.build_request("POST", url, json=_body(request))
        response = client.send(outbound, follow_redirects=True)
    assert response.status_code == 307
    assert response.history == []
    assert sent_urls == [url]
    assert guard_options == [False]
    assert httpx.Client.send is resource_guard
    assert len(json.loads(audit_path.read_text())["requests"]) == 1
