"""One fresh Hermes inference process; stdin/stdout never carry shell commands."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib
import json
import mimetypes
import os
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_LEGACY_SCHEMA_MARKER = "\n\nReturn only JSON matching this schema, without Markdown fences:\n"


def _structured_request_overrides(request: dict[str, Any]) -> dict[str, Any] | None:
    """Carry the original schema to oMLX's explicit grammar API.

    A parent server may still have the previous backend class cached. Its
    request stores the exact JSON schema at the end of this fixed prompt
    marker. Recover only that metadata until the parent is restarted; never
    infer a schema from model output or modify generated content.
    """
    if "response_schema" in request:
        schema = request["response_schema"]
    else:
        _, marker, serialized = str(request.get("prompt", "")).rpartition(_LEGACY_SCHEMA_MARKER)
        schema = json.loads(serialized) if marker else None
    if schema is None:
        return None
    if not isinstance(schema, dict):
        raise ValueError("Local structured output requires the original JSON schema object")
    schema = copy.deepcopy(schema)
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "arcreel_response", "strict": True, "schema": schema},
        },
        # response_format alone can silently degrade to prompt instructions in
        # oMLX. This explicit request returns HTTP 400 if its grammar compiler
        # is unavailable or cannot compile the original schema.
        "extra_body": {"structured_outputs": {"json": schema}},
    }


def _schema_hash(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def _audit_wire_requests(request: dict[str, Any], audit_path: Path) -> Generator[None, None, None]:
    """Inspect serialized HTTP bodies inside this fresh worker, before sending.

    The receipt contains flags and schema hashes, never prompts, credentials,
    headers or response content. It proves the outbound contract, not that the
    server's logits processor ran. The existing HTTP resource barrier remains
    inside the wrapped send method.
    """
    httpx = importlib.import_module("httpx")
    original_send = httpx.Client.send
    endpoint = urlsplit(str(request["base_url"]).rstrip("/") + "/chat/completions")
    original_overrides = _structured_request_overrides(request)
    expected_schema = original_overrides["extra_body"]["structured_outputs"]["json"] if original_overrides else None
    expected_hash = _schema_hash(expected_schema)
    entries: list[dict[str, Any]] = []
    mutex = threading.Lock()

    def persist() -> None:
        receipt = {
            "schema": "arcreel-hermes-wire-audit/v1",
            "expected_schema_sha256": expected_hash,
            "grammar_enforcement": "not_observable_from_client",
            "requests": entries,
        }
        temporary = audit_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            temporary.chmod(0o600)
            json.dump(receipt, stream, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(audit_path)

    def checked_send(client: Any, outbound: Any, *args: Any, **kwargs: Any) -> Any:
        actual = urlsplit(str(outbound.url))
        endpoint_matches = (
            actual.scheme == endpoint.scheme
            and actual.hostname == endpoint.hostname
            and actual.port == endpoint.port
            and actual.path == endpoint.path
            and not actual.query
            and not actual.fragment
        )
        try:
            body = json.loads(outbound.content)
        except (ValueError, httpx.RequestNotRead):
            body = None
        payload = body if isinstance(body, dict) else {}
        response_format = payload.get("response_format")
        response_format = response_format if isinstance(response_format, dict) else {}
        json_schema = response_format.get("json_schema")
        json_schema = json_schema if isinstance(json_schema, dict) else {}
        structured = payload.get("structured_outputs")
        structured = structured if isinstance(structured, dict) else {}
        response_hash = _schema_hash(json_schema.get("schema"))
        structured_hash = _schema_hash(structured.get("json"))
        entry = {
            "method_is_post": outbound.method == "POST",
            "endpoint_matches_configured": endpoint_matches,
            "endpoint_is_loopback": actual.hostname in {"127.0.0.1", "localhost", "::1"},
            "model_matches_configured": payload.get("model") == request["model"],
            "json_object_body": isinstance(body, dict),
            "response_format_is_json_schema": response_format.get("type") == "json_schema",
            "response_format_strict": json_schema.get("strict") is True,
            "response_format_schema_sha256": response_hash,
            "structured_outputs_schema_sha256": structured_hash,
            "response_format_schema_matches": response_hash == expected_hash if expected_hash else None,
            "structured_outputs_schema_matches": structured_hash == expected_hash if expected_hash else None,
        }
        accepted = all(
            entry[key]
            for key in (
                "method_is_post",
                "endpoint_matches_configured",
                "endpoint_is_loopback",
                "model_matches_configured",
                "json_object_body",
            )
        )
        if expected_hash:
            accepted = accepted and all(
                entry[key]
                for key in (
                    "response_format_is_json_schema",
                    "response_format_strict",
                    "response_format_schema_matches",
                    "structured_outputs_schema_matches",
                )
            )
        entry["accepted_for_send"] = accepted
        with mutex:
            entries.append(entry)
            persist()
        if not accepted:
            raise ValueError("Local Hermes outbound inference contract was not preserved")
        # httpx follows redirects inside send, bypassing this route check.
        kwargs["follow_redirects"] = False
        return original_send(client, outbound, *args, **kwargs)

    persist()
    httpx.Client.send = checked_send
    try:
        yield
        if (expected_hash and not entries) or any(not entry["accepted_for_send"] for entry in entries):
            raise ValueError("Local Hermes structured request has no accepted outbound contract")
    finally:
        httpx.Client.send = original_send


def _image_parts(images: list[str]) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    for value in images:
        if value.startswith("data:image/"):
            uri = value
        else:
            path = Path(value)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Image input must be an existing local file")
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("Unsupported local image format")
            uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


def _isolate_env_loader() -> None:
    """Hermes source-tree .env and external secret stores are outside this job."""
    module = importlib.import_module("hermes_cli.env_loader")
    original = module.load_hermes_dotenv

    def local_dotenv(**kwargs: Any) -> list[Path]:
        del kwargs
        return original(hermes_home=os.environ["HERMES_HOME"], project_env=None, load_external_secrets=False)

    module.__dict__["load_hermes_dotenv"] = local_dotenv


def _run(request: dict[str, Any], *, audit_path: Path) -> dict[str, Any]:
    source = Path(str(request["source"]))
    endpoint = str(request["base_url"]).rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Hermes inference endpoint must be loopback")
    sys.path.insert(0, str(source))
    _isolate_env_loader()
    provider_module = importlib.import_module("hermes_cli.runtime_provider")
    agent_module = importlib.import_module("run_agent")
    runtime = provider_module.resolve_runtime_provider(requested="omlx", target_model=str(request["model"]))
    if str(runtime.get("base_url", "")).rstrip("/") != endpoint:
        raise ValueError("Hermes resolved an unexpected inference endpoint")
    api_key = runtime.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Hermes did not resolve the configured local endpoint credential")
    overrides = _structured_request_overrides(request)
    agent = agent_module.AIAgent(
        model=str(request["model"]),
        provider=runtime.get("provider"),
        requested_provider="omlx",
        api_mode="chat_completions",
        base_url=endpoint,
        api_key=api_key,
        max_tokens=request.get("max_output_tokens"),
        request_overrides=overrides,
        enabled_toolsets=["context_engine"],
        max_iterations=2,
        fallback_model=None,
        quiet_mode=True,
        tool_progress_mode="off",
        save_trajectories=False,
        ephemeral_system_prompt=request.get("system_prompt"),
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        platform="arcreel-text",
    )
    # The text backend cannot call project tools, native commands or other models.
    agent.tools = []
    agent.valid_tool_names = set()
    agent.enabled_toolsets = ["context_engine"]
    agent.stream_delta_callback = None
    agent.reasoning_callback = None
    try:
        images = request.get("images") or []
        prompt = str(request["prompt"])
        message = [{"type": "text", "text": prompt}, *_image_parts(images)] if images else prompt
        with _audit_wire_requests(request, audit_path):
            result = agent.run_conversation(user_message=message, conversation_history=[])
        messages = result.get("messages") or []
        finish_reason = next(
            (item.get("finish_reason") for item in reversed(messages) if item.get("role") == "assistant"), None
        )
        return {
            "text": result.get("final_response") or "",
            "provider": "omlx",
            "model": agent.model,
            "base_url": endpoint,
            "completed": bool(result.get("completed")),
            "failed": bool(result.get("failed")),
            "partial": bool(result.get("partial")),
            "interrupted": bool(result.get("interrupted")),
            "finish_reason": finish_reason,
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
        }
    finally:
        agent.close()


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    request_path = Path(sys.argv[1])
    result_path = request_path.with_name("result.json")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(request, audit_path=request_path.with_name("wire-audit.json"))
        serialized = json.dumps(result, ensure_ascii=False)
        result_path.write_text(serialized + "\n", encoding="utf-8")
        result_path.chmod(0o600)
        print(serialized)
        return 0
    except Exception:
        result_path.write_text('{"failed":true,"completed":false}\n', encoding="utf-8")
        result_path.chmod(0o600)
        print("Local Hermes worker failed; no fallback was attempted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
