"""Child-process MCP image transport and private, content-free HTTP receipts."""

from __future__ import annotations

import base64
import binascii
import contextvars
import functools
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROTOCOL = "arcreel-mcp-vision/v2"
TRUSTED_VISION_NOTICE = (
    "\n\n[ArcReel transport: arcreel-mcp-vision/v2] "
    "inspect_images now supplies actual native image content. Historical MEDIA:path text alone was not visual input "
    "and does not establish that an image was seen. After recovering a session containing those old placeholders, "
    "use inspect_images again on the existing first-frame A and tail-candidate B versions before selecting frames "
    "or generating videos. Preserve existing images; do not regenerate them merely to recover visual observation. "
    "Only describe an image as observed when its actual image content was supplied."
)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGES = 96
_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_RESOURCE_TYPES = {"character", "scene", "prop", "product", "storyboard", "character_derivative"}


class _SessionPixels:
    def __init__(self, session_id: str | None = None, parent_session_id: str | None = None) -> None:
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.transport_id = uuid.uuid4().hex
        self.expected: Counter[str] = Counter()
        self.pending: dict[tuple[Any, ...], dict[str, Any] | None] = {}
        self.observed_images: list[dict[str, Any]] = []
        self.observed_video_units: list[dict[str, Any]] = []
        self.pending_video_units: list[dict[str, Any]] = []
        self.turn = 0
        self.request = 0


def _generation_parameters(payload: Any) -> dict[str, Any]:
    """Expose actual non-sensitive generation knobs without changing the request."""
    if not isinstance(payload, dict):
        return {}
    result = {
        key: payload[key]
        for key in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "enable_thinking")
        if key in payload and type(payload[key]) in (int, float, bool)
    }
    levels = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    if isinstance(payload.get("reasoning_effort"), str) and payload["reasoning_effort"] in levels:
        result["reasoning_effort"] = payload["reasoning_effort"]
    template = payload.get("chat_template_kwargs")
    if isinstance(template, dict):
        result["chat_template_kwargs"] = {
            key: template[key] for key in ("enable_thinking", "preserve_thinking") if type(template.get(key)) is bool
        }
        if isinstance(template.get("reasoning_effort"), str) and template["reasoning_effort"] in levels:
            result["chat_template_kwargs"]["reasoning_effort"] = template["reasoning_effort"]
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        selected = {"enabled": reasoning["enabled"]} if type(reasoning.get("enabled")) is bool else {}
        if isinstance(reasoning.get("effort"), str) and reasoning["effort"] in levels:
            selected["effort"] = reasoning["effort"]
        if selected:
            result["reasoning"] = selected
    return result


def _image_part(data: object, mime: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(data, str)
        or not isinstance(mime, str)
        or mime not in _MIMES
        or len(data) > (_MAX_IMAGE_BYTES * 4 // 3 + 8)
    ):
        raise ValueError("ArcReel MCP image has an unsupported MIME type or size")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ArcReel MCP image has invalid base64") from exc
    if not raw or len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError("ArcReel MCP image is empty or oversized")
    if mime == "image/jpeg" and not raw.startswith(b"\xff\xd8\xff"):
        raise ValueError("ArcReel MCP image is not JPEG data")
    facts = {"mime_type": mime, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}, facts


def _observation_rows(payload: Any, images: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Bind same-response source provenance to the actual typed JPEG blocks."""
    inspected = payload.get("structuredContent", {}).get("inspect_images", {})
    resource_type = inspected.get("resource_type")
    if resource_type not in _RESOURCE_TYPES:
        raise ValueError("ArcReel image observation has no valid resource type")
    attached = [row for row in inspected.get("images", []) if isinstance(row, dict) and row.get("image") is not None]
    if len(attached) != len(images):
        raise ValueError("ArcReel MCP image count does not match the native result")
    result = []
    for index, (row, (_, facts)) in enumerate(zip(attached, images, strict=True), 1):
        if type(row.get("image")) is not int or row["image"] != index:
            raise ValueError("ArcReel image observation has inconsistent attachment order")
        identity = row.get("id")
        version = row.get("version")
        if (
            not isinstance(identity, str)
            or not identity
            or not (version is None or (type(version) is int and version >= 1))
        ):
            raise ValueError("ArcReel image observation has no valid resource identity")
        for field in ("source_sha256", "source_pixel_sha256", "wire_image_sha256"):
            if not isinstance(row.get(field), str) or re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                raise ValueError("ArcReel image observation is missing source or pixel provenance")
        if facts["mime_type"] != "image/jpeg" or row["wire_image_sha256"] != facts["sha256"]:
            raise ValueError("ArcReel image observation does not match the typed JPEG bytes")
        result.append(
            {
                "resource_type": resource_type,
                "id": identity,
                "version": version,
                **{field: row[field] for field in ("source_sha256", "source_pixel_sha256", "wire_image_sha256")},
            }
        )
    return result


def _video_observations(payload: Any, images: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    """Bind native unit versions to the contact sheets attached in that result."""
    result = []
    used: set[int] = set()
    for unit in payload.get("structuredContent", {}).get("inspect_video_units", {}).get("units", []):
        if unit.get("status") != "ok":
            continue
        identity, version = unit.get("unit_id"), unit.get("version")
        sheets = unit.get("sheets")
        if not isinstance(identity, str) or not identity or type(version) is not int or version < 1:
            raise ValueError("ArcReel video inspection has no valid unit version")
        if not isinstance(sheets, list) or not sheets:
            raise ValueError("ArcReel video inspection has no contact sheets")
        hashes = []
        for sheet in sheets:
            number = sheet.get("image")
            if type(number) is not int or number < 1 or number > len(images) or number in used:
                raise ValueError("ArcReel video inspection has inconsistent attachment order")
            used.add(number)
            hashes.append(images[number - 1][1]["sha256"])
        result.append({"unit_id": identity, "version": version, "wire_image_sha256s": hashes})
    if used != set(range(1, len(images) + 1)):
        raise ValueError("ArcReel video contact sheets do not match the native unit result")
    return result


class VisionTransport:
    """Retain actual MCP bytes until the next pinned local model request."""

    def __init__(self, home: Path, *, base_url: str, model: str, writer_tools: tuple[str, ...] = ()) -> None:
        self.home = home
        self.endpoint = urlsplit(base_url.rstrip("/") + "/chat/completions")
        if self.endpoint.scheme not in {"http", "https"} or self.endpoint.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("ArcReel vision transport requires a loopback endpoint")
        self.model = model
        self.writer_tools = frozenset(writer_tools)
        self._capture: contextvars.ContextVar[list[tuple[dict[str, Any], dict[str, Any]]] | None] = (
            contextvars.ContextVar("arcreel_mcp_image_capture", default=None)
        )
        self._mutex = threading.Lock()
        self._request_kind: contextvars.ContextVar[str] = contextvars.ContextVar(
            "arcreel_native_request_kind", default="main"
        )
        self._root_pixels = _SessionPixels()
        self._agent_pixels: contextvars.ContextVar[_SessionPixels | None] = contextvars.ContextVar(
            "arcreel_agent_pixels", default=None
        )

    @property
    def _pixels(self) -> _SessionPixels:
        return self._agent_pixels.get() or self._root_pixels

    @property
    def transport_id(self) -> str:
        return self._pixels.transport_id

    @property
    def _expected(self) -> Counter[str]:
        return self._pixels.expected

    @_expected.setter
    def _expected(self, value: Counter[str]) -> None:
        self._pixels.expected = value

    @property
    def _pending(self) -> dict[tuple[Any, ...], dict[str, Any] | None]:
        return self._pixels.pending

    @property
    def _session_id(self) -> str | None:
        return self._pixels.session_id

    @_session_id.setter
    def _session_id(self, value: str) -> None:
        self._pixels.session_id = value

    @property
    def _turn(self) -> int:
        return self._pixels.turn

    @_turn.setter
    def _turn(self, value: int) -> None:
        self._pixels.turn = value

    @property
    def _request(self) -> int:
        return self._pixels.request

    @_request.setter
    def _request(self, value: int) -> None:
        self._pixels.request = value

    @contextmanager
    def agent_scope(self, session_id: str, *, parent_session_id: str):
        """Keep delegated reviewers' pixels and receipts in their own context."""
        token = self._agent_pixels.set(_SessionPixels(session_id=session_id, parent_session_id=parent_session_id))
        try:
            self.begin_turn(session_id)
            with self._mutex:
                self._record("agent_started")
            yield self._pixels
        finally:
            self._agent_pixels.reset(token)

    def _record(self, event: str, **facts: Any) -> None:
        # Called under the mutex from model/tool worker threads, never the ACP event loop.
        path = self.home / "vision-wire-audit.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "protocol": PROTOCOL,
                        "transport_id": self.transport_id,
                        "event": event,
                        "timestamp": time.time(),
                        "session_id": self._session_id,
                        "parent_session_id": self._pixels.parent_session_id,
                        "turn": self._turn,
                        **facts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def begin_turn(self, session_id: str) -> None:
        with self._mutex:
            if session_id != self._session_id:
                self._expected.clear()
                self._pending.clear()
            self._session_id = session_id
            self._turn += 1

    def _native_result(self, tool_name: str, arguments: Any, payload: Any) -> None:
        """Record real native mutations and resource handoff, without grading media."""
        if not isinstance(payload, dict) or "error" in payload or payload.get("is_error") is True:
            return
        content = payload.get("structuredContent")
        if not isinstance(content, dict) or "problem" in content or not self._session_id:
            return
        project = arguments.get("project") if isinstance(arguments, dict) else None
        if not isinstance(project, str) or not project:
            return
        if tool_name == "finish_local_session":
            local = content.get("local_session")
            restoration = local.get("restoration") if isinstance(local, dict) else None
            if (
                not isinstance(local, dict)
                or local.get("project") != project
                or not isinstance(restoration, dict)
                or restoration.get("state") != "restored"
                or restoration.get("original_service_restored") is not True
            ):
                return
            event, source = "native_finalization", local
        elif tool_name in self.writer_tools or tool_name.startswith("generate_"):
            event, source = "native_write_success", content
        else:
            return
        with self._mutex:
            self._record(
                event,
                tool_name=tool_name,
                project_name=project,
                source_sha256=hashlib.sha256(
                    json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                **({"original_service_restored": True} if event == "native_finalization" else {}),
            )

    def install_mcp(self, module: Any) -> None:
        """Preserve native MCP discovery, trust gates, timeouts and interrupts."""
        original_factory = module._make_tool_handler
        original_cache = module._cache_mcp_image_block
        original_run = module._run_on_mcp_loop

        def cache_image(block: Any) -> str:
            collector = self._capture.get()
            if collector is not None and getattr(block, "type", None) == "image":
                mime = getattr(block, "mime_type", None) or getattr(block, "mimeType", None)
                if len(collector) >= _MAX_IMAGES:
                    raise ValueError(
                        "ArcReel MCP visual input exceeds 96 image blocks; split the inspection into smaller batches"
                    )
                collector.append(_image_part(getattr(block, "data", None), mime))
            return original_cache(block)

        def run_on_loop(coro_or_factory: Any, *args: Any, **kwargs: Any) -> Any:
            collector = self._capture.get()
            if collector is None:
                return original_run(coro_or_factory, *args, **kwargs)

            async def scoped() -> Any:
                token = self._capture.set(collector)
                try:
                    coro: Any = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
                    return await coro
                finally:
                    self._capture.reset(token)

            return original_run(scoped, *args, **kwargs)

        def factory(server_name: str, tool_name: str, tool_timeout: float) -> Any:
            original = original_factory(server_name, tool_name, tool_timeout)
            if server_name != "arcreel":
                return original

            def inspect(*args: Any, **kwargs: Any) -> Any:
                images: list[tuple[dict[str, Any], dict[str, Any]]] = []
                token = self._capture.set(images)
                try:
                    result = original(*args, **kwargs)
                finally:
                    self._capture.reset(token)
                payload = json.loads(result) if isinstance(result, str) else {}
                self._native_result(tool_name, args[0] if args else kwargs, payload)
                if not images:
                    # A successful declaration of attached images must never become text-only.
                    rows = payload.get("structuredContent", {}).get("inspect_images", {}).get("images", [])
                    if tool_name == "inspect_images" and any(row.get("image") is not None for row in rows):
                        raise ValueError("ArcReel MCP image content was not captured")
                    return result
                payload = json.loads(result)
                if "error" in payload:
                    raise ValueError("ArcReel MCP returned image content in an error result")
                rows = _observation_rows(payload, images) if tool_name == "inspect_images" else []
                video_rows = _video_observations(payload, images) if tool_name == "inspect_video_units" else []
                with self._mutex:
                    self._pixels.pending_video_units.extend(video_rows)
                    if rows:
                        for row in rows:
                            key = (
                                tool_name,
                                row["resource_type"],
                                row["id"],
                                row["version"],
                                row["source_pixel_sha256"],
                                row["wire_image_sha256"],
                            )
                            self._pending[key] = row
                    else:
                        for index, (_, facts) in enumerate(images):
                            self._pending[(tool_name, index, facts["sha256"])] = None
                    self._expected = Counter(key[-1] for key in self._pending)
                    self._record(
                        "mcp_images",
                        tool_name=tool_name,
                        image_count=len(images),
                        images=[facts for _, facts in images],
                    )
                summary = result + f"\n[Actual native images attached by {PROTOCOL}: {len(images)}]"
                return {
                    "_multimodal": True,
                    "text_summary": summary,
                    "content": [{"type": "text", "text": summary}, *(part for part, _ in images)],
                }

            return inspect

        module._cache_mcp_image_block = cache_image
        module._run_on_mcp_loop = run_on_loop
        module._make_tool_handler = factory

    def install_http(self, module: Any) -> None:
        """Audit final serialized payloads while preserving the resource barrier."""
        original_send = module.Client.send
        if getattr(original_send, "_arcreel_vision_transport", False):
            return

        def send(client: Any, outbound: Any, *args: Any, **kwargs: Any) -> Any:
            actual = urlsplit(str(outbound.url))
            if (
                outbound.method == "GET"
                and (actual.scheme, actual.hostname, actual.port)
                == (self.endpoint.scheme, self.endpoint.hostname, self.endpoint.port)
                and actual.path == self.endpoint.path.removesuffix("/chat/completions") + "/models"
                and not actual.query
                and not actual.fragment
            ):
                kwargs["follow_redirects"] = False
                return original_send(client, outbound, *args, **kwargs)
            endpoint_matches = (
                (actual.scheme, actual.hostname, actual.port, actual.path)
                == (self.endpoint.scheme, self.endpoint.hostname, self.endpoint.port, self.endpoint.path)
                and not actual.query
                and not actual.fragment
            )
            try:
                raw = outbound.content
                payload = json.loads(raw)
            except (ValueError, module.RequestNotRead):
                raw, payload = b"", None
            images = []
            invalid_images = False
            messages = payload.get("messages", []) if isinstance(payload, dict) else []
            for message in messages:
                parts = message.get("content") if isinstance(message, dict) else None
                for part in parts if isinstance(parts, list) else []:
                    if not isinstance(part, dict) or part.get("type") not in {"image_url", "input_image"}:
                        continue
                    image = part.get("image_url", {})
                    url = image.get("url") if isinstance(image, dict) else None
                    if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                        invalid_images = True
                        continue
                    header, data = url.split(";base64,", 1)
                    try:
                        _, facts = _image_part(data, header.removeprefix("data:"))
                    except ValueError:
                        invalid_images = True
                        continue
                    images.append({"role": message.get("role"), **facts})
            with self._mutex:
                self._request += 1
                request_id = self._request
                actual_images = Counter(image["sha256"] for image in images)
                expected_snapshot = self._expected.copy()
                pending_snapshot = self._pending.copy()
                video_snapshot = list(self._pixels.pending_video_units)
                request_session = self._session_id
                request_turn = self._turn
                request_kind = self._request_kind.get()
                auxiliary = request_kind == "auxiliary"
                required_present = auxiliary or all(
                    actual_images[key] >= count for key, count in expected_snapshot.items()
                )
                accepted = (
                    endpoint_matches
                    and outbound.method == "POST"
                    and isinstance(payload, dict)
                    and payload.get("model") == self.model
                    and not invalid_images
                    and required_present
                )
                self._record(
                    "request",
                    request_id=request_id,
                    request_kind=request_kind,
                    model=self.model,
                    generation_parameters=_generation_parameters(payload),
                    endpoint_matches_configured=endpoint_matches,
                    model_matches_configured=isinstance(payload, dict) and payload.get("model") == self.model,
                    body_sha256=hashlib.sha256(raw).hexdigest(),
                    body_bytes=len(raw),
                    image_count=len(images),
                    pending_observation_count=len(pending_snapshot),
                    images=images,
                    required_images_present=required_present,
                    accepted_for_send=accepted,
                )
            if not accepted:
                raise ValueError("ArcReel local vision inference contract was not preserved")
            kwargs["follow_redirects"] = False
            response = original_send(client, outbound, *args, **kwargs)
            with self._mutex:
                self._record(
                    "response",
                    request_id=request_id,
                    status_code=response.status_code,
                    request_kind=request_kind,
                    model=self.model,
                    body_sha256=hashlib.sha256(raw).hexdigest(),
                    endpoint_matches_configured=endpoint_matches,
                    model_matches_configured=True,
                    session_id=request_session,
                    turn=request_turn,
                )
                if not auxiliary and 200 <= response.status_code < 300:
                    rows = [row for row in pending_snapshot.values() if row is not None]
                    if rows:
                        self._pixels.observed_images.extend(rows)
                        self._record(
                            "observed_images",
                            request_id=request_id,
                            request_kind="main",
                            model=self.model,
                            endpoint_matches_configured=endpoint_matches,
                            model_matches_configured=True,
                            status_code=response.status_code,
                            body_sha256=hashlib.sha256(raw).hexdigest(),
                            rows=rows,
                            session_id=request_session,
                            turn=request_turn,
                        )
                    if video_snapshot:
                        self._pixels.observed_video_units.extend(video_snapshot)
                        self._record(
                            "observed_video_units",
                            request_id=request_id,
                            request_kind="main",
                            model=self.model,
                            endpoint_matches_configured=endpoint_matches,
                            model_matches_configured=True,
                            status_code=response.status_code,
                            body_sha256=hashlib.sha256(raw).hexdigest(),
                            rows=video_snapshot,
                            session_id=request_session,
                            turn=request_turn,
                        )
                        del self._pixels.pending_video_units[: len(video_snapshot)]
                    # A later request may legitimately compress already-delivered pixels.
                    if self._session_id == request_session:
                        for key in pending_snapshot:
                            self._pending.pop(key, None)
                    self._expected = Counter(key[-1] for key in self._pending)
            return response

        send.__dict__["_arcreel_vision_transport"] = True
        module.Client.send = send

    def install_auxiliary(self, module: Any) -> None:
        """Native auxiliary calls may summarize text without observing MCP images."""
        original = module.call_llm

        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            token = self._request_kind.set("auxiliary")
            try:
                return original(*args, **kwargs)
            finally:
                self._request_kind.reset(token)

        module.call_llm = call

    def _has_pending_pixels(self, content: Any) -> bool:
        with self._mutex:
            pending = set(self._expected)
        if not pending:
            return False
        if isinstance(content, dict) and content.get("_multimodal") is True:
            content = content.get("content")
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                continue
            header, data = url.split(";base64,", 1)
            _, facts = _image_part(data, header.removeprefix("data:"))
            if facts["sha256"] in pending:
                return True
        return False

    def install_compression(self, module: Any, auxiliary: Any) -> None:
        """Keep unobserved native tool images until one main request receives them."""
        original_strip = module._strip_images_from_tool_msg
        original_historical = module._strip_historical_media
        # This module binds call_llm at import time, before an ACP agent exists.
        module.call_llm = auxiliary.call_llm

        def strip(message: Any) -> Any:
            if isinstance(message, dict) and self._has_pending_pixels(message.get("content")):
                return None
            return original_strip(message)

        def historical(messages: Any) -> Any:
            protected = {
                index: message
                for index, message in enumerate(messages)
                if isinstance(message, dict) and self._has_pending_pixels(message.get("content"))
            }
            result = original_historical(messages)
            if protected:
                if len(result) != len(messages):
                    raise ValueError("Hermes changed the native media-strip message correspondence")
                result = list(result)
                for index, message in protected.items():
                    result[index] = message
            return result

        module._strip_images_from_tool_msg = strip
        module._strip_historical_media = historical


def install_acp_text_projection(server: Any, events: Any) -> None:
    """ACP/UI receives the summary; only the inference transport receives pixels."""
    original = events.build_tool_complete

    def display_result(value: Any) -> Any:
        if isinstance(value, dict) and value.get("_multimodal") is True:
            return value["text_summary"]
        if isinstance(value, list) and any(
            isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"} for part in value
        ):
            return "\n".join(
                str(part.get("text", ""))
                for part in value
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
            )
        return value

    def build(tool_call_id: str, tool_name: str, result: Any = None, **kwargs: Any) -> Any:
        return original(tool_call_id, tool_name, result=display_result(result), **kwargs)

    events.build_tool_complete = build
    server.build_tool_complete = build
    original_step_factory = server.make_step_cb

    def make_step(*args: Any, **kwargs: Any) -> Any:
        callback = original_step_factory(*args, **kwargs)

        def step(api_call_count: int, prev_tools: Any = None) -> Any:
            projected = prev_tools
            if isinstance(prev_tools, list):
                projected = []
                for item in prev_tools:
                    if isinstance(item, dict):
                        item = item.copy()
                        for key in ("result", "output"):
                            if key in item:
                                item[key] = display_result(item[key])
                    projected.append(item)
            return callback(api_call_count, projected)

        return step

    # The stock step callback stringifies tool results before calling its builder.
    server.make_step_cb = make_step
