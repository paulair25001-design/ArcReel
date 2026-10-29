"""Local route presentation and protection for model connection probes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from lib.local_runtime.config import get_local_runtime_config, local_only_enabled
from server.i18n import get_translator


def is_external_inference_probe(method: str, path: str) -> bool:
    """Connection tests and trial runs bypass normal generation backend assembly."""
    if method != "POST":
        return False
    parts = path.rstrip("/").split("/")
    if path == "/api/v1/custom-endpoints/trial-runs":
        return True
    if path.startswith(("/api/v1/providers/", "/api/v1/custom-providers/")):
        return parts[-1] in {"test", "discover", "discover-anthropic"}
    return path.startswith("/api/v1/agent/") and parts[-1] in {"test", "test-connection"}


class LocalOnlyProbeMiddleware:
    """Fail before any alternate provider connection or inference is attempted."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] == "http"
            and local_only_enabled()
            and is_external_inference_probe(scope["method"], scope["path"])
        ):
            request = Request(scope)
            response = JSONResponse(
                {
                    "detail": get_translator(request)("local_only_external_inference_disabled"),
                    "code": "local_only_policy",
                },
                status_code=403,
            )
            return response(scope, receive, send)
        return self.app(scope, receive, send)


async def local_runtime_status() -> dict[str, Any]:
    config = get_local_runtime_config()

    def installed() -> bool:
        from pathlib import Path

        return Path(config.hermes_python).is_file() and (config.hermes_source / "acp_adapter").is_dir()

    return {
        "mode": "local-hermes-l20",
        "agent_available": await asyncio.to_thread(installed),
        "agent_model": config.hermes_model,
        "image_model": "l20/qwen-image-2.1-int8",
        "video_model": "l20/fasth3-v2-4step",
        "audio_model": "local-tts/qwen3-tts-voice-design",
        "cloud_fallback": False,
    }
