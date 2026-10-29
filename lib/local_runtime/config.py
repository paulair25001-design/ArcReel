"""Explicit local execution settings; no discovery or provider fallback."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from lib.infra.data_root_layout import DataRootLayout
from lib.local_runtime.l20_capabilities import IMAGE_MODEL, VIDEO_MODEL

HERMES_PROVIDER = "local-hermes"
L20_PROVIDER = "l20"
TTS_PROVIDER = "local-tts"
TTS_MODEL = "qwen3-tts-voice-design"
RESOURCE_QUEUE_HELPER = Path(__file__).with_name("resource_queue.py").resolve()
TTS_WORKER = Path(__file__).with_name("qwen_tts_worker.py").resolve()


class LocalOnlyPolicyError(ValueError):
    """A configured route would leave the explicitly selected local runtime."""


def local_only_enabled() -> bool:
    value = os.environ.get("ARCREEL_LOCAL_ONLY", "true").strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise LocalOnlyPolicyError("ARCREEL_LOCAL_ONLY must be true or false")


def _positive_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise LocalOnlyPolicyError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise LocalOnlyPolicyError(f"{name} must be a positive finite number")
    return value


def _loopback_url(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise LocalOnlyPolicyError(f"{name} must be a valid loopback HTTP endpoint") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        raise LocalOnlyPolicyError(f"{name} must be an explicit loopback HTTP endpoint")
    return value


def _project_worker(name: str, expected: Path) -> Path:
    configured = os.environ.get(name)
    if configured is not None and Path(configured).expanduser().resolve() != expected:
        raise LocalOnlyPolicyError(f"{name} must use ArcReel's project-owned worker")
    return expected


@dataclass(frozen=True, slots=True)
class LocalRuntimeConfig:
    enabled: bool
    hermes_executable: str
    hermes_python: str
    hermes_source: Path
    hermes_model: str
    hermes_base_url: str
    hermes_api_key: str = field(repr=False)
    hermes_resource_queue_helper: Path
    resource_root: Path
    hermes_home: Path
    hermes_timeout_seconds: float
    hermes_mcp_url: str
    hermes_mcp_token: str | None = field(repr=False)
    tts_python: str
    tts_script: Path
    tts_site_packages: Path
    tts_model: Path
    tts_timeout_seconds: float
    tts_receipts_dir: Path


def get_local_runtime_config() -> LocalRuntimeConfig:
    """Read environment settings without touching files or starting any engine."""
    home = Path.home()
    layout = DataRootLayout.current()
    hermes_source = Path(os.environ.get("ARCREEL_HERMES_SOURCE", str(home / ".hermes/hermes-agent"))).expanduser()
    return LocalRuntimeConfig(
        enabled=local_only_enabled(),
        hermes_executable=os.environ.get("ARCREEL_HERMES_EXECUTABLE", str(home / ".local/bin/hermes")),
        hermes_python=os.environ.get("ARCREEL_HERMES_PYTHON", str(hermes_source / "venv/bin/python")),
        hermes_source=hermes_source,
        hermes_model=os.environ.get("ARCREEL_HERMES_MODEL", "Qwen3.8-Flash-Next-Uncensored-oQ5e-mtp"),
        hermes_base_url=_loopback_url("ARCREEL_HERMES_BASE_URL", "http://127.0.0.1:8000/v1"),
        hermes_api_key=os.environ.get("ARCREEL_HERMES_API_KEY", "arcreel-local-only"),
        hermes_resource_queue_helper=_project_worker("ARCREEL_HERMES_RESOURCE_QUEUE_HELPER", RESOURCE_QUEUE_HELPER),
        resource_root=layout.runtime_dir / "resources",
        hermes_home=Path(os.environ.get("ARCREEL_HERMES_HOME", str(layout.hermes_runtime_dir))).expanduser(),
        hermes_timeout_seconds=_positive_number("ARCREEL_HERMES_TIMEOUT_SECONDS", 1800),
        hermes_mcp_url=_loopback_url("ARCREEL_HERMES_MCP_URL", "http://127.0.0.1:1241/mcp"),
        hermes_mcp_token=os.environ.get("ARCREEL_HERMES_MCP_TOKEN") or None,
        tts_python=os.environ.get(
            "ARCREEL_TTS_PYTHON", "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3.11"
        ),
        tts_script=_project_worker("ARCREEL_TTS_SCRIPT", TTS_WORKER),
        tts_site_packages=Path(
            os.environ.get(
                "ARCREEL_TTS_SITE_PACKAGES",
                "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages",
            )
        ).expanduser(),
        tts_model=Path(
            os.environ.get("ARCREEL_TTS_MODEL", "/Volumes/Data/Codex-Local-Models/qwen3-tts-voice-design")
        ).expanduser(),
        tts_timeout_seconds=_positive_number("ARCREEL_TTS_TIMEOUT_SECONDS", 1800),
        tts_receipts_dir=Path(os.environ.get("ARCREEL_TTS_RECEIPTS_DIR", str(layout.tts_receipts_dir))).expanduser(),
    )


def require_local_provider(provider_id: str, media_type: str) -> None:
    expected = {"text": HERMES_PROVIDER, "image": L20_PROVIDER, "video": L20_PROVIDER, "audio": TTS_PROVIDER}
    if local_only_enabled() and provider_id != expected.get(media_type):
        raise LocalOnlyPolicyError(f"Local-only {media_type} execution requires {expected.get(media_type)}")


def local_model_pair(media_type: str) -> tuple[str, str]:
    pairs = {
        "text": (HERMES_PROVIDER, get_local_runtime_config().hermes_model),
        "image": (L20_PROVIDER, IMAGE_MODEL),
        "video": (L20_PROVIDER, VIDEO_MODEL),
        "audio": (TTS_PROVIDER, TTS_MODEL),
    }
    try:
        return pairs[media_type]
    except KeyError as exc:
        raise LocalOnlyPolicyError(f"Unsupported local media type: {media_type}") from exc
