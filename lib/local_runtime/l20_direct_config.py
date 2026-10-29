"""Connection settings for ArcReel's own L20 provider, without factory paths."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from lib.infra.data_root_layout import DataRootLayout


@dataclass(frozen=True, slots=True)
class L20DirectConfig:
    comfy_url: str
    router_url: str
    state_dir: Path
    poll_interval: float = 2
    timeout_seconds: float = 9600

    def __post_init__(self) -> None:
        for value in (self.comfy_url, self.router_url):
            try:
                parsed = urlsplit(value)
                port = parsed.port
            except ValueError as error:
                raise ValueError("L20 must use the explicit project-owned loopback connection") from error
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
                or port == 0
            ):
                raise ValueError("L20 must use the explicit project-owned loopback connection")
        if not all(math.isfinite(x) and x > 0 for x in (self.poll_interval, self.timeout_seconds)):
            raise ValueError("L20 polling and timeout must be positive and finite")


def get_l20_direct_config() -> L20DirectConfig:
    return L20DirectConfig(
        comfy_url=os.environ.get("ARCREEL_L20_COMFY_URL", "http://127.0.0.1:18191").rstrip("/"),
        router_url=os.environ.get("ARCREEL_L20_ROUTER_URL", "http://127.0.0.1:18080").rstrip("/"),
        state_dir=Path(
            os.environ.get("ARCREEL_L20_DIRECT_STATE_DIR", str(DataRootLayout.current().runtime_dir / "l20-direct"))
        ).expanduser(),
        poll_interval=float(os.environ.get("ARCREEL_L20_POLL_INTERVAL", "2")),
        timeout_seconds=float(os.environ.get("ARCREEL_L20_TIMEOUT_SECONDS", "9600")),
    )
