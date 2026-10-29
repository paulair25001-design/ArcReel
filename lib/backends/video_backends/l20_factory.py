"""Lazy entry point for ArcReel's own native ComfyUI video backend."""

from __future__ import annotations

from arcreel_market_core.video_backend_contract import (
    VideoBackend,
    VideoCapabilities,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from lib.billing.metering import Usage
from lib.local_runtime.l20_capabilities import VIDEO_MODEL, video_capabilities
from lib.local_runtime.l20_direct_config import L20DirectConfig, get_l20_direct_config


class L20VideoFactory:
    def __call__(self, config: L20DirectConfig | None = None, *, model: str = VIDEO_MODEL) -> VideoBackend:
        from lib.local_runtime.l20_direct_backend import L20DirectVideoBackend

        return L20DirectVideoBackend(config or get_l20_direct_config(), model=model)

    @staticmethod
    def video_capabilities_for_model(model: str) -> VideoCapabilities:
        if model != VIDEO_MODEL:
            raise ValueError("Unsupported local L20 video model")
        return video_capabilities()

    @staticmethod
    def effective_generate_audio_for_model(model: str) -> bool:
        from lib.local_runtime.l20_direct_backend import L20DirectVideoBackend

        return L20DirectVideoBackend.effective_generate_audio_for_model(model)

    @staticmethod
    def metered_usage(
        model: str, request: VideoGenerationRequest, result: VideoGenerationResult | None = None
    ) -> Usage:
        from lib.local_runtime.l20_direct_backend import L20DirectVideoBackend

        return L20DirectVideoBackend.metered_usage(model, request, result)
