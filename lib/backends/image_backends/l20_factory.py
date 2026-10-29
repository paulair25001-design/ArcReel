"""Lazy entry point for ArcReel's own native ComfyUI image backend."""

from __future__ import annotations

from lib.backends.image_backends.base import BOTH_IMAGE_CAPABILITIES, ImageBackend, ImageRequestFacts
from lib.billing.metering import Usage
from lib.local_runtime.l20_capabilities import IMAGE_MODEL
from lib.local_runtime.l20_direct_config import L20DirectConfig, get_l20_direct_config


class L20ImageFactory:
    CHANNEL_CAPABILITIES = BOTH_IMAGE_CAPABILITIES

    def __call__(self, config: L20DirectConfig | None = None, *, model: str = IMAGE_MODEL) -> ImageBackend:
        from lib.local_runtime.l20_direct_backend import L20DirectImageBackend

        return L20DirectImageBackend(config or get_l20_direct_config(), model=model)

    @staticmethod
    def estimate_usage(facts: ImageRequestFacts) -> Usage:
        from lib.local_runtime.l20_direct_backend import L20DirectImageBackend

        return L20DirectImageBackend.estimate_usage(facts)
