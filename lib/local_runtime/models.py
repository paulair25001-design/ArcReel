"""Model facts for the explicitly configured self-hosted execution routes."""

from __future__ import annotations

from typing import Any

from lib.local_runtime.config import HERMES_PROVIDER, L20_PROVIDER, TTS_MODEL, TTS_PROVIDER, get_local_runtime_config
from lib.local_runtime.l20_capabilities import IMAGE_MODEL, IMAGE_SIZES, VIDEO_MODEL, video_model_facts

LOCAL_PROVIDERS = frozenset({HERMES_PROVIDER, L20_PROVIDER, TTS_PROVIDER})


def model_specs(provider_id: str) -> dict[str, dict[str, Any]]:
    if provider_id == HERMES_PROVIDER:
        model = get_local_runtime_config().hermes_model
        return {
            model: {
                "display_name": model,
                "media_type": "text",
                "capabilities": ["text_generation", "structured_output", "vision"],
                "max_output_tokens": 65536,
                "default": True,
            }
        }
    if provider_id == L20_PROVIDER:
        facts = video_model_facts()
        return {
            IMAGE_MODEL: {
                "display_name": "Qwen-Image 2.1 · L20",
                "media_type": "image",
                "capabilities": ["text_to_image", "image_to_image"],
                "resolutions": list(IMAGE_SIZES),
                "default": True,
            },
            VIDEO_MODEL: {
                "display_name": "FastH3 V2 · L20",
                "media_type": "video",
                "capabilities": [],
                "supported_durations": facts["supported_durations"],
                "resolutions": facts["resolutions"],
                "default": True,
            },
        }
    if provider_id == TTS_PROVIDER:
        return {
            TTS_MODEL: {
                "display_name": "Qwen3-TTS · Local",
                "media_type": "audio",
                "capabilities": ["text_to_speech"],
                "default": True,
            }
        }
    return {}
