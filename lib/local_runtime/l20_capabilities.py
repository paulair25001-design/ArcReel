"""Capabilities of the protected Qwen-Image and FastH3 execution routes."""

from __future__ import annotations

import re

from arcreel_market_core.aspect_size import aspect_size
from arcreel_market_core.video_backend_contract import VideoAudioMode, VideoCapabilities

IMAGE_MODEL = "qwen-image-2.1-int8"
VIDEO_MODEL = "fasth3-v2-4step"
IMAGE_REFERENCE_LIMIT = 4
IMAGE_SIZES = ("1K",)
VIDEO_DURATIONS = tuple(range(1, 11))
VIDEO_RESOLUTIONS = ("576p",)
VIDEO_FPS = 24
VIDEO_SAMPLING_STEPS = 4
REQUIRES_END_FRAME = True


def video_capabilities() -> VideoCapabilities:
    """The native graph consumes an exact first/target-tail pair and produces audio."""
    return VideoCapabilities(
        text_to_video=False,
        first_frame=True,
        last_frame=True,
        max_reference_images=0,
        max_prompt_chars=32_768,
        audio_track=VideoAudioMode.ALWAYS_ON,
    )


def canvas(aspect_ratio: str) -> tuple[int, int]:
    """Use the exact 32-pixel grid required by Qwen reference-edit latents."""
    if not re.fullmatch(r"[1-9][0-9]{0,3}:[1-9][0-9]{0,3}", aspect_ratio):
        raise ValueError("The protected L20 route requires an explicit positive aspect ratio")
    size = aspect_size(aspect_ratio, 576, round_to=32, max_long_edge=2048)
    if min(size) < 64 or max(size) > 4096:
        raise ValueError("The protected L20 canvas must stay within 64..4096 pixels")
    return size


def image_canvas(aspect_ratio: str, image_size: str | None) -> tuple[int, int]:
    if image_size not in (None, *IMAGE_SIZES):
        raise ValueError("The protected L20 image route supports the 1K canvas preset only")
    return canvas(aspect_ratio)


def video_model_facts() -> dict[str, object]:
    return {
        "supported_durations": list(VIDEO_DURATIONS),
        "resolutions": list(VIDEO_RESOLUTIONS),
        "fps": VIDEO_FPS,
        "sampling_steps": VIDEO_SAMPLING_STEPS,
        "requires_end_frame": REQUIRES_END_FRAME,
        "reference_images": 0,
        "reference_audio": 0,
    }
