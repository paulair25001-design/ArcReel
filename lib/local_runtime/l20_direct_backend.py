"""L20 model backends using ArcReel's queue and native ComfyUI API only."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from arcreel_market_core.video_backend_contract import VideoCapabilities, VideoGenerationRequest, VideoGenerationResult
from lib.backends.backend_runtime import ProviderJobIdPersistenceMixin
from lib.backends.image_backends.base import (
    BOTH_IMAGE_CAPABILITIES,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageRequestFacts,
    per_image_usage,
)
from lib.backends.video_backends.metering import video_usage
from lib.billing.metering import Usage
from lib.infra.async_thread import run_noninterruptible_sync, run_sync_transaction
from lib.infra.ffmpeg import ffmpeg_executable, local_file_input
from lib.infra.media_probe import probe_media
from lib.local_runtime.l20_capabilities import (
    IMAGE_MODEL,
    IMAGE_REFERENCE_LIMIT,
    VIDEO_DURATIONS,
    VIDEO_FPS,
    VIDEO_MODEL,
    VIDEO_RESOLUTIONS,
    canvas,
    image_canvas,
    video_capabilities,
)
from lib.local_runtime.l20_direct_config import L20DirectConfig
from lib.local_runtime.l20_direct_gateway import DirectArtifact, L20DirectError, L20DirectGateway, ResourceSession


def _source(path: Path) -> dict[str, Any]:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise L20DirectError("The ArcReel input is missing or redirected")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": checksum, "bytes": path.stat().st_size}


def _seed(value: int | None, identity: str) -> int:
    if value is None:
        return int(hashlib.sha256(identity.encode()).hexdigest()[:15], 16)
    if type(value) is not int or not 0 <= value < 2**64:
        raise L20DirectError("The L20 seed must be an unsigned 64-bit integer")
    return value


def _gateway(
    config: L20DirectConfig, gateway: L20DirectGateway | None, resource_session: ResourceSession | None
) -> L20DirectGateway:
    if gateway is not None:
        return gateway
    if resource_session is None:
        from lib.local_runtime.l20_resource_session import L20ResourceSession

        return L20DirectGateway(config, L20ResourceSession(config))
    return L20DirectGateway(config, resource_session)


def _image_deliver(artifact: DirectArtifact, destination: Path, size: tuple[int, int]) -> None:
    with Image.open(artifact.path) as image:
        image.load()
        if image.format != "PNG" or image.size != size:
            raise L20DirectError("ComfyUI returned a different image canvas or format")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".l20-", suffix=".png", dir=destination.parent)
    os.close(fd)
    path = Path(temporary)
    try:
        shutil.copyfile(artifact.path, path)
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)  # hard-delete: temp -- atomic image publication


def _video_deliver(artifact: DirectArtifact, destination: Path, frames: int, size: tuple[int, int]) -> None:
    executable = ffmpeg_executable()
    decoded = subprocess.run(
        [executable, "-v", "error", "-nostdin", *local_file_input(artifact.path), "-f", "framecrc", "-"],
        capture_output=True,
        timeout=300,
        check=False,
    )
    if decoded.returncode:
        raise L20DirectError("The downloaded L20 video failed full decoding")
    dimensions = re.search(rb"^#dimensions\s+\d+:\s+(\d+)x(\d+)\s*$", decoded.stdout, re.MULTILINE)
    if dimensions is None or tuple(map(int, dimensions.groups())) != size:
        raise L20DirectError("The native L20 video canvas differs from its request")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".l20-", suffix=".mp4", dir=destination.parent)
    os.close(fd)
    path = Path(temporary)
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-nostdin",
                "-y",
                *local_file_input(artifact.path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-frames:v",
                str(frames),
                "-t",
                str(frames / VIDEO_FPS),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(path),
            ],
            capture_output=True,
            timeout=300,
            check=False,
        )
        if result.returncode or not path.stat().st_size:
            raise L20DirectError("The L20 delivery interval could not be encoded")
        os.replace(path, destination)
    finally:
        path.unlink(missing_ok=True)  # hard-delete: temp -- atomic video publication


class L20DirectImageBackend:
    CHANNEL_CAPABILITIES = BOTH_IMAGE_CAPABILITIES

    def __init__(
        self,
        config: L20DirectConfig,
        *,
        model: str = IMAGE_MODEL,
        gateway: L20DirectGateway | None = None,
        resource_session: ResourceSession | None = None,
    ) -> None:
        if model != IMAGE_MODEL:
            raise L20DirectError("Unsupported local image model")
        self.model = model
        self.gateway = _gateway(config, gateway, resource_session)

    @property
    def name(self) -> str:
        return "l20"

    @property
    def capabilities(self) -> set[Any]:
        return set(self.CHANNEL_CAPABILITIES)

    async def reference_image_limit(self) -> int:
        return IMAGE_REFERENCE_LIMIT

    @staticmethod
    def estimate_usage(facts: ImageRequestFacts) -> Usage:
        return per_image_usage(output_images=1, input_images=facts.reference_count)

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        if len(request.reference_images) > IMAGE_REFERENCE_LIMIT:
            raise L20DirectError(f"The local image graph supports at most {IMAGE_REFERENCE_LIMIT} references")
        identity = str(request.output_path.absolute())
        size = image_canvas(request.aspect_ratio, request.image_size)
        intent = {
            "project": request.project_name,
            "media_type": "image",
            "prompt": request.prompt,
            "seed": _seed(request.seed, identity),
            "canvas": list(size),
            "references": await run_sync_transaction(
                lambda: [_source(Path(row.path)) for row in request.reference_images]
            ),
        }
        handle = await run_sync_transaction(self.gateway.prepare, intent, task_identity=identity)
        artifact = await run_noninterruptible_sync(self.gateway.run, handle)
        await run_noninterruptible_sync(_image_deliver, artifact, request.output_path, size)
        return ImageGenerationResult(
            image_path=request.output_path,
            provider=self.name,
            model=self.model,
            image_uri="l20:" + handle,
            seed=int(intent["seed"]),
            usage=self.estimate_usage(
                ImageRequestFacts(
                    model=self.model, aspect_ratio=request.aspect_ratio, reference_count=len(request.reference_images)
                )
            ),
        )


class L20DirectVideoBackend(ProviderJobIdPersistenceMixin):
    requires_end_frame = True

    def __init__(
        self,
        config: L20DirectConfig,
        *,
        model: str = VIDEO_MODEL,
        gateway: L20DirectGateway | None = None,
        resource_session: ResourceSession | None = None,
    ) -> None:
        if model != VIDEO_MODEL:
            raise L20DirectError("Unsupported local video model")
        self.model = model
        self.gateway = _gateway(config, gateway, resource_session)

    @property
    def name(self) -> str:
        return "l20"

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return video_capabilities()

    @staticmethod
    def video_capabilities_for_model(model: str) -> VideoCapabilities:
        if model != VIDEO_MODEL:
            raise L20DirectError("Unsupported local video model")
        return video_capabilities()

    @staticmethod
    def effective_generate_audio_for_model(model: str) -> bool:
        if model != VIDEO_MODEL:
            raise L20DirectError("Unsupported local video model")
        return True

    @staticmethod
    def metered_usage(
        model: str, request: VideoGenerationRequest, result: VideoGenerationResult | None = None
    ) -> Usage:
        L20DirectVideoBackend.video_capabilities_for_model(model)
        return video_usage(
            request,
            result,
            seconds=request.duration_seconds,
            resolution=request.resolution or VIDEO_RESOLUTIONS[0],
            audio=True,
            reference_images=0,
            reference_to_video=False,
        )

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        if (
            request.start_image is None
            or request.end_image is None
            or request.reference_images
            or request.reference_audio_files
        ):
            raise L20DirectError("FastH3 requires exactly the explicit first and target-tail frames")
        if (
            type(request.duration_seconds) is not int
            or request.duration_seconds not in VIDEO_DURATIONS
            or request.resolution not in (None, *VIDEO_RESOLUTIONS)
        ):
            raise L20DirectError("Unsupported local video duration or resolution")
        identity = request.task_id or str(request.output_path.absolute())
        first, tail = request.start_image, request.end_image
        intent = {
            "project": request.project_name,
            "media_type": "video",
            "prompt": request.prompt,
            "seed": _seed(request.seed, identity),
            "canvas": list(canvas(request.aspect_ratio, request.resolution)),
            "requested_frames": request.duration_seconds * VIDEO_FPS,
            "references": await run_sync_transaction(lambda: [_source(first), _source(tail)]),
        }
        handle = await run_sync_transaction(self.gateway.prepare, intent, task_identity=identity)
        await self._persist_provider_job_id(request, handle, provider=self.name)
        if request.on_provider_resubmit_unsafe is not None:
            request.on_provider_resubmit_unsafe()
        return await self._finish(handle, request)

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        return await self._finish(job_id, request)

    async def _finish(self, handle: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        intent = await run_sync_transaction(self.gateway.intent, handle)
        if (
            intent["project"] != request.project_name
            or intent["prompt"] != request.prompt
            or intent["canvas"] != list(canvas(request.aspect_ratio, request.resolution))
            or intent["requested_frames"] != request.duration_seconds * VIDEO_FPS
            or (request.seed is not None and request.seed != intent["seed"])
        ):
            raise L20DirectError("The accepted video belongs to a different frozen ArcReel request")
        artifact = await run_noninterruptible_sync(self.gateway.run, handle)
        native = await probe_media(artifact.path)
        stream = native.first_stream("video")
        count = artifact.record["native_frames"]
        if (
            stream is None
            or stream.packet_count != count
            or stream.duration_seconds is None
            or not math.isclose(stream.duration_seconds, count / VIDEO_FPS, abs_tol=0.001)
            or native.first_stream("audio") is None
        ):
            raise L20DirectError("The native video canvas, frame count, duration or audio differs")
        await run_noninterruptible_sync(
            _video_deliver, artifact, request.output_path, intent["requested_frames"], tuple(intent["canvas"])
        )
        media = await probe_media(request.output_path)
        delivered = media.first_stream("video")
        if (
            delivered is None
            or delivered.packet_count != intent["requested_frames"]
            or delivered.duration_seconds is None
            or not math.isclose(delivered.duration_seconds, request.duration_seconds, abs_tol=0.001)
            or media.first_stream("audio") is None
        ):
            raise L20DirectError("The delivered video does not match its declared interval")
        return VideoGenerationResult(
            video_path=request.output_path,
            provider=self.name,
            model=self.model,
            duration_seconds=request.duration_seconds,
            task_id=handle,
            video_uri="l20:" + handle,
            seed=int(intent["seed"]),
            generate_audio=True,
            provenance={
                "runtime": "arcreel-native-comfyui",
                "prompt_id": artifact.record["prompt_id"],
                "workflow_sha256": artifact.record["workflow_sha256"],
                "native_sha256": artifact.sha256,
                "source_reference_sha256": [row["sha256"] for row in intent["references"]],
                "sampling_steps": 4,
                "quality_accepted": False,
            },
        )
