"""ArcReel-owned inference graphs for the installed Qwen and FastH3 weights.

These are ordinary ComfyUI API graphs, authored from node inputs. They contain
no workflow controller, project state, external script or admission callback.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from lib.local_runtime.l20_capabilities import IMAGE_REFERENCE_LIMIT, VIDEO_FPS, VIDEO_SAMPLING_STEPS

IMAGE_CHECKPOINT = "qwen_image_2.1_int8_convrot.safetensors"
VIDEO_CHECKPOINT = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"


@dataclass(frozen=True)
class DirectGraph:
    workflow: dict[str, Any]
    output_node: str
    native_frames: int | None = None

    @property
    def sha256(self) -> str:
        return workflow_sha256(self.workflow)


def workflow_sha256(workflow: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(workflow, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalized_workflow_sha256(workflow: dict[str, Any]) -> str:
    """Compare JSON numbers after ComfyUI's lossless FLOAT input coercion.

    Booleans, strings, links, graph structure and nonintegral numbers retain
    their JSON identities. The original wire graph and its raw hash stay intact.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("A workflow cannot contain non-finite JSON numbers")
            if value.is_integer():
                return int(value)
        return value

    return workflow_sha256(normalize(workflow))


def aligned_frames(requested_frames: int) -> int:
    """The installed H3 temporal latent has 17-frame groups and five anchors."""
    if type(requested_frames) is not int or not 2 <= requested_frames <= 3600:
        raise ValueError("Unsupported H3 frame count")
    frames = 5 + 17 * max(0, (requested_frames - 5 + 16) // 17)
    if frames > 3600:
        raise ValueError("Unsupported H3 native frame count")
    return frames


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def build_image_graph(
    prompt: str, width: int, height: int, seed: int, references: list[str], prefix: str
) -> DirectGraph:
    if len(references) > IMAGE_REFERENCE_LIMIT:
        raise ValueError(f"Qwen accepts at most {IMAGE_REFERENCE_LIMIT} explicit image references")
    graph = {
        "model": _node("UNETLoader", unet_name=IMAGE_CHECKPOINT, weight_dtype="default"),
        "text": _node(
            "CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default"
        ),
        "vae": _node("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "cache": _node("QwenImage21Cache", model=["model", 0], device="auto", dtype="default"),
        "condition": _node(
            "TextEncodeQwenImage21",
            clip=["text", 0],
            vae=["vae", 0],
            prompt=prompt,
            negative_prompt="",
            resolution=0 if references else min(width, height),
        ),
        "sample": _node(
            "KSampler",
            model=["cache", 0],
            seed=seed,
            steps=25,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            positive=["condition", 0],
            negative=["condition", 1],
            latent_image=["condition", 2] if references else ["latent", 0],
            denoise=1.0,
        ),
        "decode": _node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "output": _node("SaveImage", images=["decode", 0], filename_prefix=prefix),
    }
    if references:
        for index, reference in enumerate(references, 1):
            graph[f"reference{index}"] = _node("LoadImage", image=reference)
            graph["condition"]["inputs"][f"images.image_{index}"] = [f"reference{index}", 0]
    else:
        graph["latent"] = _node("EmptySD3LatentImage", width=width, height=height, batch_size=1)
    return DirectGraph(graph, "output")


def build_video_graph(
    prompt: str, width: int, height: int, seed: int, references: list[str], requested_frames: int, prefix: str
) -> DirectGraph:
    if len(references) != 2:
        raise ValueError("FastH3 needs an explicit first frame and target-tail frame")
    frames = aligned_frames(requested_frames)
    graph = {
        "model": _node("UNETLoader", unet_name=VIDEO_CHECKPOINT, weight_dtype="default"),
        "text": _node(
            "CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"
        ),
        "vae": _node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "audio_vae": _node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "first": _node("LoadImage", image=references[0]),
        "tail": _node("LoadImage", image=references[1]),
        "condition": _node(
            "MiniMaxH3ImageToVideo",
            clip=["text", 0],
            vae=["vae", 0],
            prompt=prompt,
            width=width,
            height=height,
            length=frames,
            first_frame=["first", 0],
            last_frame=["tail", 0],
        ),
        "shift": _node("MiniMaxH3SigmaShift", model=["model", 0], shift_video=10.0, shift_audio=3.0),
        "sparse": _node(
            "BlockSparseAttention",
            model=["shift", 0],
            **{
                "selection": "vsa",
                "selection.keep_percent": 10.0,
                "start_percent": 0.2,
                "end_percent": 1.0,
                "dense_blocks": "",
                "min_tokens": 12288,
                "extra_tokens": 256,
                "sink_conditioning": "exact_kv_and_rows",
                "verbose": False,
            },
        ),
        "attention": _node("ModelAttentionBackend", model=["sparse", 0], attention="comfy kitchen attention"),
        "guider": _node("BasicGuider", model=["attention", 0], conditioning=["condition", 0]),
        "noise": _node("RandomNoise", noise_seed=seed),
        "sampler": _node("KSamplerSelect", sampler_name="res_multistep"),
        "schedule": _node(
            "BasicScheduler", model=["attention", 0], scheduler="simple", steps=VIDEO_SAMPLING_STEPS, denoise=1.0
        ),
        "sample": _node(
            "SamplerCustomAdvanced",
            noise=["noise", 0],
            guider=["guider", 0],
            sampler=["sampler", 0],
            sigmas=["schedule", 0],
            latent_image=["condition", 1],
        ),
        "decode": _node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "decode_audio": _node("VAEDecodeAudio", samples=["sample", 0], vae=["audio_vae", 0]),
        "video": _node(
            "CreateVideo",
            images=["decode", 0],
            audio=["decode_audio", 0],
            fps=VIDEO_FPS,
            bit_depth=8,
            color_space="sRGB",
        ),
        "output": _node(
            "SaveVideo",
            video=["video", 0],
            filename_prefix=prefix,
            **{
                "format": "mp4",
                "format.codec": "h264",
                "format.codec.encoding": "re-encode",
                "format.codec.encoding.crf": 18,
            },
        ),
    }
    return DirectGraph(graph, "output", frames)
