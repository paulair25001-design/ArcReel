"""Local route policy and assembly without invoking any inference or transport."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.backends.backend_assembly import assemble_backend
from lib.backends.backend_assembly.specs import builtin_video_capabilities_for_model
from lib.backends.image_backends.base import ImageGenerationRequest, ReferenceImage
from lib.catalog import query as catalog
from lib.catalog.errors import ModelNotInCatalogError
from lib.config.resolver import ConfigResolver
from lib.local_runtime.config import (
    LocalOnlyPolicyError,
    get_local_runtime_config,
    local_model_pair,
    local_only_enabled,
    require_local_provider,
)
from lib.local_runtime.l20_capabilities import IMAGE_MODEL, VIDEO_DURATIONS, VIDEO_MODEL, VIDEO_RESOLUTIONS
from lib.local_runtime.l20_direct_backend import L20DirectImageBackend, L20DirectVideoBackend
from lib.local_runtime.l20_direct_config import get_l20_direct_config
from lib.local_runtime.l20_direct_gateway import L20DirectError
from lib.script.grid.grid_resolution import grid_image_size


@pytest.fixture(autouse=True)
def explicit_local_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    monkeypatch.setenv("ARCREEL_HERMES_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("ARCREEL_HERMES_MCP_URL", "http://127.0.0.1:1241/mcp")
    monkeypatch.setenv("ARCREEL_L20_DIRECT_STATE_DIR", str(tmp_path / "l20-direct"))
    monkeypatch.setenv("ARCREEL_HERMES_HOME", str(tmp_path / "hermes-home"))


def test_local_mode_is_default_and_unknown_policy_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARCREEL_LOCAL_ONLY")
    assert local_only_enabled() is True
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "maybe")
    with pytest.raises(LocalOnlyPolicyError):
        local_only_enabled()


@pytest.mark.parametrize("setting", ["ARCREEL_HERMES_BASE_URL", "ARCREEL_HERMES_MCP_URL"])
@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "http://127.0.0.1.example.com/v1",
        "file:///tmp/local.sock",
        "http://secret@127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/v1?api_key=secret",
        "http://127.0.0.1:8000/v1#remote",
        "http://127.0.0.1:invalid/v1",
        "http://localhost:99999/v1",
    ],
)
def test_remote_or_ambiguous_semantic_endpoints_are_rejected(monkeypatch: pytest.MonkeyPatch, setting: str, url: str):
    monkeypatch.setenv(setting, url)
    with pytest.raises(LocalOnlyPolicyError, match="loopback HTTP endpoint"):
        get_local_runtime_config()


def test_fractional_direct_l20_timeout_is_not_truncated_to_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCREEL_L20_TIMEOUT_SECONDS", "0.1")
    assert get_l20_direct_config().timeout_seconds == 0.1
    assert not hasattr(get_local_runtime_config(), "l20_factory_root")


@pytest.mark.parametrize("media_type", ["text", "image", "video", "audio"])
@pytest.mark.parametrize("provider", ["openai", "gemini-aistudio", "custom:1"])
def test_cloud_and_custom_providers_are_rejected_for_every_media(media_type: str, provider: str):
    with pytest.raises(LocalOnlyPolicyError, match="Local-only"):
        require_local_provider(provider, media_type)


@pytest.mark.asyncio
async def test_catalog_and_assembled_backends_share_exact_local_identities(db_factory, tmp_path: Path):
    resolver = ConfigResolver(db_factory)
    expected = {
        "text": local_model_pair("text"),
        "image": ("l20", IMAGE_MODEL),
        "video": ("l20", VIDEO_MODEL),
        "audio": local_model_pair("audio"),
    }
    for media_type, (provider, model) in expected.items():
        assert catalog.default_model(provider, media_type) == model
        assert catalog.get_model(provider, model, media_type=media_type) is not None
        backend = await assemble_backend(provider_id=provider, media_type=media_type, model_id=None, resolver=resolver)
        assert (backend.name, backend.model) == (provider, model)
        if media_type == "image":
            assert isinstance(backend, L20DirectImageBackend)
        elif media_type == "video":
            assert isinstance(backend, L20DirectVideoBackend)
    video = catalog.get_model("l20", VIDEO_MODEL)
    assert video is not None
    assert tuple(video.supported_durations or ()) == VIDEO_DURATIONS
    assert tuple(video.resolutions or ()) == VIDEO_RESOLUTIONS
    caps = builtin_video_capabilities_for_model("l20", VIDEO_MODEL)
    assert caps.first_frame is True
    assert caps.last_frame is True
    assert caps.text_to_video is False
    assert caps.max_reference_images == 0
    # Constructing a backend must not create a controller session or Hermes home.
    assert not (tmp_path / "l20-direct").exists()
    assert not (tmp_path / "hermes-home").exists()


@pytest.mark.asyncio
async def test_assembly_rejects_wrong_provider_or_model_without_cloud_fallback(db_factory):
    resolver = ConfigResolver(db_factory)
    with pytest.raises(LocalOnlyPolicyError):
        await assemble_backend(provider_id="openai", media_type="text", model_id="gpt-5", resolver=resolver)
    with pytest.raises(ModelNotInCatalogError):
        await assemble_backend(provider_id="l20", media_type="video", model_id=IMAGE_MODEL, resolver=resolver)


@pytest.mark.asyncio
async def test_queue_resolver_keeps_local_routes_and_rejects_inflight_cloud_identity(db_factory):
    resolver = ConfigResolver(db_factory)
    assert await resolver.default_text_backend() == local_model_pair("text")
    assert await resolver.default_audio_backend() == local_model_pair("audio")
    image = await resolver.resolve_image_backend({}, {}, generation_type="i2i")
    video = await resolver.resolve_video_backend({}, {}, generation_type="i2v")
    assert (image.provider_id, image.model_id) == local_model_pair("image")
    assert (video.provider_id, video.model_id) == local_model_pair("video")
    with pytest.raises(LocalOnlyPolicyError):
        await resolver.resolve_image_backend({}, {"image_provider": "openai"}, generation_type="t2i")
    with pytest.raises(LocalOnlyPolicyError):
        await resolver.resolve_video_backend({}, {"video_provider_i2v": "openai/sora-2"}, generation_type="i2v")
    with pytest.raises(LocalOnlyPolicyError, match="reference-to-video"):
        await resolver.resolve_video_backend({}, {}, generation_type="r2v")


@pytest.mark.asyncio
async def test_local_resolution_is_explicit_and_unsupported_saved_grid_or_video_sizes_are_refused(db_factory):
    resolver = ConfigResolver(db_factory)
    assert await resolver.resolve_resolution({}, "l20", IMAGE_MODEL) == "1K"
    assert await resolver.resolve_resolution({}, "l20", VIDEO_MODEL) == "576p"
    assert grid_image_size(None, "l20", IMAGE_MODEL) is None
    assert grid_image_size(await resolver.resolve_resolution({}, "l20", IMAGE_MODEL), "l20", IMAGE_MODEL) == "1K"
    for model, unsupported in ((IMAGE_MODEL, "2K"), (VIDEO_MODEL, "720p")):
        project = {"model_settings": {f"l20/{model}": {"resolution": unsupported}}}
        with pytest.raises(LocalOnlyPolicyError, match="supports only"):
            await resolver.resolve_resolution(project, "l20", model)


@pytest.mark.asyncio
async def test_l20_refuses_more_than_four_image_references_before_any_input_or_service_access(tmp_path: Path):
    backend = L20DirectImageBackend(get_l20_direct_config())
    request = ImageGenerationRequest(
        "a cube",
        tmp_path / "output.png",
        [ReferenceImage(str(tmp_path / f"missing-{index}.png")) for index in range(5)],
        project_name="demo",
    )
    with pytest.raises(L20DirectError, match="at most 4 references"):
        await backend.generate(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(("resolution", "available"), [("576p", True), ("720p", False), ("1080p", False)])
async def test_unsaved_video_resolution_preview_respects_local_graph_contract(
    db_factory, resolution: str, available: bool
):
    from lib.generation.video_request_facts import (
        CONFIGURED_VIDEO_IDENTITY,
        ResolutionOverride,
        VideoRequestFacts,
        VideoRequestFactsFailure,
        evaluate_video_request_facts,
    )

    result = await evaluate_video_request_facts(
        {},
        route="storyboard",
        generation_type="i2v",
        identity=CONFIGURED_VIDEO_IDENTITY,
        resolver=ConfigResolver(db_factory),
        resolution_override=ResolutionOverride(resolution),
    )
    if available:
        assert isinstance(result, VideoRequestFacts)
        assert result.resolution == "576p"
    else:
        assert isinstance(result, VideoRequestFactsFailure)
        assert result.code == "video_capability_unavailable"
