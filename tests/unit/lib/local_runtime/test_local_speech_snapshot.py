"""Local speech settings survive the task executor, basis, and version restore."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lib.artifacts.artifact_version_provenance import parse_typed_audio_settings
from lib.config.resolver import ProviderModel
from lib.local_runtime.config import TTS_MODEL, TTS_PROVIDER
from lib.speech.narration_config import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_VOICE_DESCRIPTION,
    LOCAL_NARRATOR_VOICE,
    project_tts_settings,
    tts_snapshot_fields,
)
from lib.speech.narration_delivery import build_narration_audio_basis_from_canonical_text
from server.services.tasks import generation_tasks
from server.services.tasks.generation_context import AudioLaneResult, GenerationContext
from tests.fakes import use_generation_context
from tests.unit.server.services.tasks.test_execute_tts_task import tts_env as tts_env


def test_effective_local_snapshot_binds_local_defaults_and_discards_legacy_paid_speed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    settings = project_tts_settings(
        {"audio_backend": "dashscope/paid-tts", "narration_voice": "Cherry", "narration_speed": 1.2}
    )
    assert settings is not None
    assert (settings.provider_id, settings.model_id) == (TTS_PROVIDER, TTS_MODEL)
    assert settings.voice == LOCAL_NARRATOR_VOICE
    assert settings.speed is None
    assert settings.voice_description == DEFAULT_VOICE_DESCRIPTION
    assert settings.instructions == DEFAULT_INSTRUCTIONS


def test_voice_description_and_performance_round_trip_and_change_narration_basis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    settings = project_tts_settings(
        {
            "narration_voice_description": "成年男声，沉稳，略带沙哑",
            "narration_instructions": "压低声音，最后一句加快。",
        }
    )
    assert settings is not None
    assert project_tts_settings(tts_snapshot_fields(settings)) == settings
    basis = build_narration_audio_basis_from_canonical_text("旁白正文", settings)
    assert basis != build_narration_audio_basis_from_canonical_text(
        "旁白正文", replace(settings, voice_description="年轻女声")
    )
    assert basis != build_narration_audio_basis_from_canonical_text(
        "旁白正文", replace(settings, instructions="轻快。")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change_during_synthesis", [False, True])
async def test_tts_task_uses_local_snapshot_and_preserves_performance_provenance(
    tts_env, monkeypatch: pytest.MonkeyPatch, change_during_synthesis: bool
):
    monkeypatch.setenv("ARCREEL_LOCAL_ONLY", "true")
    pm, generator = tts_env
    pm.project.update(
        {
            "narration_speed": 1.25,
            "narration_voice_description": "成年男声，沉稳，略带沙哑",
            "narration_instructions": "压低声音，最后一句加快。",
        }
    )
    context = GenerationContext(
        generator=generator,
        audio_lane=AudioLaneResult(
            provider_model=ProviderModel(TTS_PROVIDER, TTS_MODEL),
            backend_name=TTS_PROVIDER,
            backend_model=TTS_MODEL,
            voices=(),
        ),
    )

    async def resolve(*args, **kwargs):
        return context

    use_generation_context(monkeypatch, resolve)
    if change_during_synthesis:
        original = generator.generate_audio_async

        async def changed_performance(**kwargs):
            pm.project["narration_instructions"] = "语气轻快，不压低声音。"
            return await original(**kwargs)

        monkeypatch.setattr(generator, "generate_audio_async", changed_performance)

    result = await generation_tasks.execute_tts_task("demo", "E1S01", {"script_file": "episode_1.json"})
    call = generator.audio_calls[-1]
    assert call["voice"] == LOCAL_NARRATOR_VOICE
    assert call["speed"] is None
    assert call["voice_description"] == "成年男声，沉稳，略带沙哑"
    assert call["instructions"] == "压低声音，最后一句加快。"
    assert result["selected_current"] is not change_during_synthesis
    record = generator.version_records[-1]
    restored = parse_typed_audio_settings(record)
    assert restored.voice_description == call["voice_description"]
    assert restored.instructions == call["instructions"]
    assert restored.provider_id == TTS_PROVIDER
