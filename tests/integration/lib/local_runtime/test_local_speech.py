"""Offline speech receipts survive real audio staging and version commits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lib.artifacts.version_manager import PaidVersionCommit
from lib.billing.ledger import Ledger
from lib.db.models.api_call import ApiCall
from lib.generation.media_generator import MediaGenerator
from lib.local_runtime.config import get_local_runtime_config
from lib.local_runtime.local_audio import LOCAL_NARRATOR_VOICE, LocalAudioBackend


class _FinishedProcess:
    """Substitute only the installed TTS subprocess boundary."""

    pid = None
    returncode = 0

    def __init__(self) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        raise AssertionError("completed synthesis must not be terminated")

    def kill(self) -> None:
        raise AssertionError("completed synthesis must not be killed")


def _write_wrapper_output(args: tuple[str, ...]) -> None:
    options = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
    output = Path(options["output"])
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x00\x00" * 240)
    receipt = {
        "status": "AUDIO_GENERATED_LISTENING_NOT_VERIFIED",
        "variant": "voice_design",
        "output": str(output),
        "samples_per_channel": 240,
        "sample_rate": 24000,
        "network": "denied_by_macos_sandbox",
        "paid_voice_called": False,
        "model_downloaded": False,
        "worker": "arcreel-offline-qwen3-tts/v1",
        "network_sandbox_proof": {"blocked_by_os": True, "external_connections_attempted": 0},
        "resource_protection": {
            "root": options["resource_root"],
            "exclusive_local_inference": True,
            "serialized_tts": True,
        },
        "model": options["model"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "text": options["text"],
        "instructions": "\n".join((options["voice_description"], options["instructions"])),
    }
    output.with_suffix(".voice-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


async def _spawn_tts(*args: str, **kwargs: Any) -> _FinishedProcess:
    await asyncio.to_thread(_write_wrapper_output, args)
    return _FinishedProcess()


@pytest.mark.parametrize("commit_mode", ["standard", "selected", "history"])
async def test_local_speech_receipt_is_bound_to_committed_version(
    tmp_path: Path, session_factory: async_sessionmaker[AsyncSession], commit_mode: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_tts","tts_model_type":"voice_design"}', encoding="utf-8")
    config = replace(
        get_local_runtime_config(),
        tts_model=model,
        resource_root=tmp_path / "resources",
        tts_receipts_dir=tmp_path / "runtime" / "tts" / "receipts",
    )
    generator = MediaGenerator(
        tmp_path / "projects" / "demo",
        audio_backend=LocalAudioBackend(config, spawn=_spawn_tts),
        audio_provider_id="local-tts",
    )
    generator.ledger = Ledger(session_factory=session_factory)

    def commit(staged: Path, canonical: Path) -> PaidVersionCommit:
        return generator.versions.commit_staged_paid_version(
            "audio",
            "E1S01",
            "原始旁白",
            staged_file=staged,
            current_file=canonical,
            select_current=commit_mode == "selected",
        )

    output, version = await generator.generate_audio_async(
        text="原始旁白",
        resource_id="E1S01",
        voice=LOCAL_NARRATOR_VOICE,
        voice_description="成熟温暖的中文女声",
        instructions="轻声讲述，按句意停顿。",
        commit_staged=None if commit_mode == "standard" else commit,
    )
    records = await asyncio.to_thread(generator.versions.get_versions, "audio", "E1S01")
    record = next(item for item in records["versions"] if item["version"] == version)
    evidence = record["local_speech"]
    receipt_path = Path(evidence["receipt_path"])
    receipt_bytes = await asyncio.to_thread(receipt_path.read_bytes)
    receipt = json.loads(receipt_bytes)
    snapshot = generator.project_path / record["file"]
    snapshot_bytes = await asyncio.to_thread(snapshot.read_bytes)
    assert evidence["receipt_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert receipt == evidence["receipt"]
    assert receipt["sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
    assert receipt["output"] != str(output)
    assert receipt["instructions"] == "成熟温暖的中文女声\n轻声讲述，按句意停顿。"
    assert evidence["selected_current"] is (commit_mode != "history")
    assert evidence["canonical_output"] == (str(output) if commit_mode != "history" else None)
    assert await asyncio.to_thread(output.exists) is (commit_mode != "history")
    assert not await asyncio.to_thread(lambda: list(output.parent.glob(".*.voice-receipt.json")))
    async with session_factory() as session:
        rows = (await session.scalars(select(ApiCall))).all()
        assert len(rows) == 1
        assert rows[0].status == "success"
        assert rows[0].provider == "local-tts"
