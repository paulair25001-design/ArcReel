"""Checkpoints retain raw history privately and fail before lossy compression."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent_runtime.hermes_checkpoint import TranscriptCheckpoint, install_project_checkpoint


def owned_home(tmp_path: Path) -> Path:
    home = tmp_path / "private-hermes"
    home.mkdir()
    (home / ".arcreel-local-hermes").write_text("arcreel-local-hermes/v1\n")
    return home


def test_checkpoint_round_trip_preserves_raw_media_and_native_evidence_on_cold_resume(tmp_path: Path):
    home = owned_home(tmp_path)
    agent = SimpleNamespace(session_id="original-session")
    checkpoint = TranscriptCheckpoint(home, lambda: agent.session_id)
    raw = [
        {"role": "user", "content": "retain this decision"},
        {"role": "tool", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,PRIVATE"}}]},
    ]
    evidence = raw[:1]
    reference = checkpoint.on_pre_compress(raw, evidence_messages=evidence, require_checkpoint=True)
    archive = next((home / "checkpoints/pre-compress/original-session").glob("*.json"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == archive.stem
    assert digest in reference
    assert "PRIVATE" not in reference
    assert "retain this decision" not in reference
    assert archive.stat().st_mode & 0o777 == 0o600
    assert archive.parent.stat().st_mode & 0o777 == 0o700
    assert not list(home.rglob("*.tmp"))
    cold = TranscriptCheckpoint(home, lambda: agent.session_id)
    assert cold.read_checkpoint(agent.session_id, digest)["messages"] == raw
    assert cold.read_checkpoint(agent.session_id, digest)["evidence_messages"] == evidence
    assert checkpoint.on_pre_compress(raw, evidence_messages=evidence) == reference
    assert len(list(archive.parent.glob("*.json"))) == 1

    # Native session rotation is read at checkpoint time; no cached parent ID.
    agent.session_id = "rotated-session"
    checkpoint.on_pre_compress(raw, evidence_messages=evidence)
    assert list((home / "checkpoints/pre-compress/rotated-session").glob("*.json"))
    assert cold.read_checkpoint("original-session", digest)["messages"] == raw


def test_checkpoint_fsync_failure_leaves_no_acknowledged_or_partial_snapshot(tmp_path: Path, monkeypatch):
    home = owned_home(tmp_path)
    checkpoint = TranscriptCheckpoint(home, lambda: "session")
    original = os.fsync

    def failing_file_sync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("simulated full disk")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", failing_file_sync)
    with pytest.raises(OSError, match="simulated full disk"):
        checkpoint.on_pre_compress([{"role": "user", "content": "keep"}], require_checkpoint=True)
    assert not list(home.rglob("*.json"))
    assert not list(home.rglob("*.tmp"))


def test_checkpoint_tamper_rejected_without_overwriting_retained_history(tmp_path: Path):
    home = owned_home(tmp_path)
    checkpoint = TranscriptCheckpoint(home, lambda: "session")
    raw = [{"role": "user", "content": "before"}]
    checkpoint.on_pre_compress(raw)
    archive = next(home.rglob("*.json"))
    archive.write_text(json.dumps({"changed": True}))
    with pytest.raises(RuntimeError, match="digest differs"):
        checkpoint.on_pre_compress(raw, require_checkpoint=True)
    assert json.loads(archive.read_text()) == {"changed": True}


@pytest.mark.parametrize("session_id", ["../outside", "", "bad/session"])
def test_checkpoint_rejects_unsafe_session_identity(tmp_path: Path, session_id: str):
    home = owned_home(tmp_path)
    checkpoint = TranscriptCheckpoint(home, lambda: session_id)
    with pytest.raises(RuntimeError, match="session identity"):
        checkpoint.on_pre_compress([])
    assert not (home / "checkpoints").exists()


def test_checkpoint_refuses_unowned_home_symlink_or_external_provider(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not owned"):
        TranscriptCheckpoint(tmp_path, lambda: "session")
    home = owned_home(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "checkpoints").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        TranscriptCheckpoint(home, lambda: "session").on_pre_compress([])
    assert list(outside.iterdir()) == []
    agent = SimpleNamespace(_memory_manager=SimpleNamespace(providers=["external"]))
    with pytest.raises(RuntimeError, match="active external"):
        install_project_checkpoint(agent, home, SimpleNamespace)
