"""Private transcript archives for Hermes' required pre-compress checkpoint API."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SCHEMA = "arcreel-hermes-pre-compress-checkpoint/v1"
_SESSION_ID = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
_DIGEST = re.compile(r"[a-f0-9]{64}")


def _check_path(path: Path) -> None:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise RuntimeError("Hermes checkpoint path must not contain symlinks")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened, current = os.fstat(descriptor), path.lstat()
        if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("Hermes checkpoint directory identity differs")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TranscriptCheckpoint:
    """Archive raw tool/media history and native normalized evidence together.

    Like Hermes' detached-review manager, this implements the manager-level v2
    handoff: regular providers receive only normalized evidence. It does not
    register a memory plugin or write project/production state. Compression's
    native rollback remains responsible for preserving live history on failure.
    """

    def __init__(self, home: Path, session_id: Callable[[], str]) -> None:
        self.home = Path(os.path.abspath(home))
        self._session_id = session_id
        self._require_owned_home()

    def _require_owned_home(self) -> None:
        _check_path(self.home)
        marker = self.home / ".arcreel-local-hermes"
        _check_path(marker)
        if self.home == (Path.home() / ".hermes").resolve() or not marker.is_file():
            raise RuntimeError("Hermes checkpoint home is not owned by ArcReel")
        if marker.read_text(encoding="utf-8") != "arcreel-local-hermes/v1\n":
            raise RuntimeError("Hermes checkpoint home marker is invalid")

    def _directory(self, session_id: object, *, create: bool) -> Path:
        if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
            raise RuntimeError("Hermes checkpoint session identity is invalid")
        self._require_owned_home()
        current = self.home
        for part in ("checkpoints", "pre-compress", session_id):
            child = current / part
            _check_path(child)
            if create:
                child.mkdir(mode=0o700, exist_ok=True)
                os.chmod(child, 0o700)
                # Persist each directory entry before acknowledging its snapshot.
                _sync_directory(current)
            current = child
        return current

    def on_pre_compress(
        self,
        messages: list[dict[str, Any]],
        *,
        evidence_messages: list[dict[str, Any]] | None = None,
        require_checkpoint: bool = False,
        checkpoint_api_version: int = 2,
    ) -> str:
        if checkpoint_api_version != 2:
            raise RuntimeError("Unsupported Hermes checkpoint API version")
        session_id = self._session_id()
        directory = self._directory(session_id, create=True)
        body = json.dumps(
            {
                "schema": _SCHEMA,
                "session_id": session_id,
                "messages": messages,
                "evidence_messages": evidence_messages,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        checksum = hashlib.sha256(body).hexdigest()
        target = directory / (checksum + ".json")
        _check_path(target)
        if not target.exists():
            temporary = directory / ("." + uuid.uuid4().hex + ".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                # hard-delete: temp -- Unpublished snapshot bytes are disposable; committed archives are retained.
                temporary.unlink(missing_ok=True)
        _sync_directory(directory)
        self.read_checkpoint(session_id, checksum)
        # This is archival evidence, not a summary, authority, or media review.
        return f"ArcReel private pre-compress checkpoint: session={session_id}; SHA256={checksum}."

    def read_checkpoint(self, session_id: str, checksum: str) -> dict[str, Any]:
        """Verify a retained archive before returning its exact pre-compress data."""
        if _DIGEST.fullmatch(checksum) is None:
            raise RuntimeError("Hermes checkpoint digest is invalid")
        target = self._directory(session_id, create=False) / (checksum + ".json")
        _check_path(target)
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            current = target.lstat()
            if (
                not stat.S_ISREG(current.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise RuntimeError("Hermes checkpoint file must be private and regular")
            body = stream.read()
        if hashlib.sha256(body).hexdigest() != checksum:
            raise RuntimeError("Hermes checkpoint readback digest differs")
        snapshot = json.loads(body)
        if snapshot.get("schema") != _SCHEMA or snapshot.get("session_id") != session_id:
            raise RuntimeError("Hermes checkpoint identity differs")
        return snapshot


def install_project_checkpoint(agent: Any, home: Path, memory_manager_type: Any) -> None:
    """Bind a native empty MemoryManager without loading global memory plugins."""
    previous = getattr(agent, "_memory_manager", None)
    if previous is not None and getattr(previous, "providers", ()):
        raise RuntimeError("ArcReel checkpoint cannot replace an active external memory provider")
    checkpoint = TranscriptCheckpoint(home, lambda: agent.session_id)
    manager = memory_manager_type()
    manager.supports_pre_compress_checkpoint = lambda api_version=2: api_version == 2
    manager.on_pre_compress = checkpoint.on_pre_compress
    agent._memory_manager = manager
