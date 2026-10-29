"""Read native Hermes persisted results without exposing a filesystem tool.

This module is also loaded by path inside the isolated ACP process. Native
pagination, result formatting and redaction are imported only during a read.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_MARKER = ".arcreel-local-hermes"
_FILENAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.txt")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("JSON with repeated object keys uses the text view")
        result[name] = value
    return result


def _native_api() -> Any:
    operations = importlib.import_module("tools.file_operations")
    files = importlib.import_module("tools.file_tools")
    limits = importlib.import_module("tools.tool_output_limits")
    redact = importlib.import_module("agent.redact")
    # Only the native pure formatter is used, never a terminal environment.
    formatter = object.__new__(operations.ShellFileOperations)
    return SimpleNamespace(
        normalize=operations.normalize_read_pagination,
        result=operations.ReadResult,
        render=formatter._add_line_numbers,
        redact=redact.redact_sensitive_text,
        max_chars=files._get_max_read_chars(),
        max_line_length=limits.get_max_line_length(),
    )


def _no_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("Native persisted-result paths cannot contain symlinks")


def _read_regular(path: Path) -> bytes:
    _no_symlinks(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened, current = os.fstat(stream.fileno()), path.lstat()
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode):
            raise ValueError("Native persisted results must be regular files")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Native persisted-result identity changed while opening")
        _no_symlinks(path)
        return stream.read()


class NativeSpillReader:
    """Expose ``read_file`` only for one ArcReel-owned session's spill cache."""

    def __init__(self, home: Path, *, native_api: Any = None) -> None:
        self.home = Path(home)
        if not self.home.is_absolute() or self.home == Path.home() / ".hermes":
            raise ValueError("Native persisted-result home must be ArcReel-owned")
        _no_symlinks(self.home)
        if _read_regular(self.home / _MARKER) != b"arcreel-local-hermes/v1\n":
            raise ValueError("Native persisted-result home has no ArcReel ownership marker")
        self.home = self.home.resolve(strict=True)
        self.root = self.home / "cache" / "spillover"
        self.native_api = native_api

    def _path(self, path: object) -> Path:
        if not isinstance(path, str) or not path or any(char in path for char in ("\\", "\0", "~")):
            raise ValueError("Native persisted-result path is invalid")
        candidate = Path(path)
        if ".." in candidate.parts:
            raise ValueError("Native persisted-result path cannot traverse directories")
        if not candidate.is_absolute():
            candidate = self.home / candidate
        if candidate.parent != self.root or _FILENAME.fullmatch(candidate.name) is None:
            raise ValueError("read_file is limited to the current session's cache/spillover text results")
        return candidate

    def read_file(self, path: object, offset: Any = 1, limit: Any = 2000, task_id: str = "default") -> str:
        """Return native line pagination over a lossless, explicitly labelled view.

        JSON is pretty-printed and long lines are wrapped before native line
        formatting, so the native per-line clamp cannot silently discard the
        remainder of a minified result. Offsets refer to this deterministic
        view; the persisted source is never modified or re-requested.
        """
        del task_id
        target = self._path(path)
        # Revalidate ownership on every invocation, including cold reconnects.
        if _read_regular(self.home / _MARKER) != b"arcreel-local-hermes/v1\n":
            raise ValueError("Native persisted-result home ownership changed")
        raw = _read_regular(target)
        text = raw.decode("utf-8")
        if "\0" in text:
            raise ValueError("Native persisted-result content must be text")
        api = self.native_api or _native_api()
        offset, limit = api.normalize(offset, limit)
        kind = "text"
        try:
            parsed = json.loads(text, object_pairs_hook=_unique_object)
        except ValueError:
            pass
        else:
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
            kind = "pretty-json"
        # Redact before wrapping, so splitting cannot hide a token from the
        # native sensitive-text matcher at a page or line boundary.
        text = api.redact(text, file_read=True)
        max_chars = int(api.max_chars)
        width = min(int(api.max_line_length), max_chars - 32)
        if width < 1:
            raise RuntimeError("Native read_file character budget cannot fit a numbered line")
        lines: list[str] = []
        wrapped = False
        for line in text.split("\n") if text else []:
            wrapped = wrapped or len(line) > width
            lines.extend([line[index : index + width] for index in range(0, len(line), width)] or [""])
        selected: list[str] = []
        page_chars = 0
        for number, line in enumerate(lines[offset - 1 : offset - 1 + limit], start=offset):
            size = len(str(number)) + 1 + len(line) + bool(selected)
            if page_chars + size > max_chars:
                break
            selected.append(line)
            page_chars += size
        next_offset = offset + len(selected)
        truncated = next_offset <= len(lines)
        hint = "offset and limit address the stable display view, not physical source lines."
        if truncated:
            hint += f" Use offset={next_offset} to continue."
        elif offset > len(lines):
            hint += f" Offset {offset} is beyond the end ({len(lines)} view lines)."
        result = api.result(
            content=api.render("\n".join(selected), offset) if selected else "",
            total_lines=len(lines),
            file_size=len(raw),
            truncated=truncated,
            hint=hint,
        ).to_dict()
        result.update(
            path=str(target),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            view_sha256=hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
            view=kind + ("+wrapped-lines" if wrapped else ""),
        )
        if truncated:
            result["next_offset"] = next_offset
        return json.dumps(result, ensure_ascii=False)
