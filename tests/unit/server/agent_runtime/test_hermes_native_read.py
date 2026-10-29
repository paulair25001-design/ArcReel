"""Native spill recovery remains scoped and can paginate minified output."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent_runtime.hermes_native_read import NativeSpillReader


class NativeResult:
    def __init__(self, **fields):
        self.fields = fields

    def to_dict(self):
        return self.fields


def _api(*, max_chars=180, max_line_length=80):
    return SimpleNamespace(
        normalize=lambda offset, limit: (max(1, int(offset)), min(2000, max(1, int(limit)))),
        result=NativeResult,
        render=lambda text, start: "\n".join(f"{number}|{line}" for number, line in enumerate(text.split("\n"), start)),
        redact=lambda text, **kwargs: text.replace("secret-token", "[REDACTED]"),
        max_chars=max_chars,
        max_line_length=max_line_length,
    )


def _home(tmp_path):
    home = tmp_path / "owned-session"
    (home / "cache/spillover").mkdir(parents=True)
    (home / ".arcreel-local-hermes").write_text("arcreel-local-hermes/v1\n")
    return home


def _body(page):
    return [line.split("|", 1)[1] for line in page["content"].split("\n")] if page["content"] else []


def test_native_line_pagination_and_cold_reconnect_keep_source_bytes(tmp_path):
    home = _home(tmp_path)
    path = home / "cache/spillover/call_123.txt"
    raw = b"one\ntwo\nthree\nfour"
    path.write_bytes(raw)
    reader = NativeSpillReader(home, native_api=_api())
    first = json.loads(reader.read_file(str(path), offset=2, limit=1))
    assert first["content"] == "2|two"
    assert first["total_lines"] == 4
    assert first["next_offset"] == 3
    assert first["source_sha256"] == hashlib.sha256(raw).hexdigest()
    second = json.loads(NativeSpillReader(home, native_api=_api()).read_file("cache/spillover/call_123.txt", 3, 2))
    assert second["content"] == "3|three\n4|four"
    assert not second["truncated"]
    assert first["view_sha256"] == second["view_sha256"]
    assert path.read_bytes() == raw


def test_minified_json_and_long_string_page_without_lost_business_fields_or_respill(tmp_path):
    home = _home(tmp_path)
    path = home / "cache/spillover/call.big-result.txt"
    payload = {"units": [{"id": "E2S02", "text": "完整中文剧情" * 100}], "structuredContent": {"still_here": "END"}}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(raw)
    reader = NativeSpillReader(home, native_api=_api())
    offset, recovered, sources, views = 1, [], set(), set()
    for _ in range(100):
        page = json.loads(reader.read_file(str(path), offset, 2000))
        recovered.extend(_body(page))
        sources.add(page["source_sha256"])
        views.add(page["view_sha256"])
        assert len(page["content"]) <= 180
        assert "[truncated]" not in page["content"]
        assert "persisted-output" not in page["content"]
        assert page["view"] == "pretty-json+wrapped-lines"
        if not page["truncated"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    else:
        pytest.fail("Read view failed to make forward progress")
    reconstructed = "".join(recovered)
    assert reconstructed == json.dumps(payload, ensure_ascii=False, indent=2).replace("\n", "")
    assert sources == {hashlib.sha256(raw).hexdigest()}
    assert views == {hashlib.sha256("\n".join(recovered).encode()).hexdigest()}
    assert list((home / "cache/spillover").iterdir()) == [path]
    assert path.read_bytes() == raw


def test_repeated_json_keys_are_preserved_as_wrapped_text(tmp_path):
    home = _home(tmp_path)
    path = home / "cache/spillover/call_dup.txt"
    raw = '{"same":"first", "same":"second"}'
    path.write_text(raw)
    page = json.loads(NativeSpillReader(home, native_api=_api()).read_file(str(path)))
    assert page["view"] == "text"
    assert "".join(_body(page)) == raw


def test_redaction_precedes_wrapping_and_out_of_range_has_no_loop_cursor(tmp_path):
    home = _home(tmp_path)
    path = home / "cache/spillover/call_secret.txt"
    path.write_text("x" * 75 + "secret-token" + "END")
    reader = NativeSpillReader(home, native_api=_api())
    page = json.loads(reader.read_file(str(path)))
    assert "secret-token" not in page["content"]
    assert "[REDACTED]" in "".join(_body(page))
    end = json.loads(reader.read_file(str(path), 999, 2000))
    assert end["content"] == ""
    assert not end["truncated"]
    assert "next_offset" not in end


@pytest.mark.parametrize(
    "bad",
    [
        "config.yaml",
        "cache/other.txt",
        "cache/spillover/../token.txt",
        "cache/spillover/child/call.txt",
        "cache/spillover/.env",
        "cache\\spillover\\call.txt",
        "~/cache/spillover/call.txt",
        "/etc/passwd",
        "https://external.invalid/call.txt",
        None,
    ],
)
def test_other_files_are_refused_before_native_reader_is_loaded(tmp_path, bad):
    reader = NativeSpillReader(_home(tmp_path))
    with pytest.raises(ValueError, match=r"path|limited|traverse"):
        reader.read_file(bad)


@pytest.mark.parametrize("target", ["file", "cache", "marker", "home"])
def test_symlinks_at_every_owned_boundary_are_refused(tmp_path, target):
    home = _home(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private-unrelated")
    path = home / "cache/spillover/call.txt"
    if target == "home":
        linked = tmp_path / "linked"
        linked.symlink_to(home, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            NativeSpillReader(linked, native_api=_api())
        return
    reader = NativeSpillReader(home, native_api=_api())
    if target == "marker":
        marker = home / ".arcreel-local-hermes"
        marker.unlink()
        marker.symlink_to(outside)
    elif target == "cache":
        (home / "cache/spillover").rmdir()
        (home / "cache").rmdir()
        (home / "cache").symlink_to(tmp_path, target_is_directory=True)
    else:
        path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        reader.read_file(str(path))


def test_unowned_home_wrong_marker_fifo_and_cross_session_are_refused(tmp_path):
    home = _home(tmp_path)
    reader = NativeSpillReader(home, native_api=_api())
    other = tmp_path / "other-session"
    other.mkdir()
    (other / "call.txt").write_text("unrelated")
    with pytest.raises(ValueError, match="limited"):
        reader.read_file(str(other / "call.txt"))
    fifo = home / "cache/spillover/call_fifo.txt"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        reader.read_file(str(fifo))
    (home / ".arcreel-local-hermes").write_text("another-owner")
    with pytest.raises(ValueError, match="ownership"):
        NativeSpillReader(home, native_api=_api())
    with pytest.raises(ValueError, match="ownership"):
        reader.read_file(str(fifo))


def test_spill_reader_loads_by_file_without_application_or_native_imports():
    path = Path(__file__).resolve().parents[4] / "server/agent_runtime/hermes_native_read.py"
    spec = importlib.util.spec_from_file_location("standalone_arcreel_spill", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.NativeSpillReader.__module__ == "standalone_arcreel_spill"
