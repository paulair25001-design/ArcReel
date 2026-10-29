"""Verify native finalization occurred after the parent's real production work."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

from lib.db.base import DEFAULT_USER_ID
from lib.infra.content_digest import prefixed_canonical_json_digest
from lib.local_runtime.config import LocalRuntimeConfig, get_local_runtime_config

_PROTOCOL = "arcreel-mcp-vision/v2"
_REPORT_SCHEMA = "arcreel-native-agent-report/v1"
_SHA = re.compile(r"[a-f0-9]{64}")


def _events(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                # In-progress append bytes cannot acknowledge an action.
                continue
            if isinstance(row, dict):
                result.append(row)
    return result


def _timestamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _read_status(
    project_name: str,
    session_id: str,
    user_id: str,
    config: LocalRuntimeConfig,
    audit_path: Path | None,
    reports_path: Path | None,
    not_before: object,
) -> dict[str, Any]:
    from server.agent_runtime.hermes_client import recorded_hermes_home

    problem = None
    cutoff = None
    if not_before is not None:
        if (
            isinstance(not_before, bool)
            or not isinstance(not_before, (int, float))
            or not math.isfinite(not_before)
            or not_before < 0
        ):
            problem = "native_delivery_not_before_invalid"
        else:
            cutoff = float(not_before)
    if audit_path is None or reports_path is None:
        try:
            home = recorded_hermes_home(config, session_id, user_id)
            if home is None:
                problem = "session_binding_unavailable"
            else:
                audit_path = audit_path or home / "vision-wire-audit.jsonl"
                reports_path = reports_path or home / "native-agent-reports.jsonl"
        except (OSError, ValueError, RuntimeError):
            problem = "session_binding_unavailable"
    review_times: list[float] = []
    write_times: list[float] = []
    finalizations: list[dict[str, Any]] = []
    try:
        reports, events = _events(reports_path), _events(audit_path)
    except (OSError, UnicodeError):
        reports, events = [], []
        problem = "native_delivery_audit_unreadable"
    for row in reports:
        if (
            row.get("schema") != _REPORT_SCHEMA
            or row.get("project") != project_name
            or row.get("parent_session_id") != session_id
            or row.get("agent_type") != "review-footage"
            or row.get("status") != "completed"
            or row.get("model") != config.hermes_model
            or not isinstance(row.get("report"), str)
            or not row["report"].strip()
        ):
            continue
        stamp = _timestamp(row.get("timestamp"))
        if stamp is None:
            problem = "native_review_timestamp_invalid"
        else:
            review_times.append(stamp)
    for row in events:
        if row.get("protocol") != _PROTOCOL or row.get("project_name") != project_name:
            continue
        parent = row.get("parent_session_id")
        direct = row.get("session_id") == session_id and parent is None
        child = parent == session_id and isinstance(row.get("session_id"), str) and row["session_id"] != session_id
        event = row.get("event")
        if event == "native_write_success" and (direct or child):
            if row.get("tool_name") == "finish_local_session":
                continue
        elif event == "native_finalization" and direct:
            if row.get("tool_name") != "finish_local_session" or row.get("original_service_restored") is not True:
                continue
        else:
            continue
        stamp = _timestamp(row.get("timestamp"))
        source = row.get("source_sha256")
        if stamp is None or not isinstance(source, str) or _SHA.fullmatch(source) is None:
            problem = "native_delivery_event_invalid"
            continue
        if event == "native_write_success":
            write_times.append(stamp)
        else:
            finalizations.append({"timestamp": stamp, "source_sha256": source, "tool_name": "finish_local_session"})
    last_review = max(review_times, default=None)
    last_write = max(write_times, default=None)
    finalization = max(finalizations, key=lambda row: row["timestamp"], default=None)
    ready = (
        problem is None
        and last_review is not None
        and finalization is not None
        and finalization["timestamp"] > last_review
        and (last_write is None or finalization["timestamp"] > last_write)
        and (cutoff is None or finalization["timestamp"] > cutoff)
    )
    if not ready and problem is None:
        if last_review is None:
            problem = "native_review_report_missing"
        elif finalization is None:
            problem = "native_finalization_missing"
        else:
            problem = "native_finalization_outdated"
    facts = {
        "session_id": session_id,
        "project_name": project_name,
        "not_before": cutoff,
        "last_review_timestamp": last_review,
        "last_write_timestamp": last_write,
        "finalization": finalization,
    }
    return {
        "enabled": True,
        "ready": ready,
        **facts,
        "action": None if ready else "finish_local_session",
        "proof_digest": prefixed_canonical_json_digest(facts),
        **({"problem": problem} if problem else {}),
    }


async def read_native_delivery_status(
    project_name: str,
    session_id: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    config: LocalRuntimeConfig | None = None,
    audit_path: Path | None = None,
    reports_path: Path | None = None,
    not_before: float | None = None,
) -> dict[str, Any]:
    """Require parent finalization after native review and successful writes.

    The producer records only actual successful tool results; model prose and
    invented batch handles cannot establish delivery. Resource closure and
    artifact currency retain their existing native checks. Explicit audit paths
    are dependency-injection seams for isolated tests, never model arguments.
    ``not_before`` is the new goal's creation time when it reuses an owned SID;
    earlier reports remain usable, but earlier finalization cannot finish it.
    """
    config = config or await asyncio.to_thread(get_local_runtime_config)
    return await asyncio.to_thread(
        _read_status, project_name, session_id, user_id, config, audit_path, reports_path, not_before
    )
