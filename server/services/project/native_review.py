"""Read native review-footage execution facts without interpreting its verdicts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from lib.agent.agent_profile import agent_profile_dir
from lib.agent.profile_manifest import resolve_profile_files_for_mode
from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.final_cut.basis import version_snapshot
from lib.infra.content_digest import prefixed_canonical_json_digest
from lib.local_runtime.config import LocalRuntimeConfig, get_local_runtime_config
from lib.project.project_manager import ProjectManager, find_episode, is_reference_video_project
from lib.script.script_editor import resolve_items

_SCHEMA = "arcreel-native-agent-report/v1"
_AGENT = "review-footage"
_PROFILE_PATH = ".claude/agents/review-footage.md"


def _requirements(
    projects: ProjectManager,
    project_name: str,
    episode_id: int,
    required_versions: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Any]:
    project = projects.load_project(project_name)
    episode = find_episode(project, episode_id)
    if episode is None:
        raise ValueError("This episode does not belong to the project")
    script_file = episode.get("script_file")
    if not isinstance(script_file, str) or not script_file:
        return [{"unit_id": None, "version": None, "problem": "formal_script_missing"}], project["content_mode"]
    filename = projects.normalize_script_filename(script_file)
    directory = projects.get_project_path(project_name)
    versions = VersionManager(directory)
    with ExitStack() as stack:
        try:
            project, script = stack.enter_context(projects.locked_project_script_snapshot(project_name, filename))
        except FileNotFoundError:
            return [{"unit_id": None, "version": None, "problem": "formal_script_missing"}], project["content_mode"]
        episode = find_episode(project, episode_id)
        if episode is None or projects.normalize_script_filename(episode.get("script_file") or "") != filename:
            return [{"unit_id": None, "version": None, "problem": "formal_script_binding_changed"}], project[
                "content_mode"
            ]
        items, id_field, _ = resolve_items(script)
        unit_ids = list(dict.fromkeys(str(item[id_field]) for item in items if item.get(id_field)))
        resource_type = "reference_videos" if is_reference_video_project(project) else "videos"
        requested = (
            required_versions
            if required_versions is not None
            else [
                {"unit_id": unit_id, "version": versions.get_current_version(resource_type, unit_id)}
                for unit_id in unit_ids
            ]
        )
        required: list[dict[str, Any]] = []
        for selection in requested:
            unit_id, version = selection.get("unit_id"), selection.get("version")
            entry = {"unit_id": unit_id, "version": version}
            if not isinstance(unit_id, str) or unit_id not in unit_ids:
                entry["problem"] = "video_unit_not_in_episode"
            elif type(version) is not int or version < 1:
                entry["problem"] = "video_version_missing"
            elif version_snapshot(directory, versions, resource_type, unit_id, version) is None:
                entry["problem"] = "video_version_unavailable"
            if entry not in required:
                required.append(entry)
        return required, project["content_mode"]


def _reports(path: Path | None, session_id: str, model: str, profile_sha256: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    reports = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                # A dispatcher may still be appending; partial records establish no execution fact.
                continue
            if not isinstance(row, dict):
                continue
            report = row.get("report")
            child = row.get("session_id")
            inspections = row.get("inspections")
            if (
                row.get("schema") != _SCHEMA
                or row.get("parent_session_id") != session_id
                or row.get("agent_type") != _AGENT
                or row.get("status") != "completed"
                or row.get("model") != model
                or row.get("profile_sha256") != profile_sha256
                or not isinstance(child, str)
                or not child.strip()
                or child == session_id
                or not isinstance(report, str)
                or not report.strip()
                or not isinstance(inspections, list)
                or not inspections
                or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("unit_id"), str)
                    or not item["unit_id"]
                    or type(item.get("version")) is not int
                    or item["version"] < 1
                    for item in inspections
                )
            ):
                continue
            reports.append(
                {
                    "session_id": child,
                    "inspections": inspections,
                    "profile_sha256": profile_sha256,
                    "report_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
                    "report": report,
                }
            )
    return reports


def _read_status(
    project_name: str,
    episode_id: int,
    session_id: str,
    projects: ProjectManager,
    user_id: str,
    config: LocalRuntimeConfig,
    audit_path: Path | None,
    profile_dir: Path | None,
    required_versions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    from server.agent_runtime.hermes_client import recorded_hermes_home

    required, content_mode = _requirements(projects, project_name, episode_id, required_versions)
    problem = None
    profile_sha256 = None
    reports: list[dict[str, Any]] = []
    try:
        root = profile_dir or agent_profile_dir()
        source = resolve_profile_files_for_mode(root, content_mode)[_PROFILE_PATH]
        profile_sha256 = hashlib.sha256((root / source).read_bytes()).hexdigest()
    except (OSError, ValueError, RuntimeError, KeyError):
        problem = "native_review_profile_unavailable"
    if audit_path is None:
        try:
            home = recorded_hermes_home(config, session_id, user_id)
            audit_path = home / "native-agent-reports.jsonl" if home is not None else None
            if home is None:
                problem = "session_binding_unavailable"
        except (OSError, ValueError, RuntimeError):
            problem = "session_binding_unavailable"
    if profile_sha256 is not None:
        try:
            reports = _reports(audit_path, session_id, config.hermes_model, profile_sha256)
        except (OSError, UnicodeError):
            problem = "native_review_audit_unreadable"
    reviewed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    selected_reports: dict[tuple[str, str], dict[str, Any]] = {}
    for selection in required:
        matches = [
            report
            for report in reports
            if "problem" not in selection
            and any(
                row["unit_id"] == selection["unit_id"] and row["version"] == selection["version"]
                for row in report["inspections"]
            )
        ]
        if not matches:
            missing.append({**selection, "problem": selection.get("problem", "native_review_report_missing")})
        else:
            reviewed.append(
                {
                    **selection,
                    **{key: value for key, value in matches[0].items() if key not in {"inspections", "report"}},
                }
            )
            for report in matches:
                selected_reports.setdefault((report["session_id"], report["report_sha256"]), report)
    return {
        "enabled": True,
        "ready": bool(required) and not missing and problem is None,
        "session_id": session_id,
        "agent_type": _AGENT,
        "required_count": len(required),
        "reviewed_count": len(reviewed),
        "required": required,
        "reviewed": reviewed,
        "reports": list(selected_reports.values()),
        "missing": missing,
        "action": "dispatch_agent" if missing or problem else None,
        "profile_sha256": profile_sha256,
        "proof_digest": prefixed_canonical_json_digest(
            {"required": required, "reviewed": reviewed, "reports": list(selected_reports)}
        ),
        **({"problem": problem} if problem else {}),
    }


async def read_native_review_status(
    project_name: str,
    episode_id: int,
    session_id: str,
    *,
    projects: ProjectManager,
    user_id: str = DEFAULT_USER_ID,
    config: LocalRuntimeConfig | None = None,
    audit_path: Path | None = None,
    profile_dir: Path | None = None,
    required_versions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require completed native reports for selected versions, without claiming quality approval.

    The dispatcher records only inspections whose contact sheets reached its
    child model. Production reads the user-owned parent session's private audit.
    ``audit_path`` and ``profile_dir`` are explicit test dependency seams.
    ``required_versions`` optionally scopes this read to explicit native
    unit/version pairs. It is not a timeline field: native clips consume current
    versions. Omitted, the read covers the episode's current selected versions.
    """
    config = config or await asyncio.to_thread(get_local_runtime_config)
    return await asyncio.to_thread(
        _read_status,
        project_name,
        episode_id,
        session_id,
        projects,
        user_id,
        config,
        audit_path,
        profile_dir,
        required_versions,
    )
