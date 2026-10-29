"""Project-native frame observations backed by successful local model requests.

The private Hermes audit is transport evidence, not a media ledger. Current
formal frame pixels and native image versions remain the source of truth.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from lib.artifacts.artifact_manifest import ProjectArtifactManifestAdapter
from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.final_cut.basis import version_snapshot
from lib.generation.generation_result import GenerationAction, GenerationProblem, encode_generation_problem
from lib.infra.content_digest import prefixed_canonical_json_digest
from lib.infra.image_utils import compress_image_bytes, image_pixel_sha256
from lib.local_runtime.config import LocalRuntimeConfig, get_local_runtime_config, local_only_enabled
from lib.project.project_manager import ProjectManager, find_episode
from lib.script.script_models import get_generated_assets
from lib.script.storyboard_sequence import get_storyboard_items

_PROTOCOL = "arcreel-mcp-vision/v2"
_SHA = re.compile(r"[0-9a-f]{64}")
_PROVENANCE = ("source_sha256", "source_pixel_sha256", "wire_image_sha256")


def _image_facts(data: bytes) -> dict[str, str]:
    return {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "source_pixel_sha256": image_pixel_sha256(data),
        "wire_image_sha256": hashlib.sha256(compress_image_bytes(data, max_long_edge=2000, quality=85)).hexdigest(),
    }


def _artifact_image(adapter: ProjectArtifactManifestAdapter, path: object) -> dict[str, str] | None:
    if not isinstance(path, str) or not path:
        return None
    observation = adapter.inspect_artifact_snapshot(path)
    if not observation.present or observation.blocker is not None or observation.content_bytes is None:
        return None
    try:
        return _image_facts(observation.content_bytes)
    except ValueError:
        return None


def _version_candidates(
    directory: Path, adapter: ProjectArtifactManifestAdapter, versions: VersionManager, unit_id: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in versions.get_versions("storyboards", unit_id).get("versions", []):
        number = record.get("version")
        if type(number) is not int or number < 1:
            continue
        found = version_snapshot(directory, versions, "storyboards", unit_id, number)
        if found is None:
            continue
        facts = _artifact_image(adapter, found[0].relative_to(directory).as_posix())
        if facts is not None:
            candidates.append({"resource_type": "storyboard", "id": unit_id, "version": number, **facts})
    return candidates


def _required_frames(projects: ProjectManager, project_name: str, episode_id: int) -> list[dict[str, Any]]:
    project = projects.load_project(project_name)
    episode = find_episode(project, episode_id)
    if episode is None:
        raise ValueError("This episode does not belong to the project")
    script_file = episode.get("script_file")
    if not isinstance(script_file, str) or not script_file:
        return [{"id": None, "role": "script", "problem": "formal_script_missing", "candidates": []}]
    filename = projects.normalize_script_filename(script_file)
    directory = projects.get_project_path(project_name)
    adapter = ProjectArtifactManifestAdapter(directory)
    versions = VersionManager(directory)
    required: list[dict[str, Any]] = []
    with projects.locked_project_script_snapshot(project_name, filename) as (current_project, script):
        current_episode = find_episode(current_project, episode_id)
        if (
            current_episode is None
            or projects.normalize_script_filename(current_episode.get("script_file") or "") != filename
        ):
            return [{"id": None, "role": "script", "problem": "formal_script_binding_changed", "candidates": []}]
        items, id_field, *_ = get_storyboard_items(script)
        for item in items:
            if not isinstance(item, dict) or not item.get(id_field):
                continue
            unit_id = str(item[id_field])
            history = versions.get_versions("storyboards", unit_id)
            candidates = _version_candidates(directory, adapter, versions, unit_id)
            for role, path in (
                ("first", get_generated_assets(item).get("storyboard_image")),
                ("tail", item.get("end_frame_image")),
            ):
                facts = _artifact_image(adapter, path)
                matching = [
                    candidate
                    for candidate in candidates
                    if facts is not None and candidate["source_pixel_sha256"] == facts["source_pixel_sha256"]
                ]
                if role == "first" and facts is not None and not candidates and not history.get("current_version"):
                    matching = [{"resource_type": "storyboard", "id": unit_id, "version": None, **facts}]
                required.append(
                    {
                        "resource_type": "storyboard",
                        "id": unit_id,
                        "role": role,
                        "path": path if isinstance(path, str) else None,
                        "selected_version": history.get("current_version"),
                        **(facts or {}),
                        "candidates": matching,
                        **({"problem": "frame_missing_or_unreadable"} if facts is None else {}),
                        **(
                            {"problem": "frame_has_no_inspectable_version"}
                            if facts is not None and not matching
                            else {}
                        ),
                    }
                )
    return required


def _frozen_frames(
    projects: ProjectManager, project_name: str, resource_id: str, first: Path, tail: Path | None
) -> list[dict[str, Any]]:
    directory = projects.get_project_path(project_name)
    adapter = ProjectArtifactManifestAdapter(directory)
    candidates = _version_candidates(directory, adapter, VersionManager(directory), resource_id)
    required: list[dict[str, Any]] = []
    for role, path in (("first", first), ("tail", tail)):
        locator = path.relative_to(directory).as_posix() if path is not None else None
        facts = _artifact_image(adapter, locator)
        matching = [
            candidate
            for candidate in candidates
            if facts is not None
            and candidate["source_pixel_sha256"] == facts["source_pixel_sha256"]
            and candidate["wire_image_sha256"] == facts["wire_image_sha256"]
        ]
        required.append(
            {
                "resource_type": "storyboard",
                "id": resource_id,
                "role": role,
                "path": locator,
                **(facts or {}),
                "candidates": matching,
                **({"problem": "frozen_frame_missing_or_unreadable"} if facts is None else {}),
            }
        )
    return required


def _event_key(row: dict[str, Any], session_id: str) -> tuple[str, int] | None:
    if row.get("protocol") != _PROTOCOL or row.get("session_id") != session_id or row.get("request_kind") != "main":
        return None
    transport = row.get("transport_id")
    request = row.get("request_id")
    if not isinstance(transport, str) or not transport or type(request) is not int or request < 1:
        return None
    return transport, request


def _successful_observations(path: Path | None, session_id: str, model: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    requests: dict[tuple[str, int], dict[str, Any]] = {}
    responses: dict[tuple[str, int], dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    observers = {session_id}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                # An append may be in progress; only complete events establish evidence.
                continue
            if not isinstance(row, dict):
                continue
            observer = row.get("session_id")
            if (
                row.get("protocol") == _PROTOCOL
                and row.get("event") == "agent_started"
                and row.get("parent_session_id") == session_id
                and isinstance(observer, str)
                and observer
                and observer != session_id
            ):
                observers.add(observer)
                continue
            if not isinstance(observer, str) or observer not in observers or (key := _event_key(row, observer)) is None:
                continue
            event = row.get("event")
            if event == "request":
                if key in requests:
                    requests[key] = {}  # Duplicate identity cannot establish which request was observed.
                else:
                    requests[key] = row
                continue
            if event == "response":
                if key in responses:
                    responses[key] = {}
                else:
                    responses[key] = row
                continue
            if event != "observed_images":
                continue
            request = requests.get(key, {})
            response = responses.get(key, {})
            if request.get("accepted_for_send") is not True:
                continue
            body = request.get("body_sha256")
            if not isinstance(body, str) or _SHA.fullmatch(body) is None:
                continue
            if any(
                record.get("body_sha256") != body or record.get("model") != model for record in (request, response, row)
            ):
                continue
            if any(
                record.get("endpoint_matches_configured") is not True
                or record.get("model_matches_configured") is not True
                for record in (request, row)
            ):
                continue
            status = response.get("status_code")
            if type(status) is not int or not 200 <= status < 300 or row.get("status_code") != status:
                continue
            images = request.get("images")
            rows = row.get("rows")
            if (
                not isinstance(images, list)
                or not isinstance(rows, list)
                or not rows
                or request.get("image_count") != len(images)
            ):
                continue
            hashes = [
                image["sha256"]
                for image in images
                if isinstance(image, dict)
                and isinstance(image.get("sha256"), str)
                and _SHA.fullmatch(image["sha256"]) is not None
            ]
            if len(hashes) != len(images):
                continue
            wire: Counter[str] = Counter(hashes)
            used: Counter[str] = Counter()
            valid: list[dict[str, Any]] = []
            for image in rows:
                if not isinstance(image, dict) or any(
                    not isinstance(image.get(name), str) or _SHA.fullmatch(image[name]) is None for name in _PROVENANCE
                ):
                    continue
                used[image["wire_image_sha256"]] += 1
                valid.append(
                    {
                        **image,
                        "transport_id": key[0],
                        "request_id": key[1],
                        "body_sha256": body,
                        "model": model,
                        "observer_session_id": observer,
                    }
                )
            if valid and not (used - wire):
                observations.extend(valid)
    return observations


def _read_status(
    project_name: str,
    episode_id: int,
    session_id: str,
    projects: ProjectManager,
    config: LocalRuntimeConfig,
    audit_path: Path | None,
    user_id: str,
    frozen: tuple[str, Path, Path | None] | None = None,
) -> dict[str, Any]:
    from server.agent_runtime.hermes_client import recorded_hermes_home

    try:
        required = (
            _frozen_frames(projects, project_name, *frozen)
            if frozen is not None
            else _required_frames(projects, project_name, episode_id)
        )
    except FileNotFoundError:
        required = [{"id": None, "role": "script", "problem": "formal_script_missing", "candidates": []}]
    problem = None
    if audit_path is None:
        try:
            home = recorded_hermes_home(config, session_id, user_id)
            audit_path = home / "vision-wire-audit.jsonl" if home is not None else None
        except (OSError, ValueError, RuntimeError):
            audit_path = None
            problem = "session_binding_unavailable"
    try:
        observed = _successful_observations(audit_path, session_id, config.hermes_model)
    except (OSError, UnicodeError):
        observed = []
        problem = "vision_audit_unreadable"
    missing: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    inspection: dict[int | None, list[str]] = {}
    public_required: list[dict[str, Any]] = []
    for frame in required:
        public = {key: value for key, value in frame.items() if key != "candidates"}
        public_required.append(public)
        found = next(
            (
                receipt
                for receipt in observed
                if any(
                    type(receipt.get("version")) is type(candidate.get("version"))
                    and all(
                        receipt.get(key) == candidate.get(key)
                        for key in ("resource_type", "id", "version", *_PROVENANCE)
                    )
                    for candidate in frame["candidates"]
                )
            ),
            None,
        )
        if found is not None:
            matches.append({"id": frame["id"], "role": frame["role"], **found})
            continue
        missing.append({**public, "problem": frame.get("problem", "frame_not_observed_by_model")})
        for candidate in frame["candidates"][:1]:
            ids = inspection.setdefault(candidate["version"], [])
            if candidate["id"] not in ids:
                ids.append(candidate["id"])
    requests = [
        {
            "resource_type": "storyboard",
            "ids": ids[start : start + 16],
            **({"version": version} if version is not None else {}),
        }
        for version, ids in inspection.items()
        for start in range(0, len(ids), 16)
    ]
    return {
        "enabled": True,
        "ready": bool(required) and not missing,
        "session_id": session_id,
        "required_count": len(required),
        "matched_count": len(matches),
        "required": public_required,
        "missing": missing,
        "inspection_requests": requests,
        "action": "inspect_images" if missing else None,
        "proof_digest": prefixed_canonical_json_digest({"required": public_required, "matches": matches}),
        **({"problem": problem} if problem else {}),
    }


async def read_visual_observation_status(
    project_name: str,
    episode_id: int,
    session_id: str,
    *,
    projects: ProjectManager,
    audit_path: Path | None = None,
    user_id: str = DEFAULT_USER_ID,
    config: LocalRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Verify both current native frame pixels were supplied in a successful main turn.

    ``audit_path`` is an explicit dependency-injection seam for tests. Production
    callers resolve only the user-owned native session binding, never a model path.
    """
    config = config or await asyncio.to_thread(get_local_runtime_config)
    return await asyncio.to_thread(
        _read_status, project_name, episode_id, session_id, projects, config, audit_path, user_id
    )


async def assert_frozen_video_frames_observed(
    project_name: str,
    episode_id: int,
    resource_id: str,
    *,
    first: Path,
    tail: Path | None,
    projects: ProjectManager,
    user_id: str = DEFAULT_USER_ID,
    goal_store: Any = None,
    config: LocalRuntimeConfig | None = None,
    audit_path: Path | None = None,
) -> None:
    """Admit the actual staged frames before a new provider submission.

    Native artifact input claims independently ensure that the formal input has
    not changed. This check proves observation of the staged bytes, without using
    the current canonical frame to authorize a different frozen frame.
    """
    if not local_only_enabled():
        return
    from server.agent_runtime.autonomy_store import ACTIVE_STATES, AutonomyStore

    store = goal_store or AutonomyStore()
    goal = await store.get_latest(project_name, user_id)
    states = ACTIVE_STATES | {"attention"}
    if goal is None or goal.get("state") not in states or goal.get("episode_id") != episode_id:
        return
    session = goal.get("session_id")
    config = config or await asyncio.to_thread(get_local_runtime_config)
    observation = (
        await asyncio.to_thread(
            _read_status,
            project_name,
            episode_id,
            session,
            projects,
            config,
            audit_path,
            user_id,
            (resource_id, first, tail),
        )
        if isinstance(session, str) and session
        else {"ready": False, "missing": [], "inspection_requests": []}
    )
    fresh = await store.get_latest(project_name, user_id)
    unchanged = (
        fresh is not None
        and fresh.get("id") == goal.get("id")
        and fresh.get("session_id") == session
        and fresh.get("episode_id") == episode_id
        and fresh.get("state") in states
    )
    if observation.get("ready") is not True or not unchanged:
        raise ValueError(
            encode_generation_problem(
                GenerationProblem(
                    code="autonomous_visual_observation_required",
                    action=GenerationAction.FIX_INPUT,
                    detail="Inspect the actual frozen first and target-tail image versions before a new video submission.",
                    params={
                        "resource_id": resource_id,
                        "next_action": "inspect_images",
                        "missing": observation.get("missing", []),
                        "inspection_requests": observation.get("inspection_requests", []),
                        "proof_digest": observation.get("proof_digest"),
                    },
                )
            )
        )
