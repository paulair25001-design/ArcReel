"""Read production facts from formal artifacts, native versions and the queue.

This is a projection, not another production ledger. A completed provider job or
available video clip does not establish delivery of the episode's final cut.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from lib.artifacts.artifact_currency import active_artifact_currency_resolver
from lib.artifacts.artifact_manifest import ArtifactKey, ArtifactStatus, ProjectArtifactManifestAdapter
from lib.artifacts.version_manager import VersionManager
from lib.db.base import DEFAULT_USER_ID
from lib.edit_timeline.service import EditTimelineService
from lib.edit_timeline.sources import load_episode_script_units
from lib.edit_timeline.store import EditTimelineStore
from lib.episode.episode_paths import episode_source_relpath
from lib.final_cut.basis import BURNED_SUBTITLES, WITH_NARRATION
from lib.final_cut.service import FinalCutService
from lib.generation.generation_queue import GenerationQueue, get_generation_queue
from lib.generation.generation_result import decode_generation_problem, provider_checkpoint_from_task
from lib.generation.task_failure import parse_failure
from lib.infra.api_errors import ApiError
from lib.infra.content_digest import prefixed_canonical_json_digest
from lib.infra.logging_utils import redact_diagnostic_text
from lib.infra.media_probe import MediaProbe, probe_media
from lib.infra.path_safety import safe_join
from lib.local_runtime.l20_direct_config import get_l20_direct_config
from lib.project.asset_types import ASSET_SPECS
from lib.project.project_manager import ProjectManager, find_episode, get_project_manager
from lib.script import script_review
from lib.script.draft_quarantine import (
    QUARANTINE_KIND_PROMPT_AUTHORING,
    draft_revision,
    quarantine_path,
    read_quarantine,
)
from lib.script.script_editor import resolve_items
from lib.script.script_models import get_generated_assets
from lib.script.storyboard_sequence import get_storyboard_items
from lib.workflow.workflow_state import WorkflowStateService


class ProductionEpisodeNotFound(ValueError):
    """An episode identity must exist in this project's formal episode index."""


def _authoring_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _authoring_content(child)
            for key, child in value.items()
            if key not in {"generated_assets", "end_frame_image", "metadata"}
        }
    if isinstance(value, list):
        return [_authoring_content(child) for child in value]
    return value


def _preparation_facts(
    projects: ProjectManager,
    project_name: str,
    episode_id: int,
    project_dir: Path,
    project: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    """Use native source, plan, confirmation and draft identities, without a ledger."""
    status = WorkflowStateService(projects).get_status(project_name, episode_id)
    adapter = ProjectArtifactManifestAdapter(project_dir)
    source = _file_facts(adapter, episode_source_relpath(episode_id))
    plan_path = script_review.script_plan_path(project_dir, project, episode_id)
    plan = {
        "path": plan_path.relative_to(project_dir).as_posix() if plan_path is not None else None,
        "fingerprint": script_review.content_fingerprint(plan_path) if plan_path is not None else None,
        "review_state": script_review.review_status(project_dir, project, episode_id),
        "confirmed_fingerprint": script_review.stored_review(project, episode_id).get("fingerprint"),
    }
    kinds = [kind for kind in [script_review.script_plan_quarantine_kind(project)] if kind is not None]
    if project.get("generation_mode") == "reference_video":
        kinds.append(QUARANTINE_KIND_PROMPT_AUTHORING)
    drafts: list[dict[str, Any]] = []
    for kind in kinds:
        path = quarantine_path(project_dir, episode_id, kind).relative_to(project_dir).as_posix()
        facts = _file_facts(adapter, path)
        if facts["present"]:
            draft = read_quarantine(project_dir, episode_id, kind)
            drafts.append(
                {
                    "kind": kind,
                    "path": path,
                    "revision": draft_revision(draft) if draft is not None else facts["sha256"],
                    "needs_repair": bool(draft.violations) if draft is not None else True,
                }
            )
    formal_fingerprint = None
    if isinstance(episode.get("script_file"), str) and episode["script_file"]:
        filename = projects.normalize_script_filename(episode["script_file"])
        observation = adapter.inspect_artifact_snapshot("scripts/" + filename)
        if observation.present and observation.blocker is None and observation.content_bytes is not None:
            formal_fingerprint = prefixed_canonical_json_digest(
                _authoring_content(json.loads(observation.content_bytes))
            )
    preparation = {
        "source_revision": status.source_revision,
        "episode_source": {key: source[key] for key in ("path", "present", "sha256")},
        "script_plan": plan,
        "drafts": drafts,
        "formal_authoring_fingerprint": formal_fingerprint,
    }
    preparation["fingerprint"] = prefixed_canonical_json_digest(preparation)
    content = status.content
    return {
        "preparation": preparation,
        "workflow": {
            "next_action": status.next_action.model_dump(mode="json"),
            "blockers": [blocker.model_dump(mode="json") for blocker in status.blockers],
            "pending_authoring_ids": list(content.pending_authoring_ids) if content is not None else [],
            "needs_replan_ids": list(content.needs_replan_ids) if content is not None else [],
        },
    }


def _file_facts(adapter: ProjectArtifactManifestAdapter, path: object) -> dict[str, Any]:
    if not isinstance(path, str) or not path:
        return {"path": None, "present": False, "sha256": None}
    observation = adapter.inspect_artifact_content(path)
    return {
        "path": path,
        "present": observation.present and observation.blocker is None,
        "sha256": observation.content_digest,
        **({"problem": observation.blocker.code} if observation.blocker else {}),
    }


def _wav_duration(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / source.getframerate()
    return None


def _media_facts(
    *,
    project_dir: Path,
    adapter: ProjectArtifactManifestAdapter,
    resolver: Any,
    versions: VersionManager,
    resource_type: str,
    unit_id: str,
    key: ArtifactKey,
    path: object,
    audio: bool | None,
) -> dict[str, Any]:
    facts = _file_facts(adapter, path)
    comparison = resolver.compare(key, artifact_path=path) if isinstance(path, str) and path else None
    history = versions.get_versions(resource_type, unit_id)
    records: list[dict[str, Any]] = []
    for record in history["versions"]:
        try:
            snapshot = VersionManager.resolve_snapshot_path(project_dir, resource_type, record.get("file"))
            item = _file_facts(adapter, snapshot.relative_to(project_dir).as_posix())
        except (ValueError, OSError, ApiError):
            item = {"path": None, "present": False, "sha256": None, "problem": "version_snapshot_unavailable"}
        records.append({"version": record["version"], "selected": bool(record.get("is_current")), **item})
    result = {
        **facts,
        "status": comparison.status.value if comparison else "missing",
        "current": comparison is not None and comparison.status is ArtifactStatus.CURRENT,
        "selected_version": history["current_version"],
        "versions": records,
    }
    if audio is not None:
        result["actual_seconds"] = None
        if facts["present"]:
            try:
                result["actual_seconds"] = _wav_duration(safe_join(project_dir, str(path))) if audio else None
            except (OSError, ValueError, wave.Error):
                result["duration_problem"] = "media_unreadable"
    return result


def _resource_facts(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"owner_project": None, "state": "not_started", "active": False, "original_service_restored": None}
    try:
        record = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("resource record is not an object")
        state = str(record.get("state") or "unknown")
        return {
            "owner_project": record.get("project"),
            "state": state,
            "active": state != "restored",
            "original_service_restored": record.get("original_service_restored") is True,
        }
    except (OSError, ValueError):
        return {"owner_project": None, "state": "unreadable", "active": None, "original_service_restored": None}


def _read_episode(
    project_name: str,
    episode_id: int,
    projects: ProjectManager,
    resource_state_path: Path,
) -> tuple[dict[str, Any], list[Any]]:
    if type(episode_id) is not int or episode_id < 1:
        raise ValueError("episode_id must be a positive integer")
    project = projects.load_project(project_name)
    project_dir = projects.get_project_path(project_name)
    episode = find_episode(project, episode_id)
    if episode is None:
        raise ProductionEpisodeNotFound("This episode does not belong to the project")
    base = {
        "project": project_name,
        "episode_id": episode_id,
        "script_state": "missing",
        "script_file": None,
        "script_sha256": None,
        "units": [],
        "referenced_asset_ids": [],
        "assets": [],
        "resource": _resource_facts(resource_state_path),
        **_preparation_facts(projects, project_name, episode_id, project_dir, project, episode),
    }
    documents = EditTimelineStore(projects, project_name).list_documents(episode_id, strict=True)
    if not isinstance(episode.get("script_file"), str) or not episode["script_file"]:
        return base, documents
    script_file = projects.normalize_script_filename(episode["script_file"])
    try:
        script = projects.load_script_readonly(project_name, script_file)
    except FileNotFoundError:
        return {**base, "script_file": script_file}, documents
    items, id_field, kind = resolve_items(script)
    scripted = load_episode_script_units(projects, project_name, episode_id)
    by_id = {str(item[id_field]): item for item in items if isinstance(item, dict) and item.get(id_field)}
    adapter = ProjectArtifactManifestAdapter(project_dir)
    resolver = active_artifact_currency_resolver(project_dir, project)
    versions = VersionManager(project_dir)
    units: list[dict[str, Any]] = []
    video_type = "reference_videos" if kind == "video_units" else "videos"
    for unit in scripted.units:
        item = by_id[unit.unit_id]
        assets = get_generated_assets(item)
        common = {
            "project_dir": project_dir,
            "adapter": adapter,
            "resolver": resolver,
            "versions": versions,
            "unit_id": unit.unit_id,
        }
        units.append(
            {
                "unit_id": unit.unit_id,
                "text": unit.subtitle_text,
                "declared_seconds": unit.scripted_duration_us / 1_000_000,
                "audio": _media_facts(
                    **common,
                    resource_type="audio",
                    key=ArtifactKey.episode_audio(episode_id, unit.unit_id),
                    path=assets.get("narration_audio"),
                    audio=True,
                ),
                "first": _media_facts(
                    **common,
                    resource_type="storyboards",
                    key=ArtifactKey.episode_storyboard(episode_id, unit.unit_id),
                    path=assets.get("storyboard_image"),
                    audio=None,
                ),
                "end_frame": _file_facts(adapter, item.get("end_frame_image")),
                "video": _media_facts(
                    **common,
                    resource_type=video_type,
                    key=ArtifactKey.episode_video(episode_id, unit.unit_id),
                    path=assets.get("video_clip"),
                    audio=False,
                ),
            }
        )
    documents = EditTimelineStore(projects, project_name).list_documents(episode_id, strict=True)
    asset_ids: set[str] = set()
    if kind != "video_units":
        _, _, char_field, scene_field, prop_field = get_storyboard_items(script)
        for item in items:
            for field in (char_field, scene_field, prop_field, "products_in_shot"):
                value = item.get(field) if field else None
                if isinstance(value, list):
                    asset_ids.update(name for name in value if isinstance(name, str))
    asset_facts: list[dict[str, Any]] = []
    for asset_type, spec in ASSET_SPECS.items():
        for name, asset in (project.get(spec.bucket_key) or {}).items():
            if name in asset_ids and isinstance(asset, dict):
                facts = _media_facts(
                    project_dir=project_dir,
                    adapter=adapter,
                    resolver=resolver,
                    versions=versions,
                    resource_type=spec.bucket_key,
                    unit_id=name,
                    key=ArtifactKey.asset_sheet(asset_type, name),
                    path=asset.get(spec.sheet_field),
                    audio=None,
                )
                asset_facts.append({"resource_type": asset_type, "resource_id": name, **facts})
    return {
        "preparation": base["preparation"],
        "workflow": base["workflow"],
        "project": project_name,
        "episode_id": episode_id,
        "script_file": script_file,
        "script_state": resolver.compare(
            ArtifactKey.episode_script(episode_id), artifact_path="scripts/" + script_file
        ).status.value,
        "script_sha256": hashlib.sha256((project_dir / "scripts" / script_file).read_bytes()).hexdigest(),
        "units": units,
        "referenced_asset_ids": sorted(asset_ids),
        "assets": asset_facts,
        "resource": _resource_facts(resource_state_path),
    }, documents


def narration_delivery_complete(units: list[dict[str, Any]], timeline: dict[str, Any]) -> bool:
    """A current render may retain warnings; an autonomous delivery may not lose speech."""
    readout = timeline.get("readout")
    if not isinstance(readout, dict):
        return False
    clips = readout.get("clips", [])
    if not clips or any(clip.get("source_volume") != 0 for clip in clips):
        return False
    if any(
        issue.get("code") in {"narration_overrun", "narration_source_collision", "narration_missing"}
        for issue in readout.get("issues", [])
    ):
        return False
    total = readout.get("duration")
    if not isinstance(total, (int, float)) or total <= 0:
        return False
    for unit in units:
        audio = unit.get("audio", {})
        duration = audio.get("actual_seconds")
        if audio.get("status") != "current" or not isinstance(duration, (int, float)) or duration <= 0:
            return False
        carriers = [clip for clip in clips if clip.get("unit_id") == unit["unit_id"] and clip.get("carries_narration")]
        if len(carriers) != 1:
            return False
        carrier = carriers[0]
        narration = carrier.get("narration")
        if carrier.get("status") != "ready" or not isinstance(narration, dict):
            return False
        start, end = narration.get("start"), narration.get("end")
        clip_start, clip_duration = carrier.get("start"), carrier.get("duration")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not isinstance(clip_start, (int, float))
            or not isinstance(clip_duration, (int, float))
        ):
            return False
        if not all(math.isfinite(value) for value in (start, end, clip_start, clip_duration, duration, total)):
            return False
        if abs((end - start) - duration) > 0.001 or end > total + 0.001 or end > clip_start + clip_duration + 0.001:
            return False
    return bool(units)


def episode_final_cut_complete(episode_id: int, units: list[dict[str, Any]], timelines: list[dict[str, Any]]) -> bool:
    """Require the specified episode's complete ordered units and the exact delivery variant."""
    unit_ids = [unit["unit_id"] for unit in units]
    for timeline in timelines:
        final = timeline.get("final_cut", {})
        if (
            unit_ids
            and timeline.get("episode_id") == episode_id
            and list(dict.fromkeys(timeline.get("unit_ids", []))) == unit_ids
            and final.get("episode") == episode_id
            and final.get("timeline_id") == timeline.get("timeline_id")
            and final.get("status") == "current"
            and final.get("narration") == WITH_NARRATION
            and final.get("subtitles") == BURNED_SUBTITLES
            and final.get("version") is not None
            and narration_delivery_complete(units, timeline)
        ):
            return True
    return False


def _task_facts(task: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: task.get(key)
        for key in (
            "task_id",
            "batch_id",
            "task_type",
            "resource_id",
            "resource_type",
            "script_file",
            "status",
            "provider_id",
            "provider_job_id",
            "dependency_task_id",
        )
    }
    checkpoint = provider_checkpoint_from_task(task)
    if checkpoint is not None:
        result["provider_checkpoint"] = {"submitted": checkpoint.submitted}
    if message := task.get("error_message"):
        failure = parse_failure(message)
        result["error_code"] = failure[0] if failure is not None else None
        result["error_message"] = redact_diagnostic_text(message)
        problem = decode_generation_problem(message)
        if problem is not None:
            result["problem"] = json.loads(redact_diagnostic_text(json.dumps(problem.model_dump(mode="json"))))
        elif failure is not None:
            result["problem"] = json.loads(
                redact_diagnostic_text(json.dumps({"code": failure[0], "params": failure[1]}))
            )
    return result


def _unresolved_failure(task: dict[str, Any], result: dict[str, Any], timeline_ids: set[str]) -> bool:
    """History alone is not a new gap; report only this episode's unresolved output."""
    resource_id = task.get("resource_id")
    unit = next((unit for unit in result["units"] if unit["unit_id"] == resource_id), None)
    asset = next((asset for asset in result["assets"] if asset["resource_id"] == resource_id), None)
    task_type = str(task.get("task_type") or "")
    media = {"tts": "audio", "storyboard": "first", "video": "video", "reference_video": "video"}.get(task_type)
    if unit is not None and media is not None:
        return unit[media]["status"] not in {"current", "stale"}
    if task_type == "image_edit":
        return (
            (unit["first"]["status"] not in {"current", "stale"})
            if unit is not None
            else (asset is not None and asset["status"] not in {"current", "stale"})
        )
    if asset is not None:
        return asset["status"] not in {"current", "stale"}
    if (task.get("payload") or {}).get("timeline_id") in timeline_ids:
        return True
    if task_type == "text_episode_script" and resource_id == f"episode-{result['episode_id']}":
        return result["script_state"] not in {"current", "stale"}
    script = result["script_file"]
    return script is not None and task.get("script_file") in {script, "scripts/" + script}


async def read_episode_production_status(
    project_name: str,
    episode_id: int,
    *,
    user_id: str = DEFAULT_USER_ID,
    projects: ProjectManager | None = None,
    queue: GenerationQueue | None = None,
    resource_state_path: Path | None = None,
    media_probe: Callable[[Path], Awaitable[MediaProbe]] = probe_media,
) -> dict[str, Any]:
    """Only a current narrated, subtitled final cut covering this episode completes it."""
    projects = projects or await asyncio.to_thread(get_project_manager)
    queue = queue or get_generation_queue()
    state_path = resource_state_path or await asyncio.to_thread(
        lambda: get_l20_direct_config().state_dir / "resource-session.json"
    )
    result, documents = await asyncio.to_thread(_read_episode, project_name, episode_id, projects, state_path)
    active: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_failures: set[tuple[Any, ...]] = set()
    for status in ("queued", "running", "failed"):
        page = 1
        while True:
            listed = await queue.list_tasks(project_name=project_name, status=status, page=page, page_size=500)
            tasks = [task for task in listed["items"] if task.get("user_id", DEFAULT_USER_ID) == user_id]
            if status == "failed":
                timeline_ids = {document.id for document in documents}
                for task in tasks:
                    key = (task.get("task_type"), task.get("resource_type"), task.get("resource_id"))
                    if key in seen_failures:
                        continue
                    seen_failures.add(key)
                    if _unresolved_failure(task, result, timeline_ids):
                        failed.append(
                            {
                                **_task_facts(task),
                                "target_episode_id": episode_id,
                                "relevance": "unresolved_episode_output",
                            }
                        )
            else:
                active.extend(_task_facts(task) for task in tasks)
            if page * listed["page_size"] >= listed["total"]:
                break
            page += 1
    project_dir = await asyncio.to_thread(projects.get_project_path, project_name)
    for unit in result["units"]:
        for kind in ("audio", "video"):
            media = unit[kind]
            if media["present"] and media["actual_seconds"] is None:
                try:
                    path = await asyncio.to_thread(safe_join, project_dir, media["path"])
                    stream = (await media_probe(path)).first_stream(kind)
                    media["actual_seconds"] = stream.duration_seconds if stream is not None else None
                except (OSError, ValueError):
                    media["duration_problem"] = "media_unreadable"
    result["active_tasks"] = active
    result["failed_tasks"] = failed
    units = [unit["unit_id"] for unit in result["units"]]
    timelines: list[dict[str, Any]] = []
    service = FinalCutService(projects)
    timeline_service = EditTimelineService(projects)
    for document in documents:
        clip_ids = [clip.unit_id for clip in document.latest.content.clips]
        matches = bool(units) and list(dict.fromkeys(clip_ids)) == units
        status = await service.status(project_name, document.id, narration=WITH_NARRATION, subtitles=BURNED_SUBTITLES)
        readout = await timeline_service.read(project_name, document.id, revision=document.latest.number)
        final = status.model_dump(mode="json")
        final["current"] = status.status is ArtifactStatus.CURRENT
        final["download_url"] = (
            f"/api/v1/files/{quote(project_name, safe='')}/{quote(status.artifact_path)}?v={status.version}"
            if status.version
            else None
        )
        timelines.append(
            {
                "timeline_id": document.id,
                "name": document.name,
                "episode_id": document.episode,
                "revision": document.latest.number,
                "unit_ids": clip_ids,
                "units_match": matches,
                "final_cut": final,
                "readout": readout.model_dump(mode="json"),
            }
        )
    result["timelines"] = timelines
    complete_timeline_ids = {timeline["timeline_id"] for timeline in timelines if timeline["final_cut"]["current"]}
    result["failed_tasks"] = [
        task
        for task in failed
        if not (
            task["task_type"] == "render_final_cut"
            and any(str(task.get("resource_id", "")).startswith(timeline + ".") for timeline in complete_timeline_ids)
        )
    ]
    result["goal_complete_technical"] = episode_final_cut_complete(episode_id, result["units"], timelines)
    return result
