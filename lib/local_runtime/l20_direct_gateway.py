"""ArcReel task → original ComfyUI HTTP protocol, with durable job identity.

No factory, external catalog, skill, packet or alternate production ledger is
loaded. A small provider receipt retains the actual submitted HTTP graph and
prompt ID; ArcReel's task queue and artifact versions remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import httpx
import portalocker
from PIL import Image, ImageOps

from lib.backends.http_status_errors import ArtifactDownloadError
from lib.infra.json_io import atomic_write_bytes, atomic_write_json
from lib.infra.logging_utils import redact_diagnostic_text
from lib.local_runtime.l20_capabilities import IMAGE_REFERENCE_LIMIT
from lib.local_runtime.l20_direct_config import L20DirectConfig
from lib.local_runtime.l20_direct_graphs import (
    build_image_graph,
    build_video_graph,
    normalized_workflow_sha256,
    workflow_sha256,
)

logger = logging.getLogger(__name__)


class L20DirectError(RuntimeError):
    """The exact ArcReel job requires inspection; never silently retry a POST."""


class ResourceSession(Protocol):
    def begin(self, project: str, *, fresh: bool = False) -> Any: ...

    def finish(self, project: str, *, gpu_lock_held: bool = False) -> Any: ...


@dataclass(frozen=True)
class DirectArtifact:
    path: Path
    sha256: str
    record: Mapping[str, Any]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise L20DirectError("The ArcReel L20 provider directory must not be redirected")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


class L20DirectGateway:
    def __init__(
        self,
        config: L20DirectConfig,
        resource_session: ResourceSession,
        *,
        client_factory: Callable[..., Any] = httpx.Client,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config, self.resource_session = config, resource_session
        self._client_factory, self._clock, self._sleep = client_factory, clock, sleep

    def _path(self, handle: str) -> Path:
        if not re.fullmatch(r"arcreel-l20-[a-f0-9]{64}", handle):
            raise L20DirectError("This is not an independent ArcReel L20 job identity")
        return _directory(self.config.state_dir / "jobs" / handle) / "job.json"

    def _read(self, handle: str) -> dict[str, Any]:
        path = self._path(handle)
        if path.is_symlink() or not path.is_file():
            raise L20DirectError("The exact ArcReel provider receipt is missing")
        return json.loads(path.read_bytes())

    def _save(self, record: Mapping[str, Any]) -> None:
        atomic_write_json(self._path(record["handle"]), dict(record))

    def intent(self, handle: str) -> Mapping[str, Any]:
        return self._read(handle)["intent"]

    def prepare(self, intent: Mapping[str, Any], *, task_identity: str) -> str:
        """Freeze local inputs before the backend publishes this resumable ID."""
        handle = "arcreel-l20-" + _sha(_canonical([task_identity, intent]))
        root = _directory(self.config.state_dir)
        with portalocker.Lock(str(root / "prepare.lock"), timeout=self.config.timeout_seconds):
            path = self._path(handle)
            if path.exists():
                if self._read(handle)["intent"] != intent:
                    raise L20DirectError("The provider handle belongs to different inputs")
                return handle
            width, height = intent["canvas"]
            multiple = 16 if intent["media_type"] == "video" else 32
            if any(type(x) is not int or x < 64 or x > 4096 or x % multiple for x in (width, height)):
                raise L20DirectError(f"L20 requires an exact {multiple}-pixel aligned canvas")
            prompt = intent["prompt"]
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768 or not intent.get("project"):
                raise L20DirectError("The project and non-empty prompt must be explicit")
            if intent["media_type"] not in {"image", "video"}:
                raise L20DirectError("Unsupported L20 media kind")
            if type(intent["seed"]) is not int or not 0 <= intent["seed"] < 2**64:
                raise L20DirectError("The L20 seed must be an unsigned 64-bit integer")
            frozen = []
            references = intent.get("references", [])
            if (intent["media_type"] == "image" and len(references) > IMAGE_REFERENCE_LIMIT) or (
                intent["media_type"] == "video" and len(references) != 2
            ):
                raise L20DirectError("The graph cannot represent this reference image count")
            for index, reference in enumerate(references):
                source = Path(reference["path"])
                if source.is_symlink() or _sha(source.read_bytes()) != reference["sha256"]:
                    raise L20DirectError("An input image changed before its snapshot")
                target = path.parent / f"reference-{index}.png"
                with Image.open(source) as image:
                    image.load()
                    if intent["media_type"] == "video":
                        if image.format != "PNG" or not all(64 <= value <= 4096 for value in image.size):
                            raise L20DirectError(
                                "Video first and target-tail frames must be PNGs within 64..4096 pixels"
                            )
                        atomic_write_bytes(target, source.read_bytes())
                    else:
                        ImageOps.pad(image.convert("RGB"), (width, height), color="black").save(target, format="PNG")
                checksum = _sha(target.read_bytes())
                if _sha(source.read_bytes()) != reference["sha256"] or (
                    intent["media_type"] == "video" and checksum != reference["sha256"]
                ):
                    raise L20DirectError("An input image changed during its snapshot")
                frozen.append({"path": str(target), "sha256": checksum, "source_sha256": reference["sha256"]})
            self._save(
                {
                    "handle": handle,
                    "intent": dict(intent),
                    "references": frozen,
                    "state": "prepared",
                    "comfy_url": self.config.comfy_url,
                }
            )
        return handle

    @staticmethod
    def _json(http: Any, method: str, route: str, **kwargs: Any) -> Any:
        response = http.request(method, route, **kwargs)
        response.raise_for_status()
        return response.json()

    def _upload(self, http: Any, record: dict[str, Any]) -> list[str]:
        names = []
        for index, reference in enumerate(record["references"]):
            raw = Path(reference["path"]).read_bytes()
            if _sha(raw) != reference["sha256"]:
                raise L20DirectError("The frozen input snapshot changed")
            value = self._json(
                http,
                "POST",
                "/upload/image",
                files={"image": (record["handle"] + f"-{index}.png", raw, "image/png")},
                data={"type": "input", "subfolder": "arcreel", "overwrite": "false"},
            )
            name, folder = str(value["name"]), str(value.get("subfolder", ""))
            if (
                not name
                or PurePosixPath(name).name != name
                or PurePosixPath(folder).is_absolute()
                or any(x == ".." for x in PurePosixPath(folder).parts)
            ):
                raise L20DirectError("ComfyUI returned an unsafe input reference")
            # Check the real server copy before binding its response name.
            uploaded = http.get("/view", params={"filename": name, "subfolder": folder, "type": "input"})
            uploaded.raise_for_status()
            if _sha(uploaded.content) != reference["sha256"]:
                raise L20DirectError("ComfyUI's uploaded image bytes differ")
            names.append(folder + "/" + name if folder else name)
        return names

    @staticmethod
    def _validate_nodes(http: Any, workflow: dict[str, Any]) -> None:
        info = L20DirectGateway._json(http, "GET", "/object_info")
        for node in workflow.values():
            kind = node["class_type"]
            if (
                kind not in info
                or kind.startswith("FastH3Probe")
                or kind in {"FastH3ResidentHold", "FastH3AdmissionGuard"}
            ):
                raise L20DirectError(f"The independent ComfyUI service does not expose the inference node {kind}")

    def _reconcile(self, http: Any, record: dict[str, Any]) -> str:
        """Resolve a lost POST response by exact identity, never by resubmitting."""
        if workflow_sha256(record["workflow"]) != record["workflow_sha256"]:
            raise L20DirectError("The frozen submission graph no longer matches its original hash")
        queue = self._json(http, "GET", "/queue")
        history = self._json(http, "GET", "/history", params={"max_items": 1000})
        candidates = list(queue.get("queue_running", [])) + list(queue.get("queue_pending", []))
        candidates += [entry.get("prompt", []) for entry in history.values()]
        found = set()
        for entry in candidates:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            extra = entry[3]
            marker = extra.get("arcreel", {}) if isinstance(extra, dict) else {}
            if (
                marker.get("handle") == record["handle"]
                and marker.get("workflow_sha256") == record["workflow_sha256"]
                and normalized_workflow_sha256(entry[2]) == normalized_workflow_sha256(record["workflow"])
            ):
                found.add(entry[1])
        if len(found) != 1:
            record.update(state="attention", reconciliation_matches=len(found))
            self._save(record)
            raise L20DirectError("The uncertain ComfyUI submission has no single exact match; do not submit again")
        return found.pop()

    def run(self, handle: str) -> DirectArtifact:
        """Serialize image/video GPU use and finish only the already-bound job."""
        root = _directory(self.config.state_dir)
        with portalocker.Lock(str(root / "gpu.lock"), timeout=self.config.timeout_seconds):
            record = self._read(handle)
            if "workflow" in record and workflow_sha256(record["workflow"]) != record["workflow_sha256"]:
                raise L20DirectError("The frozen submission graph no longer matches its original hash")
            if record["state"] == "completed":
                return self._artifact(record)
            if record["comfy_url"] != self.config.comfy_url:
                raise L20DirectError("Resume must use the original ComfyUI endpoint")
            self.resource_session.begin(record["intent"]["project"], fresh=record["state"] == "prepared")
            try:
                with self._client_factory(
                    base_url=record["comfy_url"], timeout=60, follow_redirects=False, trust_env=False
                ) as http:
                    if record["state"] == "prepared":
                        queue = self._json(http, "GET", "/queue")
                        if (
                            not isinstance(queue.get("queue_running"), list)
                            or not isinstance(queue.get("queue_pending"), list)
                            or queue["queue_running"]
                            or queue["queue_pending"]
                        ):
                            raise L20DirectError("The ComfyUI queue is not empty; retain this prepared ArcReel task")
                        names = self._upload(http, record)
                        intent = record["intent"]
                        prompt, width, height, seed = (
                            intent["prompt"],
                            intent["canvas"][0],
                            intent["canvas"][1],
                            intent["seed"],
                        )
                        prefix = "arcreel/" + handle
                        compiled = (
                            build_video_graph(prompt, width, height, seed, names, intent["requested_frames"], prefix)
                            if intent["media_type"] == "video"
                            else build_image_graph(prompt, width, height, seed, names, prefix)
                        )
                        self._validate_nodes(http, compiled.workflow)
                        record.update(
                            workflow=compiled.workflow,
                            workflow_sha256=compiled.sha256,
                            output_node=compiled.output_node,
                            native_frames=compiled.native_frames,
                            state="submission_started",
                        )
                        self._save(record)
                        # Exactly one POST. Persist intent first: timeout or process
                        # death is reconciled through queue/history, never replayed.
                        value = self._json(
                            http,
                            "POST",
                            "/prompt",
                            json={
                                "prompt": compiled.workflow,
                                "client_id": handle,
                                "extra_data": {"arcreel": {"handle": handle, "workflow_sha256": compiled.sha256}},
                            },
                        )
                        if (
                            value.get("node_errors")
                            or not isinstance(value.get("prompt_id"), str)
                            or not value["prompt_id"]
                        ):
                            record.update(state="attention", submission_response=value)
                            self._save(record)
                            raise L20DirectError("ComfyUI did not accept the exact graph")
                        record.update(prompt_id=value["prompt_id"], state="submitted")
                        self._save(record)
                    elif record["state"] in {"submission_started", "attention"}:
                        record.update(prompt_id=self._reconcile(http, record), state="submitted")
                        self._save(record)
                    return self._wait(http, record)
            finally:
                history = record.get("history", {})
                if (
                    record["state"] in {"prepared", "completed"}
                    or history.get("status", {}).get("status_str") == "error"
                ):
                    self._finish_round(record)

    def _finish_round(self, record: dict[str, Any]) -> None:
        """Finish a settled media job inside its already-held GPU transaction.

        Uncertain submissions and downloads keep the exact owned engine alive.
        A cleanup failure retains the valid generated artifact and the separate
        resource recovery responsibility; it cannot turn that image into a retry.
        """
        try:
            result = self.resource_session.finish(record["intent"]["project"], gpu_lock_held=True)
            record["resource_finalization"] = {
                key: result.get(key)
                for key in ("state", "original_service_restored", "restore_original_models", "restoration_targets")
            }
        except Exception as error:
            detail = redact_diagnostic_text(error)
            record["resource_finalization"] = {"state": "pending", "error": detail}
            logger.warning("L20 media finished; resource finalization remains pending: %s", detail)
        self._save(record)

    def _wait(self, http: Any, record: dict[str, Any]) -> DirectArtifact:
        deadline = self._clock() + self.config.timeout_seconds
        while self._clock() < deadline:
            body = self._json(http, "GET", "/history/" + record["prompt_id"])
            entry = body.get(record["prompt_id"])
            if entry is not None:
                if entry.get("status", {}).get("status_str") == "error":
                    record.update(state="attention", history=entry)
                    self._save(record)
                    raise L20DirectError("The exact ComfyUI inference failed; its response is retained")
                if entry.get("status", {}).get("completed") is True:
                    prompt = entry.get("prompt", [])
                    if workflow_sha256(record["workflow"]) != record["workflow_sha256"]:
                        raise L20DirectError("The frozen submission graph no longer matches its original hash")
                    if len(prompt) < 3 or not isinstance(prompt[2], dict):
                        raise L20DirectError("Completed ComfyUI history has no actual graph")
                    actual_raw_hash = workflow_sha256(prompt[2])
                    expected_normalized = normalized_workflow_sha256(record["workflow"])
                    actual_normalized = normalized_workflow_sha256(prompt[2])
                    record["history_graph_comparison"] = {
                        "policy": "json-integral-floats-v1",
                        "frozen_raw_sha256": record["workflow_sha256"],
                        "actual_raw_sha256": actual_raw_hash,
                        "frozen_normalized_sha256": expected_normalized,
                        "actual_normalized_sha256": actual_normalized,
                        "equivalent": expected_normalized == actual_normalized,
                    }
                    self._save(record)
                    if expected_normalized != actual_normalized:
                        raise L20DirectError("Completed ComfyUI history differs from the frozen graph")
                    outputs = entry.get("outputs", {}).get(record["output_node"], {})
                    extension = ".mp4" if record["intent"]["media_type"] == "video" else ".png"
                    artifacts = [
                        x
                        for name in ("images", "videos", "gifs")
                        for x in outputs.get(name, [])
                        if str(x.get("filename", "")).lower().endswith(extension)
                    ]
                    if len(artifacts) != 1:
                        raise L20DirectError("ComfyUI must return one exact output artifact")
                    artifact = artifacts[0]
                    if (
                        artifact.get("type") != "output"
                        or PurePosixPath(artifact["filename"]).name != artifact["filename"]
                        or PurePosixPath(artifact.get("subfolder", "")).is_absolute()
                        or any(part == ".." for part in PurePosixPath(artifact.get("subfolder", "")).parts)
                    ):
                        raise L20DirectError("ComfyUI output escaped its declared artifact reference")
                    path = self._path(record["handle"]).with_name("native" + extension)
                    temporary = path.with_suffix(path.suffix + ".part")
                    count = 0
                    try:
                        with http.stream(
                            "GET",
                            "/view",
                            params={key: artifact[key] for key in ("filename", "subfolder", "type") if key in artifact},
                        ) as response:
                            response.raise_for_status()
                            with temporary.open("wb") as stream:
                                for chunk in response.iter_bytes():
                                    count += len(chunk)
                                    if count > 2 * 1024**3:
                                        raise L20DirectError("ComfyUI artifact exceeds the transfer limit")
                                    stream.write(chunk)
                        if count == 0:
                            raise L20DirectError("ComfyUI returned an empty artifact")
                        os.replace(temporary, path)
                    except (httpx.HTTPError, OSError) as error:
                        # History and graph identity are already verified. This
                        # is a retrievable provider success, not new inference.
                        raise ArtifactDownloadError(detail=str(error)) from error
                    finally:
                        temporary.unlink(missing_ok=True)  # hard-delete: temp -- incomplete provider download
                    record.update(state="completed", output=str(path), output_sha256=_file_sha(path), history=entry)
                    self._save(record)
                    try:
                        queue = self._json(http, "GET", "/queue")
                        if queue.get("queue_running") == [] and queue.get("queue_pending") == []:
                            self._json(http, "POST", "/free", json={"unload_models": True, "free_memory": True})
                        else:
                            record["cleanup_observation"] = "queue_not_empty_after_completed_output"
                    except (httpx.HTTPError, ValueError) as error:
                        record["cleanup_observation"] = type(error).__name__
                    self._save(record)
                    return self._artifact(record)
            self._sleep(self.config.poll_interval)
        raise L20DirectError("ComfyUI polling timed out; resume this exact handle without resubmitting")

    @staticmethod
    def _artifact(record: dict[str, Any]) -> DirectArtifact:
        path = Path(record["output"])
        if path.is_symlink() or not path.is_file() or _file_sha(path) != record["output_sha256"]:
            raise L20DirectError("The retained provider artifact changed")
        return DirectArtifact(path, record["output_sha256"], record)

    def finalize(self, *, project: str) -> Any:
        # The resource finalizer takes gpu.lock then resource.lock itself. Do
        # not recursively lock gpu.lock from this convenience entry point.
        return self.resource_session.finish(project)
