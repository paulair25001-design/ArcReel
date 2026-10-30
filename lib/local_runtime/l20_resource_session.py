"""Release/restore the inference host through its native model router.

ArcReel owns one media resource session; its normal task queue, scripts and
artifact versions own all production state. No external workflow is loaded.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import nullcontext

import httpx
import portalocker

from lib.infra.async_thread import run_sync_transaction
from lib.infra.json_io import atomic_write_json
from lib.local_runtime.l20_direct_config import L20DirectConfig, get_l20_direct_config
from lib.local_runtime.l20_resource_state import resource_handoff_complete
from lib.local_runtime.l20_transport import L20SSHTransport


class L20ResourceSession:
    def __init__(self, config: L20DirectConfig, *, transport=None, sleep=time.sleep, clock=time.monotonic):
        self.config = config
        self._transport = transport
        self._sleep, self._clock = sleep, clock

    @property
    def transport(self):
        if self._transport is None:
            self._transport = L20SSHTransport(self.config)
        return self._transport

    @property
    def path(self):
        return self.config.state_dir / "resource-session.json"

    def _lock(self):
        self.config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (self.config.state_dir, *self.config.state_dir.parents)):
            raise ValueError("L20 session directory cannot be redirected")
        return portalocker.Lock(str(self.config.state_dir / "resource.lock"), timeout=600)

    def _save(self, record):
        atomic_write_json(self.path, record)
        history = self.config.state_dir / "sessions"
        history.mkdir(mode=0o700, exist_ok=True)
        atomic_write_json(history / (record["session"] + ".json"), record)

    def _read(self):
        if self.path.is_symlink():
            raise ValueError("L20 session receipt cannot be redirected")
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def _snapshot(self):
        return self.transport.rpc("snapshot")

    def _wait_models(self, expected, timeout=300, *, allow_extra=False):
        deadline = self._clock() + timeout
        while self._clock() < deadline:
            value = self._snapshot()
            models = value["models"]
            selected = [x for x in models if x["id"] in expected]
            if (
                {x["id"] for x in selected} == set(expected)
                and all(x["state"] == "loaded" for x in selected)
                and (allow_extra or len(selected) == len(models))
            ):
                return value
            self._sleep(2)
        raise RuntimeError("L20 model state has not settled; restoration remains required")

    def _http(self):
        return httpx.Client(base_url=self.config.comfy_url, timeout=10, trust_env=False)

    def _ready(self, record):
        process = self.transport.rpc("comfy_identity", {"root": record["remote_root"]})
        if not process["listening"]:
            raise httpx.ConnectError("Owned ComfyUI is still starting")
        with self._http() as http:
            response = http.get("/system_stats")
            response.raise_for_status()
            return response.json()

    def begin(self, project: str, *, fresh: bool = False):
        if not project:
            raise ValueError("ArcReel project ownership is required")
        with self._lock():
            record = self._read()
            if record and not resource_handoff_complete(record):
                if record["project"] != project:
                    raise RuntimeError("Another ArcReel project owns the L20 resource session")
                if fresh:
                    self._stop_engine(record)
                    record.update(
                        state="closed",
                        resource_released=True,
                        original_service_restored=False,
                        restore_original_models=False,
                        close_reason="new_media_round",
                    )
                    self._save(record)
                elif record["state"] != "active":
                    raise RuntimeError("The previous resource transition needs finalization before new media")
                else:
                    # An accepted job resumes its exact owned engine and handle;
                    # changes to the separate llama router do not invalidate it.
                    self._ready(record)
                    return record
            self.transport.connect()
            original = self.transport.rpc("snapshot")
            if self.config.standby_model and self.config.standby_model not in original["catalog"]:
                raise RuntimeError("Configured L20 standby model is not available on the current router")
            self.transport.rpc("slots", {"models": original["models"]})
            self._sleep(2)
            current = self.transport.rpc("snapshot")
            if current["router"] != original["router"] or current["models"] != original["models"]:
                raise RuntimeError("Original L20 model identity/state changed during admission")
            self.transport.rpc("slots", {"models": original["models"]})
            session = "session-" + uuid.uuid4().hex
            record = {
                "schema": "arcreel-native-l20-resources/v1",
                "session": session,
                "project": project,
                "state": "releasing",
                "restore_original_models": self.config.restore_original_models,
                "standby_model": self.config.standby_model,
                "original": original,
                "remote_root": "/nvme1/.media-workspace/work/arcreel/" + session,
                "unload_requests": [],
                "load_requests": [],
            }
            # This durable original set is written before the first unload.
            self._save(record)
            try:
                for model in original["models"]:
                    record["unload_requests"].append(model["id"])
                    self._save(record)
                    self.transport.rpc(
                        "unload", {"model": model["id"], "router": self._snapshot()["router"]}, timeout=270
                    )
                record["released"] = self._wait_models([], timeout=120)
                record["startup_requested"] = True
                self._save(record)
                record["comfy"] = self.transport.rpc(
                    "start_comfy",
                    {
                        "root": record["remote_root"],
                        "python": os.environ.get(
                            "ARCREEL_L20_COMFY_PYTHON", "/nvme1/.media-workspace/envs/h3-vsa-cu128-20261015/bin/python"
                        ),
                        "engine": os.environ.get(
                            "ARCREEL_L20_COMFY_ENGINE",
                            "/nvme1/.media-workspace/apps/ComfyUI-v0.40.0-fasth3-vsa-20261015",
                        ),
                    },
                )
                self._save(record)
                deadline = self._clock() + 180
                while True:
                    try:
                        record["engine_stats"] = self._ready(record)
                        break
                    except (httpx.HTTPError, ValueError):
                        if self._clock() >= deadline:
                            record["startup_diagnostic"] = self.transport.rpc("log", {"root": record["remote_root"]})
                            self._save(record)
                            raise RuntimeError("The ArcReel inference engine did not become ready") from None
                        self._sleep(2)
                record["state"] = "active"
                self._save(record)
                return record
            except BaseException:
                # begin() has not submitted any ComfyUI prompt. Roll back only
                # the owned resource transition, preserving its exact receipt.
                try:
                    self._restore(record, before_submission=True)
                except Exception as error:
                    record.update(
                        state="restoration_required" if self.config.restore_original_models else "closure_required",
                        restoration_error=type(error).__name__,
                    )
                    self._save(record)
                raise

    def _stop_engine(self, record, *, before_submission=False):
        if (record.get("startup_requested") or record.get("comfy")) and not record.get("engine_stop", {}).get(
            "stopped"
        ):
            if not before_submission and record.get("comfy"):
                with self._http() as http:
                    response = http.get("/queue")
                    response.raise_for_status()
                    queue = response.json()
                    if queue.get("queue_running") != [] or queue.get("queue_pending") != []:
                        raise RuntimeError("ArcReel ComfyUI still has media work; do not restore models yet")
                    response = http.post("/free", json={"unload_models": True, "free_memory": True})
                    response.raise_for_status()
            record["engine_stop"] = self.transport.rpc("stop_comfy", {"root": record["remote_root"]})
            self._save(record)

    def _restore(self, record, *, before_submission=False):
        record["restore_original_models"] = self.config.restore_original_models
        record["state"] = "restoring" if self.config.restore_original_models else "closing"
        self._save(record)
        self._stop_engine(record, before_submission=before_submission)
        if not self.config.restore_original_models:
            record.update(state="closed", original_service_restored=False, resource_released=True)
            self._save(record)
            self.transport.close()
            return record
        current = self._snapshot()
        available = set(current["catalog"])
        original_ids = [x["id"] for x in record["original"]["models"]]
        record["retired_model_ids"] = [name for name in original_ids if name not in available]
        targets = list(dict.fromkeys(name for name in original_ids if name in available))
        if self.config.standby_model:
            if self.config.standby_model not in available:
                raise RuntimeError("Configured L20 standby model is not available on the current router")
            targets = [name for name in targets if name != self.config.standby_model] + [self.config.standby_model]
        record["restoration_targets"] = targets
        self._save(record)
        for name in targets:
            current = self._snapshot()
            model = next((item for item in current["models"] if item["id"] == name), None)
            if model is None:
                # A lost load response is reconciled from router state on the
                # next finish call, without immediately repeating the POST.
                record["load_requests"].append(name)
                self._save(record)
                self.transport.rpc("load", {"model": name, "router": current["router"]}, timeout=270)
            if model is None or model["state"] != "loaded":
                # A successful load POST acknowledges startup, not residency.
                # Settle each model before the standby model can occupy VRAM.
                self._wait_models([name], allow_extra=True)
        record["restored"] = self._wait_models(targets, allow_extra=True)
        if record["restored"]["health"].get("status") != "ok":
            raise RuntimeError("Original L20 router health is unconfirmed")
        record["restored_ready"] = self.transport.rpc(
            "slots", {"models": [x for x in record["restored"]["models"] if x["id"] in targets], "require_idle": False}
        )
        record.update(state="restored", original_service_restored=True)
        self._save(record)
        self.transport.close()
        return record

    def finish(self, project: str | None = None, *, gpu_lock_held: bool = False):
        # Same order as generation: gpu lock then resource transition lock.
        self.config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        gpu_lock = (
            nullcontext() if gpu_lock_held else portalocker.Lock(str(self.config.state_dir / "gpu.lock"), timeout=600)
        )
        with gpu_lock, self._lock():
            record = self._read()
            if not record:
                if self.config.restore_original_models:
                    return {"state": "restored", "original_service_restored": True, "active": False}
                return {
                    "state": "closed",
                    "restore_original_models": False,
                    "original_service_restored": False,
                    "resource_released": True,
                    "active": False,
                }
            if resource_handoff_complete(record) and not (
                self.config.restore_original_models and self.config.standby_model
            ):
                return {**record, "active": False}
            if project is not None and record["project"] != project:
                raise RuntimeError("Only the owning ArcReel project can finish this resource session")
            return self._restore(record)


async def finalize_l20_runtime(*, project: str | None = None):
    return await run_sync_transaction(L20ResourceSession(get_l20_direct_config()).finish, project)
