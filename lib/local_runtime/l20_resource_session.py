"""Release/restore the inference host through its native model router.

ArcReel owns one media resource session; its normal task queue, scripts and
artifact versions own all production state. No external workflow is loaded.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx
import portalocker

from lib.infra.async_thread import run_sync_transaction
from lib.infra.json_io import atomic_write_json
from lib.local_runtime.l20_direct_config import L20DirectConfig, get_l20_direct_config
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

    def _snapshot(self, record):
        value = self.transport.rpc("snapshot")
        if value["router"] != record["original"]["router"]:
            raise RuntimeError("L20 original router changed; retain the restoration receipt")
        return value

    def _wait_models(self, record, expected, timeout=300):
        deadline = self._clock() + timeout
        while self._clock() < deadline:
            value = self._snapshot(record)
            models = value["models"]
            if {x["id"] for x in models} == set(expected) and all(x["state"] == "loaded" for x in models):
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

    def begin(self, project: str):
        if not project:
            raise ValueError("ArcReel project ownership is required")
        with self._lock():
            record = self._read()
            if record and record["state"] != "restored":
                if record["project"] != project:
                    raise RuntimeError("Another ArcReel project owns the L20 resource session")
                if record["state"] != "active":
                    raise RuntimeError("The previous resource transition needs restoration before new media")
                if self._snapshot(record)["models"]:
                    raise RuntimeError("Original models were reloaded during media production; pause for inspection")
                self._ready(record)
                return record
            self.transport.connect()
            original = self.transport.rpc("snapshot")
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
                    self.transport.rpc("unload", {"model": model["id"], "router": original["router"]}, timeout=270)
                record["released"] = self._wait_models(record, [], timeout=120)
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
                    record.update(state="restoration_required", restoration_error=type(error).__name__)
                    self._save(record)
                raise

    def _restore(self, record, *, before_submission=False):
        record["state"] = "restoring"
        self._save(record)
        if record.get("startup_requested") or record.get("comfy"):
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
        current = self._snapshot(record)
        original_ids = {x["id"] for x in record["original"]["models"]}
        if {x["id"] for x in current["models"]} - original_ids:
            raise RuntimeError("Unexpected model loaded; retain the original restoration receipt")
        loaded_ids = {x["id"] for x in current["models"]}
        for model in record["original"]["models"]:
            if model["id"] in loaded_ids:
                continue
            # A lost load response is reconciled from router state on the next
            # finish call. It is never followed immediately by a duplicate POST.
            record["load_requests"].append(model["id"])
            self._save(record)
            self.transport.rpc("load", {"model": model["id"], "router": record["original"]["router"]}, timeout=270)
        record["restored"] = self._wait_models(record, original_ids)
        if record["restored"]["health"].get("status") != "ok":
            raise RuntimeError("Original L20 router health is unconfirmed")
        record["restored_idle_1"] = self.transport.rpc("slots", {"models": record["restored"]["models"]})
        self._sleep(2)
        fresh = self._wait_models(record, original_ids, timeout=30)
        record["restored_idle_2"] = self.transport.rpc("slots", {"models": fresh["models"]})
        record.update(state="restored", original_service_restored=True)
        self._save(record)
        self.transport.close()
        return record

    def finish(self, project: str | None = None):
        # Same order as generation: gpu lock then resource transition lock.
        self.config.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with portalocker.Lock(str(self.config.state_dir / "gpu.lock"), timeout=600), self._lock():
            record = self._read()
            if not record or record["state"] == "restored":
                return {"state": "restored", "original_service_restored": True, "active": False}
            if project is not None and record["project"] != project:
                raise RuntimeError("Only the owning ArcReel project can finish this resource session")
            return self._restore(record)


async def finalize_l20_runtime(*, project: str | None = None):
    return await run_sync_transaction(L20ResourceSession(get_l20_direct_config()).finish, project)
