from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import respx

from lib.local_runtime.l20_direct_config import L20DirectConfig
from lib.local_runtime.l20_resource_session import L20ResourceSession


class RouterTransport:
    """The project's SSH/RPC boundary, modeling router and owned engine state."""

    def __init__(self, config):
        self.config = config
        self.router = {"pid": 100, "birth": "1000", "uid": 1000, "argv_sha256": "a" * 64, "boot_id": "fixture"}
        self.models = {
            "alpha": {"id": "alpha", "state": "loaded", "port": 7001},
            "beta": {"id": "beta", "state": "loaded", "port": 7002},
        }
        self.calls = []
        self.owned_engine = False
        self.unload_failure = None
        self.lose_start_response = False
        self.busy_slots = False

    def connect(self):
        self.calls.append(("connect", {}))

    def close(self):
        self.calls.append(("close", {}))

    def rpc(self, operation, payload=None, **kwargs):
        payload = payload or {}
        self.calls.append((operation, copy.deepcopy(payload)))
        if operation == "snapshot":
            return {
                "router": copy.deepcopy(self.router),
                "models": copy.deepcopy(list(self.models.values())),
                "health": {"status": "ok"},
                "gpu_mib": "45000, 1000",
            }
        if operation == "slots":
            if self.busy_slots:
                raise RuntimeError("Original model has an active or unknown slot")
            assert all(item["state"] == "loaded" for item in payload["models"])
            return {"idle": True}
        if operation == "unload":
            # The actual loaded set and restore responsibility must exist on
            # disk before any resource-changing request reaches this boundary.
            record = json.loads((self.config.state_dir / "resource-session.json").read_bytes())
            assert {item["id"] for item in record["original"]["models"]} == {"alpha", "beta"}
            assert payload["model"] in record["unload_requests"]
            if payload["model"] == self.unload_failure:
                raise RuntimeError("unload failed")
            self.models.pop(payload["model"])
            return {}
        if operation == "load":
            assert payload["model"] not in self.models
            name = payload["model"]
            self.models[name] = {"id": name, "state": "loaded", "port": 7010 + len(self.models)}
            return {}
        if operation == "start_comfy":
            assert self.models == {}
            self.owned_engine = True
            if self.lose_start_response:
                raise TimeoutError("owned engine started but RPC response was lost")
            return {"root": payload["root"], "process": {"pid": 200, "birth": "2000"}}
        if operation == "comfy_identity":
            return {"listening": self.owned_engine, "process": {"pid": 200, "birth": "2000"}}
        if operation == "stop_comfy":
            self.owned_engine = False
            return {"stopped": True}
        raise AssertionError(f"Unknown resource operation: {operation}")


@pytest.fixture
def resources(tmp_path):
    config = L20DirectConfig("http://127.0.0.1:18191", "http://127.0.0.1:18080", tmp_path / "runtime", 0.001, 10)
    transport = RouterTransport(config)
    session = L20ResourceSession(config, transport=transport, sleep=lambda seconds: None)
    with respx.mock(assert_all_called=False) as mock:
        mock.get(config.comfy_url + "/system_stats").respond(200, json={"system": {"comfyui_version": "test"}})
        mock.get(config.comfy_url + "/queue").respond(200, json={"queue_running": [], "queue_pending": []})
        mock.post(config.comfy_url + "/free").respond(200, json={})
        yield session, transport, mock


def record(session):
    return json.loads(session.path.read_bytes())


def test_partial_unload_failure_restores_only_the_missing_original_models(resources):
    session, transport, _ = resources
    transport.unload_failure = "beta"
    with pytest.raises(RuntimeError, match="unload failed"):
        session.begin("demo")
    assert set(transport.models) == {"alpha", "beta"}
    assert record(session)["state"] == "restored"
    assert [payload["model"] for operation, payload in transport.calls if operation == "load"] == ["alpha"]
    assert not any(operation == "start_comfy" for operation, _ in transport.calls)


def test_foreign_project_cannot_take_or_finish_the_owned_session(resources):
    session, transport, _ = resources
    session.begin("original")
    before = copy.deepcopy(transport.calls)
    with pytest.raises(RuntimeError, match="Another ArcReel project"):
        session.begin("foreign")
    with pytest.raises(RuntimeError, match="owning ArcReel project"):
        session.finish("foreign")
    assert transport.calls == before
    assert record(session)["project"] == "original"


def test_router_identity_change_blocks_restore_loads_and_retains_the_capture(resources):
    session, transport, _ = resources
    session.begin("demo")
    original = copy.deepcopy(record(session)["original"])
    transport.router["birth"] = "different-router"
    with pytest.raises(RuntimeError, match="original router changed"):
        session.finish("demo")
    assert not any(operation == "load" for operation, _ in transport.calls)
    assert record(session)["original"] == original
    assert record(session)["state"] != "restored"


def test_lost_start_response_stops_only_the_known_owned_engine_before_restoring(resources):
    session, transport, _ = resources
    transport.lose_start_response = True
    with pytest.raises(TimeoutError, match="RPC response was lost"):
        session.begin("demo")
    assert transport.owned_engine is False
    assert set(transport.models) == {"alpha", "beta"}
    assert record(session)["state"] == "restored"
    stopped = [payload for operation, payload in transport.calls if operation == "stop_comfy"]
    assert stopped == [{"root": record(session)["remote_root"]}]


@pytest.mark.parametrize("restore", [True, False])
def test_busy_comfy_queue_refuses_finish_without_stop_or_original_model_load(resources, restore):
    session, transport, mock = resources
    session.config = replace(session.config, restore_original_models=restore)
    session.begin("demo")
    before = len(transport.calls)
    mock.get(session.config.comfy_url + "/queue").respond(
        200, json={"queue_running": [[1, "accepted-job"]], "queue_pending": []}
    )
    with pytest.raises(RuntimeError, match="still has media work"):
        session.finish("demo")
    after = [operation for operation, _ in transport.calls[before:]]
    assert "stop_comfy" not in after
    assert "load" not in after
    assert transport.owned_engine is True
    assert record(session)["state"] != "restored"


def test_normal_finish_restores_original_names_without_restarting_the_router(resources):
    session, transport, _ = resources
    original_router = copy.deepcopy(transport.router)
    session.begin("demo")
    finished = session.finish("demo")
    assert finished["state"] == "restored"
    assert finished["original_service_restored"] is True
    assert transport.router == original_router
    assert set(transport.models) == {"alpha", "beta"}
    assert transport.owned_engine is False
    assert [payload["model"] for operation, payload in transport.calls if operation == "load"] == ["alpha", "beta"]


def test_retry_finish_skips_an_already_stopped_engine_and_rechecks_original_models(resources):
    session, transport, mock = resources
    session.begin("demo")
    transport.busy_slots = True
    with pytest.raises(RuntimeError, match="active or unknown slot"):
        session.finish("demo")
    assert record(session)["engine_stop"]["stopped"] is True
    assert record(session)["state"] == "restoring"
    mock.get(session.config.comfy_url + "/queue").respond(503)
    transport.busy_slots = False

    finished = session.finish("demo")

    assert finished["state"] == "restored"
    assert finished["original_service_restored"] is True
    assert len([operation for operation, _ in transport.calls if operation == "stop_comfy"]) == 1


def test_disabled_restore_closes_only_the_owned_engine_and_allows_the_next_native_session(resources):
    session, transport, _ = resources
    session.config = replace(session.config, restore_original_models=False)
    first = session.begin("demo")
    assert transport.models == {}

    finished = session.finish("demo")

    assert finished["state"] == "closed"
    assert finished["original_service_restored"] is False
    assert finished["resource_released"] is True
    assert finished["restore_original_models"] is False
    assert transport.owned_engine is False
    assert transport.models == {}
    assert not any(operation == "load" for operation, _ in transport.calls)
    session.finish("demo")
    assert len([operation for operation, _ in transport.calls if operation == "stop_comfy"]) == 1
    second = session.begin("demo")
    assert second["session"] != first["session"]
    assert second["original"]["models"] == []
    assert second["state"] == "active"


def test_disabling_restore_during_an_existing_transition_does_not_touch_a_restarted_router(resources):
    session, transport, _ = resources
    session.begin("demo")
    transport.router["birth"] = "external-restart"
    before = len(transport.calls)
    session.config = replace(session.config, restore_original_models=False)

    finished = session.finish("demo")

    assert finished["state"] == "closed"
    assert finished["original_service_restored"] is False
    assert transport.owned_engine is False
    assert not any(operation in {"snapshot", "slots", "unload", "load"} for operation, _ in transport.calls[before:])


def test_failed_admission_honors_disabled_restore_without_reloading_released_models(resources):
    session, transport, _ = resources
    session.config = replace(session.config, restore_original_models=False)
    transport.unload_failure = "beta"

    with pytest.raises(RuntimeError, match="unload failed"):
        session.begin("demo")

    assert set(transport.models) == {"beta"}
    assert record(session)["state"] == "closed"
    assert record(session)["original_service_restored"] is False
    assert not any(operation == "load" for operation, _ in transport.calls)
