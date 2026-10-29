"""The native driver waits without inference and continues exact Hermes sessions."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from server.agent_runtime.autonomous_production import (
    CONTEXT_MARKER,
    AutonomousProductionService,
    progress_digest,
    public_goal,
    resource_closed,
)
from server.agent_runtime.autonomy_store import AutonomyStore


class Queue:
    def __init__(self):
        self.stats = {"queued": 0, "running": 0}

    async def get_task_stats(self, _project):
        return dict(self.stats)


class Assistant:
    def __init__(self, store):
        self.store = store
        self.calls = []
        self.metas = {}
        self.entries = {}
        self.interruptions = []
        self.session_manager = SimpleNamespace(sessions={})
        self.event_log_store = SimpleNamespace(find_by_client_key=self.find_entry)

    async def find_entry(self, sid, key):
        return self.entries.get((sid, key))

    async def get_session(self, sid):
        return self.metas.get(sid)

    async def interrupt_session(self, sid):
        self.interruptions.append(sid)
        self.metas[sid].status = "interrupted"

    async def send_or_create(self, project, content, *, session_id=None, client_key=None, on_session_prepared=None):
        sid = session_id or "hermes-native-session-one"
        if on_session_prepared is not None:
            await on_session_prepared(sid)
            record = (await self.store.list_active())[0]
            assert record["session_id"] == sid
            assert record["pending_delivery"]["phase"] == "dispatching"
        self.metas[sid] = SimpleNamespace(status="completed")
        self.session_manager.sessions[sid] = SimpleNamespace(queued_messages=[], pending_questions={})
        self.calls.append({"project": project, "content": content, "session_id": sid, "client_key": client_key})
        entry = {"uuid": "entry-" + str(len(self.calls))}
        self.entries[sid, client_key] = entry
        return {"session_id": sid, "entry": entry, "queued_message": None}


@pytest.fixture
async def rig(db_factory):
    store = AutonomyStore(session_factory=db_factory)
    assistant = Assistant(store)
    queue = Queue()
    facts = {
        "project": "project-one",
        "episode_id": 2,
        "script_file": "episode_2.json",
        "script_state": "current",
        "units": [
            {
                "unit_id": "E2S01",
                "text": "Complete original words",
                "declared_seconds": 9,
                "audio": {"status": "current", "actual_seconds": 8.8},
                "first": {"status": "missing"},
                "end_frame": {"present": False},
                "video": {"status": "missing"},
            }
        ],
        "timelines": [],
        "active_tasks": [],
        "failed_tasks": [],
        "goal_complete_technical": False,
        "resource": {"owner_project": "project-one", "state": "restored", "original_service_restored": True},
    }
    finished = []

    async def read(_project, _episode, *, user_id):
        assert user_id == "default"
        return copy.deepcopy(facts)

    async def finish(*, project):
        finished.append(project)
        facts["resource"].update(state="restored", original_service_restored=True)

    service = AutonomousProductionService(assistant, store=store, queue=queue, facts_reader=read, finalizer=finish)
    goal = await store.create("project-one", 2, "Finish this episode autonomously", "default", "one-user-authorization")
    return SimpleNamespace(
        store=store, assistant=assistant, queue=queue, facts=facts, finished=finished, service=service, goal=goal
    )


async def test_first_delivery_binds_real_session_before_inference_and_keeps_private_outbox(rig):
    await rig.service.advance(rig.goal)
    result = await rig.store.get(rig.goal["id"], "default")
    assert result["session_id"] == "hermes-native-session-one"
    assert result["dispatch_count"] == 1
    assert result["pending_delivery"]["phase"] == "delivered"
    assert len(rig.assistant.calls) == 1
    message = rig.assistant.calls[0]["content"]
    assert message.startswith(CONTEXT_MARKER + "\n")
    context = json.loads(message.splitlines()[1])
    assert context["run_id"] == rig.goal["id"]
    assert context["authorization"]["publish"] is False
    assert "pending_delivery" not in public_goal(result)
    assert "lease_owner" not in public_goal(result)


@pytest.mark.parametrize("type_name", ["tts", "image_edit", "render_final_cut"])
async def test_any_native_writer_waits_without_a_model_turn(rig, type_name):
    rig.queue.stats.update(running=1)
    rig.facts["active_tasks"] = [{"task_id": "native-task", "task_type": type_name}]
    for _ in range(4):
        await rig.service.advance(rig.goal)
    result = await rig.store.get(rig.goal["id"], "default")
    assert result["state"] == "waiting"
    assert result["dispatch_count"] == 0
    assert rig.assistant.calls == []


async def test_only_settled_native_state_continues_the_same_hermes_session(rig):
    await rig.service.advance(rig.goal)
    sid = rig.assistant.calls[0]["session_id"]
    rig.assistant.metas[sid].status = "running"
    await rig.service.advance(rig.goal)
    assert len(rig.assistant.calls) == 1
    rig.assistant.metas[sid].status = "completed"
    rig.facts["units"][0]["first"] = {"status": "current", "sha256": "image-one"}
    await rig.service.advance(rig.goal)
    assert len(rig.assistant.calls) == 2
    assert rig.assistant.calls[1]["session_id"] == sid
    assert rig.assistant.calls[1]["client_key"] != rig.assistant.calls[0]["client_key"]


async def test_restart_after_uncertain_dispatch_reconciles_without_replaying_old_prompt(rig):
    await rig.store.claim(rig.goal["id"], "old-driver", 60, user_id="default")
    await rig.store.update(
        rig.goal["id"],
        "old-driver",
        {
            "session_id": "hermes-existing",
            "dispatch_count": 1,
            "pending_delivery": {"client_key": "old-key", "phase": "dispatching", "message": "DO NOT REPLAY THIS"},
            "lease_owner": None,
            "lease_until": 0,
        },
        user_id="default",
    )
    rig.assistant.metas["hermes-existing"] = SimpleNamespace(status="interrupted")
    await rig.service.advance(rig.goal)
    call = rig.assistant.calls[0]
    assert call["session_id"] == "hermes-existing"
    assert "DO NOT REPLAY THIS" not in call["content"]
    assert json.loads(call["content"].splitlines()[1])["mode"] == "reconcile"


async def test_final_cut_and_resource_restoration_are_required_for_goal_completion(rig):
    rig.facts["goal_complete_technical"] = True
    rig.facts["resource"].update(state="active", original_service_restored=False)
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "running"
    rig.facts["resource"].update(state="restored", original_service_restored=True)
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "completed"
    assert len(rig.assistant.calls) == 1


async def test_user_stop_drains_accepted_jobs_before_restoration_and_never_restarts(rig):
    await rig.service.advance(rig.goal)
    await rig.service.stop(rig.goal["id"], "default")
    rig.queue.stats.update(running=1)
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopping"
    assert rig.finished == []
    rig.queue.stats.update(running=0)
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopped"
    await rig.service.advance(rig.goal)
    assert rig.finished == ["project-one"]
    assert len(rig.assistant.calls) == 1


async def test_unchanged_turns_do_not_loop_forever_or_report_a_false_completion(rig):
    for _ in range(5):
        await rig.service.advance(rig.goal)
    result = await rig.store.get(rig.goal["id"], "default")
    assert result["state"] == "attention"
    assert result["last_error"] == "no_native_progress_after_three_turns"
    assert len(rig.assistant.calls) == 3
    assert rig.finished == ["project-one"]


def test_failed_job_ids_and_timestamps_are_not_media_progress(rig):
    before = progress_digest(rig.facts)
    rig.facts["failed_tasks"] = [{"task_id": "new-uuid", "error_message": "same missing image"}]
    rig.facts["updated_at"] = "a new clock reading"
    assert progress_digest(rig.facts) == before


def test_asset_and_preparation_advancement_count_as_progress(rig):
    before = progress_digest(rig.facts)
    rig.facts["assets"] = [{"resource_type": "prop", "resource_id": "jar", "selected_version": 2, "sha256": "clean"}]
    assert progress_digest(rig.facts) != before
    before = progress_digest(rig.facts)
    rig.facts["preparation"] = {"draft_fingerprint": "new-native-draft"}
    assert progress_digest(rig.facts) != before


@pytest.mark.parametrize("state", [None, "unknown", "unreadable", "active", "restored"])
def test_unknown_or_unconfirmed_resource_state_cannot_complete(state):
    assert not resource_closed({"resource": {"owner_project": None, "state": state}}, "project-one")


async def test_unreadable_resource_record_does_not_silently_complete(rig):
    rig.facts["goal_complete_technical"] = True
    rig.facts["resource"] = {"owner_project": None, "state": "unreadable"}
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "running"


@pytest.mark.parametrize("existing_session", [False, True])
async def test_stop_before_dispatch_admission_prevents_inference(rig, existing_session):
    if existing_session:
        await rig.service.advance(rig.goal)
        rig.assistant.calls.clear()
        rig.facts["units"][0]["first"] = {"status": "current", "sha256": "new-image"}
    original_send = rig.assistant.send_or_create

    async def stop_before_prepared(*args, **kwargs):
        await rig.service.stop(rig.goal["id"], "default")
        return await original_send(*args, **kwargs)

    rig.assistant.send_or_create = stop_before_prepared
    await rig.service.advance(rig.goal)
    assert rig.assistant.calls == []
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopping"
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] == "stopped"


@pytest.mark.parametrize("resumed_writer", ["session", "media_task"])
async def test_work_resumed_during_facts_probe_prevents_premature_completion(rig, resumed_writer):
    await rig.service.advance(rig.goal)
    original_read = rig.service.facts_reader

    async def read_with_resume(*args, **kwargs):
        result = await original_read(*args, **kwargs)
        result["goal_complete_technical"] = True
        if resumed_writer == "session":
            rig.assistant.metas[rig.assistant.calls[0]["session_id"]].status = "running"
        else:
            rig.queue.stats.update(running=1)
        return result

    rig.service.facts_reader = read_with_resume
    await rig.service.advance(rig.goal)
    assert (await rig.store.get(rig.goal["id"], "default"))["state"] != "completed"
    assert len(rig.assistant.calls) == 1
    assert rig.finished == []
