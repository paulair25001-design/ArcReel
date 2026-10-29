"""A reused current render cannot substitute for native final delivery actions."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lib.db.base import DEFAULT_USER_ID
from lib.local_runtime.config import get_local_runtime_config
from server.services.project.native_delivery import read_native_delivery_status


@pytest.fixture
def delivery_case(tmp_path):
    config = replace(get_local_runtime_config(), hermes_home=tmp_path / "hermes", hermes_model="local-qwen")
    home = config.hermes_home / "sessions/native-parent"
    home.mkdir(parents=True)
    bindings = config.hermes_home / "bindings"
    bindings.mkdir()
    binding = bindings / "parent-sid.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "arcreel-hermes-session/v1",
                "home": "sessions/native-parent",
                "user_id": DEFAULT_USER_ID,
            }
        )
    )
    return {
        "config": config,
        "binding": binding,
        "audit": home / "vision-wire-audit.jsonl",
        "reports": home / "native-agent-reports.jsonl",
    }


def _review(timestamp=100, **changes):
    return {
        "schema": "arcreel-native-agent-report/v1",
        "project": "demo",
        "parent_session_id": "parent-sid",
        "session_id": "review-child",
        "agent_type": "review-footage",
        "status": "completed",
        "report": "Native footage report; concerns remain explicitly recorded.",
        "model": "local-qwen",
        "timestamp": timestamp,
        **changes,
    }


def _event(event="native_finalization", timestamp=120, **changes):
    return {
        "protocol": "arcreel-mcp-vision/v2",
        "event": event,
        "project_name": "demo",
        "session_id": "parent-sid",
        "parent_session_id": None,
        "timestamp": timestamp,
        "tool_name": "finish_local_session" if event == "native_finalization" else "edit_timeline",
        "original_service_restored": True,
        "source_sha256": "a" * 64,
        **changes,
    }


def _write(case, *, reports=None, events=None):
    case["reports"].write_text(
        "".join(json.dumps(row) + "\n" for row in (reports if reports is not None else [_review()]))
    )
    case["audit"].write_text("".join(json.dumps(row) + "\n" for row in (events or [])))


async def _read(case, **kwargs):
    return await read_native_delivery_status("demo", "parent-sid", config=case["config"], **kwargs)


async def test_current_artifacts_or_model_claims_cannot_replace_the_actual_finalization_event(delivery_case):
    _write(
        delivery_case,
        events=[
            {"event": "assistant", "session_id": "parent-sid", "content": "Old render current; batch gb45 is running."},
            _event("mcp_images"),
        ],
    )
    result = await _read(delivery_case)

    assert result["ready"] is False
    assert result["problem"] == "native_finalization_missing"
    assert result["action"] == "finish_local_session"
    assert result["finalization"] is None


async def test_parent_finalization_after_review_and_real_write_establishes_delivery_only(delivery_case):
    _write(delivery_case, events=[_event("native_write_success", 110), _event(timestamp=120)])
    result = await _read(delivery_case)

    assert result["ready"] is True
    assert result["last_review_timestamp"] == 100
    assert result["last_write_timestamp"] == 110
    assert result["finalization"] == {"timestamp": 120, "source_sha256": "a" * 64, "tool_name": "finish_local_session"}
    assert result["action"] is None
    assert "quality" not in result
    assert "approved" not in result


@pytest.mark.parametrize("late_event", ["review", "write", "child_write"])
async def test_new_native_work_invalidates_a_previous_finalization(delivery_case, late_event):
    reports = [_review()]
    events = [_event(timestamp=120)]
    if late_event == "review":
        reports.append(_review(130, session_id="second-review-child"))
    else:
        extra = {"session_id": "writer-child", "parent_session_id": "parent-sid"} if late_event == "child_write" else {}
        events.append(_event("native_write_success", 130, **extra))
    _write(delivery_case, reports=reports, events=events)

    outdated = await _read(delivery_case)
    assert outdated["ready"] is False
    assert outdated["problem"] == "native_finalization_outdated"

    events.append(_event(timestamp=140))
    _write(delivery_case, reports=reports, events=events)
    assert (await _read(delivery_case))["ready"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": "other-parent"},
        {"session_id": "child-sid", "parent_session_id": "parent-sid"},
        {"project_name": "another-project"},
        {"protocol": "arcreel-mcp-vision/v1"},
        {"original_service_restored": False},
        {"original_service_restored": 1},
        {"tool_name": "get_episode_production_status"},
    ],
)
async def test_other_scope_child_or_unsuccessful_finish_is_not_final_delivery(delivery_case, changes):
    _write(delivery_case, events=[_event(**changes)])
    result = await _read(delivery_case)
    assert result["ready"] is False
    assert result["problem"] == "native_finalization_missing"


@pytest.mark.parametrize("stamp", [100, 99])
async def test_finalization_must_be_strictly_later_than_native_review(delivery_case, stamp):
    _write(delivery_case, events=[_event(timestamp=stamp)])
    assert (await _read(delivery_case))["problem"] == "native_finalization_outdated"


@pytest.mark.parametrize("stamp", [None, True, "120", float("nan"), float("inf"), -1])
async def test_unorderable_action_timestamps_fail_closed(delivery_case, stamp):
    _write(delivery_case, events=[_event(timestamp=120), _event("native_write_success", timestamp=stamp)])
    result = await _read(delivery_case)
    assert result["ready"] is False
    assert result["problem"] == "native_delivery_event_invalid"


async def test_unorderable_review_and_missing_source_hash_do_not_acknowledge_final_delivery(delivery_case):
    _write(delivery_case, reports=[_review(None)], events=[_event()])
    assert (await _read(delivery_case))["problem"] == "native_review_timestamp_invalid"
    _write(delivery_case, events=[_event(source_sha256=None)])
    assert (await _read(delivery_case))["problem"] == "native_delivery_event_invalid"


async def test_unrelated_work_and_finish_duplicate_are_not_new_parent_production(delivery_case):
    _write(
        delivery_case,
        reports=[_review(), _review(999, parent_session_id="other-parent")],
        events=[
            _event(timestamp=120),
            _event("native_write_success", 999, tool_name="finish_local_session"),
            _event("native_write_success", 999, project_name="another-project"),
            _event("native_write_success", 999, session_id="other-parent"),
        ],
    )
    assert (await _read(delivery_case))["ready"] is True


async def test_owned_parent_binding_is_required_after_cold_recovery(delivery_case):
    _write(delivery_case, events=[_event()])
    assert (await _read(delivery_case))["ready"] is True
    binding = json.loads(delivery_case["binding"].read_text())
    binding["user_id"] = "other-user"
    delivery_case["binding"].write_text(json.dumps(binding))
    result = await _read(delivery_case)
    assert result["ready"] is False
    assert result["problem"] == "session_binding_unavailable"


async def test_partial_append_does_not_destroy_prior_successful_host_actions(delivery_case):
    _write(delivery_case, events=[_event()])
    with delivery_case["audit"].open("a") as stream:
        stream.write('{"event":')
    assert (await _read(delivery_case))["ready"] is True


async def test_missing_completed_review_never_uses_zero_as_its_timestamp(delivery_case):
    _write(delivery_case, reports=[_review(status="incomplete")], events=[_event()])
    result = await _read(delivery_case)
    assert result["ready"] is False
    assert result["problem"] == "native_review_report_missing"


@pytest.mark.parametrize("created_at", [120, 130])
async def test_new_goal_reusing_owned_sid_cannot_reuse_previous_goals_finalization(delivery_case, created_at):
    _write(delivery_case, events=[_event(timestamp=120)])
    assert (await _read(delivery_case))["ready"] is True

    reused = await _read(delivery_case, not_before=created_at)
    assert reused["ready"] is False
    assert reused["problem"] == "native_finalization_outdated"
    assert reused["not_before"] == created_at
    assert reused["last_review_timestamp"] == 100

    # The same completed native reports remain valid; only delivery is renewed.
    _write(delivery_case, events=[_event(timestamp=120), _event(timestamp=created_at + 1)])
    renewed = await _read(delivery_case, not_before=created_at)
    assert renewed["ready"] is True
    assert renewed["last_review_timestamp"] == 100


async def test_finalization_must_exceed_both_new_goal_creation_and_later_native_work(delivery_case):
    _write(delivery_case, events=[_event("native_write_success", 150), _event(timestamp=140)])
    assert (await _read(delivery_case, not_before=130))["ready"] is False
    _write(delivery_case, events=[_event("native_write_success", 150), _event(timestamp=151)])
    assert (await _read(delivery_case, not_before=130))["ready"] is True


@pytest.mark.parametrize("cutoff", [True, -1, float("nan"), float("inf"), "130"])
async def test_invalid_goal_creation_cutoff_cannot_allow_old_finalization(delivery_case, cutoff):
    _write(delivery_case, events=[_event(timestamp=120)])
    result = await _read(delivery_case, not_before=cutoff)
    assert result["ready"] is False
    assert result["problem"] == "native_delivery_not_before_invalid"


async def test_zero_creation_cutoff_retains_normal_delivery_ordering(delivery_case):
    _write(delivery_case, events=[_event(timestamp=120)])
    assert (await _read(delivery_case, not_before=0))["ready"] is True
