"""Durable, event-driven continuation of one native Hermes production goal.

The driver never creates media payloads or edits creative content. ArcReel's
normal tools, queue and artifacts remain authoritative; Hermes chooses the next
native action after a committed state change. Waiting only reads runtime state.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lib.generation.generation_queue import get_generation_queue
from lib.infra.logging_utils import redact_diagnostic_text
from lib.local_runtime.config import local_only_enabled
from lib.local_runtime.l20_resource_state import resource_handoff_complete
from lib.project.project_change_hints import register_project_change_batch_listener
from server.agent_runtime.autonomy_store import ACTIVE_STATES, AutonomyStore
from server.services.project.production_status import read_episode_production_status

if TYPE_CHECKING:
    from server.agent_runtime.service import AssistantService

logger = logging.getLogger(__name__)
CONTEXT_MARKER = "[ARCREEL_AUTONOMOUS_PRODUCTION]"
LEASE_SECONDS = 60.0


def public_goal(record: dict[str, Any]) -> dict[str, Any]:
    """Only the user-visible goal; delivery prompts and leases stay internal."""
    fields = (
        "id",
        "project_name",
        "episode_id",
        "objective",
        "session_id",
        "state",
        "revision",
        "dispatch_count",
        "last_error",
        "created_at",
        "updated_at",
    )
    return {key: record.get(key) for key in fields}


def compact_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Summarize actual outputs without copying a second production ledger."""
    units = []
    for unit in facts.get("units", []):
        media = {}
        for kind in ("audio", "first", "video"):
            value = unit.get(kind, {})
            media[kind] = {
                key: value.get(key) for key in ("status", "present", "actual_seconds", "selected_version", "sha256")
            }
            if kind == "video":
                media[kind].update({key: value[key] for key in ("provider_audio", "has_audio") if key in value})
        units.append(
            {
                "unit_id": unit.get("unit_id"),
                "text": unit.get("text"),
                "declared_seconds": unit.get("declared_seconds"),
                **({"speech_timing": unit["speech_timing"]} if "speech_timing" in unit else {}),
                **({"speech": unit["speech"]} if "speech" in unit else {}),
                **media,
                "end_frame": unit.get("end_frame", {}),
            }
        )
    resource = facts.get("resource", {})
    return {
        "project": facts.get("project"),
        "episode_id": facts.get("episode_id"),
        "script_file": facts.get("script_file"),
        "script_state": facts.get("script_state"),
        "preparation": facts.get("preparation", {}),
        "workflow": facts.get("workflow", {}),
        "visual_observation": facts.get("visual_observation", {}),
        "assets": [
            {
                key: asset.get(key)
                for key in ("resource_type", "resource_id", "status", "present", "selected_version", "sha256")
            }
            for asset in facts.get("assets", [])
        ],
        "units": units,
        "active_tasks": facts.get("active_tasks", []),
        "failed_tasks": facts.get("failed_tasks", [])[:8],
        "timelines": facts.get("timelines", []),
        "native_review": {
            "ready": facts.get("native_review", {}).get("ready"),
            "missing": facts.get("native_review", {}).get("missing", []),
            "reviewed": [
                {key: row.get(key) for key in ("unit_id", "version")}
                for row in facts.get("native_review", {}).get("reviewed", [])
            ],
            "reports": facts.get("native_review", {}).get("reports", []),
        },
        "native_delivery": {key: facts.get("native_delivery", {}).get(key) for key in ("ready", "action", "problem")},
        "goal_complete_technical": bool(facts.get("goal_complete_technical")),
        "resource": {
            key: resource.get(key)
            for key in (
                "owner_project",
                "project",
                "state",
                "original_service_restored",
                "restore_original_models",
                "resource_released",
            )
        },
    }


def progress_digest(facts: dict[str, Any]) -> str:
    """Only changed content/output state counts as progress, not new error IDs."""
    value = compact_facts(facts)
    value.pop("failed_tasks", None)
    value.pop("active_tasks", None)
    value.pop("workflow", None)
    value.get("native_review", {}).pop("reports", None)

    # Version timestamps and transient runtime identifiers do not prove progress.
    def stable(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: stable(child)
                for key, child in item.items()
                if key not in {"created_at", "updated_at", "rendered_at", "poll_after_seconds", "pid", "session"}
            }
        if isinstance(item, list):
            return [stable(child) for child in item]
        return item

    body = json.dumps(stable(value), sort_keys=True, ensure_ascii=False, default=str, allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def resource_closed(facts: dict[str, Any], project: str) -> bool:
    resource = facts.get("resource", {})
    owner = resource.get("owner_project", resource.get("project"))
    state = resource.get("state")
    if state in {None, "unknown", "unreadable"}:
        return False
    if state in {"not_started", "missing"}:
        return owner is None
    if owner is not None and owner != project:
        return True
    return resource_handoff_complete(resource)


def continuation_message(goal: dict[str, Any], facts: dict[str, Any], mode: str) -> str:
    context = {
        "run_id": goal["id"],
        "project": goal["project_name"],
        "episode_id": goal["episode_id"],
        "objective": goal["objective"],
        "mode": mode,
        "authorization": {
            "end_to_end": True,
            "content_confirmation": True,
            "local_generation": True,
            "free_local_voice_only": True,
            "publish": False,
        },
        "facts": compact_facts(facts),
    }
    return (
        CONTEXT_MARKER + "\n" + json.dumps(context, ensure_ascii=False, default=str) + "\n\n"
        "这是用户通过 ArcReel 原生自动制作入口给予的持续授权。只推进该项目和该集到原生最终成片。"
        "读取 get_episode_production_status 与 get_workflow_plan，根据当前事实选择原生工具，保留已完成内容。"
        "全部推理只用本地 Hermes/oMLX、L20 和免费本地配音，不调用 Codex、云端或旧制作工作流。"
        "授权范围内的内容确认、低风险制作选择与有据的最小修复自行完成，无需逐步问用户。"
        "先经 read_agent_profile 读取当前模式的原生 video-workflow，再按原生技能派发 dispatch_agent。"
        "资产与分镜按原生 review-images 检查；剪辑按原生 edit-video 首轮全量 review-footage，不能直接跳到渲染。"
        "native_review.ready 只表示报告覆盖，并非可用结论；按 reports 原文处理问题、选版、修复和复审，"
        "未修复的问题按原生交付要求如实列出，不能把已有 current 当作质量通过。"
        "保留原生修复后比较新旧与停止规则，不另建审核评分或无限重抽。"
        "按原生 speech 发声归属处理声音：角色对白保留视频原声音轨，不要求旁白 WAV；"
        "画外旁白使用本地 TTS 并静音该镜头的视频原声，完整旁白按实测容纳；静默镜头可保留环境声。"
        "发声归属不合法时先按原生要求修复，不把角色对白送入旁白 TTS。等待全部写入任务终态后继续。"
        "按 speech_timing 先核对估算与实测配音时长，估算不等于实测；保留语义停顿，"
        "不要把超长口播硬塞进短镜头，也不要靠长定格或反复重播同一动作容纳全部旁白。"
        "剪辑前读取 editing_facts：复用片段的每次入出点都须在原生审片报告的可用区间内，"
        "同镜头尾接头的溶解不是不同镜头的衔接；按剧情处理光向、动作和声音承接。"
        "这些时长事实不是新质量评分或额外审批，不自动重生成已有素材。"
        "调用生成或编辑工具得到在途任务后，结束本轮并说明等待；后台服务会在原生任务完成后继续同一会话，"
        "不要反复轮询模型或自行启动后台脚本。所有已有提交先对账，不重放旧工具调用。"
        "最终目标是指定集的 current 成片，旁白项目需完整旁白和烧录字幕，随后 finish_local_session 按当前模型恢复设置真实收尾。"
        "旧成片存在和审片报告齐全不表示本目标已交付；以实际队列与版本为准，不把自己提过的批次名当作已受理任务。"
        "不要把一次 end_turn、视频片段齐备或创建时间线当成最终交付。没有实际听审或连续播放证据时不宣称已通过。"
        "看图以 inspect_images 经本地视觉通道实际送入的图片为据；历史 MEDIA 路径、文本占位或模型自述都不是视觉证据。"
        + (
            "\n这是恢复对账轮。先读取当前项目、任务和正式产物；上一轮投递可能已执行，禁止重放上一轮动作。"
            "使用当前视觉通道重新观察已有首图和尾图候选后再进入视频阶段，保留已生成图片，不因恢复而重生成。"
            if mode == "reconcile"
            else ""
        )
    )


class AutonomousProductionService:
    def __init__(
        self,
        assistant: AssistantService,
        *,
        store: AutonomyStore | None = None,
        queue: Any = None,
        facts_reader: Callable[..., Awaitable[dict[str, Any]]] = read_episode_production_status,
        finalizer: Callable[..., Awaitable[Any]] | None = None,
        poll_seconds: float = 2,
    ) -> None:
        self.assistant = assistant
        self.store = store or AutonomyStore()
        self.queue = queue or get_generation_queue()
        self.facts_reader = facts_reader
        self._finalizer = finalizer
        self.poll_seconds = poll_seconds
        self.owner_id = "autonomy-" + uuid4().hex
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._unregister: Callable[[], None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._runner is not None:
            return
        self._closed = False
        loop = asyncio.get_running_loop()

        def changed(_project: str, _source: Any, batches: tuple[dict[str, Any], ...]) -> None:
            if any(str(batch.get("action", "")).startswith("task_") for batch in batches):
                loop.call_soon_threadsafe(self._wake.set)

        self._unregister = register_project_change_batch_listener(changed)
        self._runner = asyncio.create_task(self._run(), name="native-hermes-autonomous-production")
        self._wake.set()

    def notify(self) -> None:
        self._wake.set()

    async def close(self) -> None:
        self._closed = True
        if self._unregister is not None:
            self._unregister()
            self._unregister = None
        if self._runner is not None:
            self._runner.cancel()
        tasks = [*self._inflight.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*(tasks + ([self._runner] if self._runner else [])), return_exceptions=True)
        self._runner = None
        self._inflight.clear()

    async def create(
        self,
        project: str,
        episode: int,
        objective: str,
        user_id: str,
        client_key: str,
        *,
        resume_session_id: str | None = None,
    ) -> dict[str, Any]:
        if not local_only_enabled():
            raise ValueError("Autonomous production requires the explicit local-only runtime")
        # Project identity and episode existence are checked by the same native facts seam.
        await self.facts_reader(project, episode, user_id=user_id)
        if resume_session_id is not None:
            from re import fullmatch

            from lib.local_runtime.config import get_local_runtime_config
            from server.agent_runtime.hermes_client import recorded_hermes_home

            if fullmatch(r"[A-Za-z0-9_-]{1,200}", resume_session_id) is None:
                raise ValueError("Invalid Hermes session identity")

            def owned_binding() -> bool:
                config = get_local_runtime_config()
                home = recorded_hermes_home(config, resume_session_id, user_id)
                if home is None or not home.is_dir() or home.is_symlink():
                    return False
                marker = home / ".arcreel-local-hermes"
                return not marker.is_symlink() and marker.read_bytes() == b"arcreel-local-hermes/v1\n"

            try:
                owned = await asyncio.to_thread(owned_binding)
            except (OSError, RuntimeError, ValueError):
                raise ValueError("The Hermes session has no valid owned binding for this user") from None
            if not owned:
                raise ValueError("The Hermes session has no valid owned binding for this user")
            # Read metadata after the binding I/O; a fork may have superseded
            # the requested session while that thread was running.
            meta = await self.assistant.get_session(resume_session_id)
            if meta is None or meta.project_name != project or meta.superseded_by is not None:
                raise ValueError("The Hermes session is missing, superseded, or belongs to another project")
        goal = await self.store.create(project, episode, objective, user_id, client_key, session_id=resume_session_id)
        self.notify()
        return goal

    async def stop(self, run_id: str, user_id: str) -> dict[str, Any]:
        result = await self.store.stop(run_id, user_id)
        self.notify()
        return result

    async def retry(self, run_id: str, user_id: str, expected_revision: int) -> dict[str, Any]:
        result = await self.store.retry(run_id, user_id, expected_revision)
        self.notify()
        return result

    async def stop_for_session(self, project: str, session_id: str) -> None:
        for goal in await self.store.list_active():
            if goal["project_name"] == project and goal.get("session_id") == session_id:
                await self.store.stop(goal["id"], goal["user_id"])
        self.notify()

    async def _run(self) -> None:
        while not self._closed:
            self._wake.clear()
            try:
                for goal in await self.store.list_active():
                    current = self._inflight.get(goal["id"])
                    if current is None or current.done():
                        task = asyncio.create_task(self.advance(goal), name="autonomy:" + goal["id"])
                        self._inflight[goal["id"]] = task
                        task.add_done_callback(self._observe_advance)
            except Exception:
                logger.exception("Native autonomous goal reconciliation failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)

    @staticmethod
    def _observe_advance(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Native goal reconciliation failed: %s", redact_diagnostic_text(error))

    async def _heartbeat(self, goal: dict[str, Any]) -> None:
        while True:
            await asyncio.sleep(LEASE_SECONDS / 3)
            if await self.store.claim(goal["id"], self.owner_id, LEASE_SECONDS, user_id=goal["user_id"]) is None:
                return

    async def _update(self, goal: dict[str, Any], **changes: Any) -> dict[str, Any]:
        return await self.store.update(goal["id"], self.owner_id, changes, user_id=goal["user_id"])

    def _managed(self, session_id: str | None) -> Any:
        return self.assistant.session_manager.sessions.get(session_id) if session_id else None

    async def _session_busy(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        meta = await self.assistant.get_session(session_id)
        managed = self._managed(session_id)
        return bool(
            (meta is not None and meta.status == "running")
            or (
                managed is not None
                and (
                    any(message.state != "unsent" for message in managed.queued_messages)
                    or getattr(managed, "pending_questions", {})
                )
            )
        )

    async def _finish_resources(self, project: str) -> None:
        if self._finalizer is None:
            from lib.local_runtime.l20_resource_session import finalize_l20_runtime

            await finalize_l20_runtime(project=project)
        else:
            await self._finalizer(project=project)

    async def _stop_owned(self, goal: dict[str, Any]) -> None:
        session_id = goal.get("session_id")
        if session_id and await self._session_busy(session_id):
            await self.assistant.interrupt_session(session_id)
            return
        stats = await self.queue.get_task_stats(goal["project_name"])
        if stats.get("queued") or stats.get("running"):
            return
        await self._finish_resources(goal["project_name"])
        await self._update(goal, state="stopped", last_error=None)

    async def advance(self, candidate: dict[str, Any]) -> None:
        claimed = await self.store.claim(candidate["id"], self.owner_id, LEASE_SECONDS, user_id=candidate["user_id"])
        if claimed is None:
            return
        goal: dict[str, Any] = claimed
        heartbeat = asyncio.create_task(self._heartbeat(goal))
        try:
            if goal["state"] == "stopping":
                await self._stop_owned(goal)
                return
            stats = await self.queue.get_task_stats(goal["project_name"])
            if stats.get("queued") or stats.get("running"):
                if goal["state"] != "waiting":
                    await self._update(goal, state="waiting")
                return
            if await self._session_busy(goal.get("session_id")):
                return
            facts = await self.facts_reader(goal["project_name"], goal["episode_id"], user_id=goal["user_id"])
            if facts.get("active_tasks"):
                await self._update(goal, state="waiting")
                return
            # Files and probes may yield while a user or native CLI resumes work.
            # Reconcile both writers and the session before any terminal decision.
            if await self._session_busy(goal.get("session_id")):
                return
            stats = await self.queue.get_task_stats(goal["project_name"])
            if stats.get("queued") or stats.get("running"):
                await self._update(goal, state="waiting")
                return
            if (
                facts.get("goal_complete_technical")
                and facts.get("native_delivery", {}).get("ready") is True
                and resource_closed(facts, goal["project_name"])
            ):
                await self._update(goal, state="completed", last_error=None)
                return
            digest = progress_digest(facts)
            count = int(goal.get("unchanged_turns", 0))
            if goal.get("dispatch_count", 0):
                count = count + 1 if digest == goal.get("last_progress_digest") else 0
            if count >= 3:
                await self._finish_resources(goal["project_name"])
                await self._update(
                    goal, state="attention", last_error="no_native_progress_after_three_turns", unchanged_turns=count
                )
                return
            pending = goal.get("pending_delivery") or {}
            mode = "continue" if goal.get("dispatch_count") else ("reconcile" if goal.get("session_id") else "start")
            if pending and pending.get("phase") != "delivered":
                session_id = goal.get("session_id")
                entry = (
                    await self.assistant.event_log_store.find_by_client_key(session_id, pending["client_key"])
                    if session_id
                    else None
                )
                # A missing entry never proves that a previous prompt was not executed.
                # Every recovery is a new state-reconciliation turn, never a replay.
                if entry is None or pending.get("phase") in {"dispatching", "reconcile_required"}:
                    mode = "reconcile"
            session_id = goal.get("session_id")
            if session_id:
                meta = await self.assistant.get_session(session_id)
                if meta is None:
                    await self._update(goal, state="attention", last_error="autonomous_session_missing")
                    await self._finish_resources(goal["project_name"])
                    return
                if meta.status in {"error", "interrupted"}:
                    mode = "reconcile"
            sequence = int(goal.get("dispatch_count", 0)) + 1
            delivery = {
                "client_key": f"autonomy:{goal['id']}:{sequence}",
                "phase": "prepared",
                "mode": mode,
                "created_at": datetime.now(UTC).isoformat(),
                "message": continuation_message(goal, facts, mode),
            }
            if "retry_of_revision" in pending:
                delivery["retry_of_revision"] = pending["retry_of_revision"]
            goal = await self._update(
                goal,
                state="running",
                dispatch_count=sequence,
                pending_delivery=delivery,
                last_progress_digest=digest,
                unchanged_turns=count,
                last_error=None,
            )

            async def prepared(sdk_id: str) -> None:
                nonlocal goal, delivery
                delivery = {**delivery, "phase": "dispatching"}
                goal = await self._update(goal, state="running", session_id=sdk_id, pending_delivery=delivery)

            response = await self.assistant.send_or_create(
                goal["project_name"],
                delivery["message"],
                session_id=session_id,
                client_key=delivery["client_key"],
                on_session_prepared=prepared,
            )
            delivery = {**delivery, "phase": "delivered" if response.get("entry") else "queued"}
            await self._update(goal, session_id=response["session_id"], pending_delivery=delivery)
        except asyncio.CancelledError:
            # The durable objective is left active for startup reconciliation.
            raise
        except Exception as error:
            logger.warning("Native autonomous dispatch failed run=%s: %s", goal["id"], redact_diagnostic_text(error))
            with contextlib.suppress(ValueError):
                latest = await self.store.get(goal["id"], goal["user_id"])
                if latest is not None and latest["state"] in ACTIVE_STATES:
                    failures = int(latest.get("unchanged_turns", 0)) + 1
                    changes: dict[str, Any] = {
                        "last_error": redact_diagnostic_text(error)[:1800],
                        "unchanged_turns": failures,
                    }
                    if failures >= 3:
                        stats = await self.queue.get_task_stats(goal["project_name"])
                        if not stats.get("queued") and not stats.get("running"):
                            with contextlib.suppress(Exception):
                                await self._finish_resources(goal["project_name"])
                            changes["state"] = "attention"
                    await self._update(goal, **changes)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            with contextlib.suppress(ValueError):
                await self._update(goal, lease_owner=None, lease_until=0)


_service: AutonomousProductionService | None = None


def bind_autonomous_production(service: AutonomousProductionService | None) -> None:
    global _service
    _service = service


def get_autonomous_production() -> AutonomousProductionService | None:
    return _service
