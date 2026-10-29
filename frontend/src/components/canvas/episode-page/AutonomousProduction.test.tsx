import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LeaveGuardProvider, useLeaveGuard } from "@/components/shared/edit-unit/LeaveGuard";
import { useAppStore } from "@/stores/app-store";
import { useAssistantStore } from "@/stores/assistant-store";
import { useConfigStatusStore } from "@/stores/config-status-store";
import type { AutonomousProductionGoal } from "@/types/autonomy";

import { AutonomousProduction } from "./AutonomousProduction";

const GOAL: AutonomousProductionGoal = {
  id: "goal-one", project_name: "project-one", episode_id: 7,
  objective: "完成本集，保留已有素材", session_id: null, state: "pending", revision: 1,
  dispatch_count: 0, last_error: null, created_at: "2026-10-29T06:00:00Z", updated_at: "2026-10-29T06:00:00Z",
};

function response(goal: AutonomousProductionGoal | null) {
  return new Response(JSON.stringify({ goal }), { headers: { "Content-Type": "application/json" } });
}

function transport(initial: AutonomousProductionGoal | null = null) {
  let goal = initial;
  const writes: Array<{ url: string; options: RequestInit }> = [];
  const reads: AbortSignal[] = [];
  let rejectNextCreate = false;
  let conflictNextRetry = false;
  const fetch = vi.fn(async (input: RequestInfo | URL, options: RequestInit = {}) => {
    const url = String(input);
    if (options.method === "POST") {
      writes.push({ url, options });
      if (url.endsWith("/stop")) goal = { ...GOAL, ...goal, state: "stopping" };
      else if (url.endsWith("/retry")) {
        if (conflictNextRetry) {
          conflictNextRetry = false;
          goal = { ...GOAL, ...goal, revision: (goal?.revision ?? 0) + 1 };
          return new Response(JSON.stringify({ detail: "目标版本已变化" }), { status: 409 });
        }
        goal = { ...GOAL, ...goal, state: "pending", revision: (goal?.revision ?? 0) + 1, last_error: null };
      } else {
        if (rejectNextCreate) {
          rejectNextCreate = false;
          throw new TypeError("connection lost");
        }
        const body = JSON.parse(String(options.body));
        goal = { ...GOAL, episode_id: body.episode_id, objective: body.objective };
      }
    } else if (options.signal) reads.push(options.signal);
    return response(goal);
  });
  vi.stubGlobal("fetch", fetch);
  return {
    fetch, writes, reads,
    setGoal(value: AutonomousProductionGoal | null) { goal = value; },
    rejectNext() { rejectNextCreate = true; },
    conflictRetry() { conflictNextRetry = true; },
  };
}

async function openEntry() {
  fireEvent.click(screen.getByRole("button", { name: /^自动制作/ }));
  return screen.findByRole("dialog", { name: "自动制作" });
}

beforeEach(() => {
  useConfigStatusStore.setState(useConfigStatusStore.getInitialState(), true);
  useConfigStatusStore.setState({ executionRuntime: {
    mode: "local-hermes-l20", agent_available: true, agent_model: "local-model",
    image_model: "local-image", video_model: "local-video", audio_model: "local-tts", cloud_fallback: false,
  } });
  useAssistantStore.setState(useAssistantStore.getInitialState(), true);
  useAppStore.setState(useAppStore.getInitialState(), true);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("native automatic production entry", () => {
  it("leaves the manual runtime untouched without a local execution mode", () => {
    const http = transport();
    useConfigStatusStore.setState({ executionRuntime: null });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    expect(screen.queryByRole("button", { name: /自动制作/ })).not.toBeInTheDocument();
    expect(http.fetch).not.toHaveBeenCalled();
  });

  it("sends one explicit episode goal with a stable client key and waits for its session", async () => {
    const http = transport();
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    expect(within(dialog).getByLabelText("制作目标")).toHaveValue("按本集现有内容自动制作至带旁白字幕的最终成片");
    expect(within(dialog).getByText(/启动即授权 Hermes/)).toBeInTheDocument();
    const button = within(dialog).getByRole("button", { name: "开始自动制作" });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.change(within(dialog).getByLabelText("制作目标"), { target: { value: "  完成本集，保留已有素材  " } });
    fireEvent.click(button);
    const current = await within(dialog).findByRole("region", { name: "当前制作目标" });
    expect(within(current).getByText("完成本集，保留已有素材")).toBeInTheDocument();
    expect(within(dialog).getByText("Hermes 正在启动，会话就绪后可从这里打开。")).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "打开 Hermes 会话" })).not.toBeInTheDocument();
    expect(http.writes).toHaveLength(1);
    expect(http.writes[0].url).toBe("/api/v1/projects/project-one/autonomous-production");
    expect(JSON.parse(String(http.writes[0].options.body))).toEqual({ episode_id: 7, objective: "完成本集，保留已有素材", client_key: expect.any(String) });
  });

  it("reuses its idempotency key after a lost response instead of silently resubmitting", async () => {
    const http = transport();
    http.rejectNext();
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    const start = within(dialog).getByRole("button", { name: "开始自动制作" });
    await waitFor(() => expect(start).toBeEnabled());
    fireEvent.click(start);
    expect(await within(dialog).findByText("connection lost")).toBeInTheDocument();
    await waitFor(() => expect(start).toBeEnabled());
    expect(http.writes).toHaveLength(1);
    fireEvent.click(start);
    expect(await within(dialog).findByText("Hermes 正在启动，会话就绪后可从这里打开。")).toBeInTheDocument();
    expect(http.writes).toHaveLength(2);
    expect(http.writes[1].options.body).toBe(http.writes[0].options.body);
  });

  it("stops the accepted goal through the native endpoint and disables repeated stopping", async () => {
    const http = transport({ ...GOAL, state: "running", session_id: "hermes-session" });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    fireEvent.click(await within(dialog).findByRole("button", { name: "停止自动制作" }));
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "停止自动制作" })).toBeDisabled());
    expect(await within(dialog).findByText("正在停止")).toBeInTheDocument();
    expect(http.writes).toHaveLength(1);
    expect(http.writes[0].url).toBe("/api/v1/projects/project-one/autonomous-production/goal-one/stop");
  });

  it("retries an attention goal with its revision while retaining its identity and Hermes session", async () => {
    const http = transport({ ...GOAL, state: "attention", revision: 12, session_id: "existing-hermes-session", dispatch_count: 4, last_error: "视觉上下文需要恢复" });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    const retry = await within(dialog).findByRole("button", { name: "重试同一目标" });
    expect(within(dialog).queryByRole("textbox", { name: "制作目标" })).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "开始自动制作" })).not.toBeInTheDocument();
    fireEvent.click(retry);
    const current = await within(dialog).findByRole("region", { name: "当前制作目标" });
    expect(await within(current).findByText("启动中")).toBeInTheDocument();
    expect(within(current).getByText(GOAL.objective)).toBeInTheDocument();
    expect(http.writes).toHaveLength(1);
    expect(http.writes[0].url).toBe("/api/v1/projects/project-one/autonomous-production/goal-one/retry");
    expect(JSON.parse(String(http.writes[0].options.body))).toEqual({ expected_revision: 12 });
    expect(within(dialog).queryByRole("button", { name: "重试同一目标" })).not.toBeInTheDocument();
    expect(useAssistantStore.getState().requestedSessionOpen).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: "打开 Hermes 会话" }));
    expect(useAssistantStore.getState().requestedSessionOpen).toEqual({ projectName: "project-one", sessionId: "existing-hermes-session" });
  });

  it("reconciles a retry revision conflict and waits for an explicit retry with the fresh revision", async () => {
    const http = transport({ ...GOAL, state: "attention", revision: 12, session_id: "existing-hermes-session" });
    http.conflictRetry();
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    fireEvent.click(await within(dialog).findByRole("button", { name: "重试同一目标" }));
    expect(await within(dialog).findByText("目标版本已变化")).toBeInTheDocument();
    expect(http.writes).toHaveLength(1);
    const retry = within(dialog).getByRole("button", { name: "重试同一目标" });
    await waitFor(() => expect(retry).toBeEnabled());
    fireEvent.click(retry);
    expect(await within(dialog).findByText("启动中")).toBeInTheDocument();
    expect(http.writes).toHaveLength(2);
    expect(http.writes.map(({ options }) => JSON.parse(String(options.body)))).toEqual([{ expected_revision: 12 }, { expected_revision: 13 }]);
    expect(http.writes.every(({ url }) => url.endsWith("/goal-one/retry"))).toBe(true);
  });

  it.each(["pending", "running", "waiting", "stopping", "completed", "stopped"] as const)("does not offer same-goal retry for %s", async (state) => {
    const http = transport({ ...GOAL, state });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    await within(dialog).findByRole("region", { name: "当前制作目标" });
    expect(within(dialog).queryByRole("button", { name: "重试同一目标" })).not.toBeInTheDocument();
    expect(http.writes).toHaveLength(0);
    const originalAction = state === "completed" || state === "stopped" ? "开始自动制作" : "停止自动制作";
    expect(within(dialog).getByRole("button", { name: originalAction })).toBeInTheDocument();
  });

  it("does not create a second goal when another episode already owns the project", async () => {
    const http = transport({ ...GOAL, episode_id: 2, state: "waiting" });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    const dialog = await openEntry();
    expect(await within(dialog).findByText("本项目已有另一集的自动制作目标正在执行。")).toBeInTheDocument();
    expect(within(dialog).queryByLabelText("制作目标")).not.toBeInTheDocument();
    expect(http.writes).toHaveLength(0);
  });

  it("opens the accepted Hermes session only after unsaved edits are released", async () => {
    transport({ ...GOAL, state: "running", session_id: "hermes-session" });
    useAppStore.setState({ assistantPanelOpen: false });
    useAssistantStore.setState({ currentSessionId: "manual-session" });
    function DirtyEditor() {
      useLeaveGuard({ dirty: true, save: async () => true, discard: () => {} });
      return <p>未保存编辑</p>;
    }
    render(<LeaveGuardProvider><DirtyEditor /><AutonomousProduction projectName="project-one" episode={7} /></LeaveGuardProvider>);
    const dialog = await openEntry();
    fireEvent.click(await within(dialog).findByRole("button", { name: "打开 Hermes 会话" }));
    expect(await screen.findByRole("alertdialog")).toBeInTheDocument();
    expect(useAssistantStore.getState().requestedSessionOpen).toBeNull();
    expect(useAppStore.getState().assistantPanelOpen).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "放弃修改" }));
    await waitFor(() => expect(useAssistantStore.getState().requestedSessionOpen).toEqual({ projectName: "project-one", sessionId: "hermes-session" }));
    expect(useAppStore.getState().assistantPanelOpen).toBe(true);
    expect(useAssistantStore.getState().currentSessionId).toBe("manual-session");
  });

  it("polls active metadata quickly, terminal metadata slowly, and cancels on unmount", async () => {
    vi.useFakeTimers();
    const http = transport({ ...GOAL, state: "waiting" });
    const view = render(<AutonomousProduction projectName="project-one" episode={7} />);
    await act(async () => {});
    expect(http.fetch).toHaveBeenCalledTimes(1);
    http.setGoal({ ...GOAL, state: "completed" });
    await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
    expect(http.fetch).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(9999); });
    expect(http.fetch).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(http.fetch).toHaveBeenCalledTimes(3);
    view.unmount();
    expect(http.reads.every((signal) => signal.aborted)).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
    expect(http.fetch).toHaveBeenCalledTimes(3);
  });

  it("discovers a goal started externally after the page initially had no goal", async () => {
    vi.useFakeTimers();
    const http = transport(null);
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    await act(async () => {});
    expect(http.fetch).toHaveBeenCalledTimes(1);
    http.setGoal({ ...GOAL, state: "running", session_id: "externally-started-session" });
    await act(async () => { await vi.advanceTimersByTimeAsync(9999); });
    expect(http.fetch).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(screen.getByRole("button", { name: "自动制作 · 制作中" })).toBeInTheDocument();
    expect(http.fetch).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
    expect(http.fetch).toHaveBeenCalledTimes(3);
    expect(http.writes).toHaveLength(0);
  });

  it("discovers a background retry of an attention goal and resumes active metadata polling", async () => {
    vi.useFakeTimers();
    const http = transport({ ...GOAL, state: "attention", session_id: "same-hermes-session", revision: 12 });
    render(<AutonomousProduction projectName="project-one" episode={7} />);
    await act(async () => {});
    expect(http.fetch).toHaveBeenCalledTimes(1);
    http.setGoal({ ...GOAL, state: "pending", session_id: "same-hermes-session", revision: 13 });
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(screen.getByRole("button", { name: "自动制作 · 启动中" })).toBeInTheDocument();
    expect(http.fetch).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(2500); });
    expect(http.fetch).toHaveBeenCalledTimes(3);
    expect(http.writes).toHaveLength(0);
    expect(useAssistantStore.getState().requestedSessionOpen).toBeNull();
  });

  it("aborts an old project read and ignores a late response after project switch", async () => {
    let resolveOld!: (response: Response) => void;
    let oldSignal: AbortSignal | null | undefined;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, options: RequestInit) => {
      if (String(input).includes("project-one")) {
        oldSignal = options.signal;
        return new Promise<Response>((resolve) => { resolveOld = resolve; });
      }
      return Promise.resolve(response(null));
    }));
    const view = render(<AutonomousProduction projectName="project-one" episode={7} />);
    view.rerender(<AutonomousProduction projectName="project-two" episode={7} />);
    await act(async () => { resolveOld(response({ ...GOAL, state: "running" })); });
    const dialog = await openEntry();
    expect(oldSignal?.aborted).toBe(true);
    expect(within(dialog).queryByText(GOAL.objective)).not.toBeInTheDocument();
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "开始自动制作" })).toBeEnabled());
  });
});
