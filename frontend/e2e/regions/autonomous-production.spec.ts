import { clearAgentOverlay } from "../support/region-helpers.ts";
import { defineRegionScenarios } from "../support/scenarios.ts";
import { FIXED_NOW, recorded } from "../support/recorded.ts";
import { expect, type ApiOverrides } from "../support/test.ts";

const PATH = "/app/projects/demo/episodes/1";
const GOAL = {
  id: "a".repeat(32), project_name: "demo", episode_id: 1,
  objective: "按本集现有内容自动制作至带旁白字幕的最终成片。保留已有资产与声音，完整使用每段配音。".repeat(3),
  session_id: null, state: "waiting", revision: 3, dispatch_count: 1,
  last_error: null, created_at: FIXED_NOW, updated_at: FIXED_NOW,
};

function localApi(goal: typeof GOAL | null): ApiOverrides {
  const config = recorded<{ settings: Record<string, unknown> }>("system-config.json");
  const envelope = recorded<{ goal: null }>("project-demo-autonomous-production-latest.json");
  return {
    "GET /api/v1/system/config": { status: 200, body: {
      ...config, settings: { ...config.settings, execution_runtime: {
        mode: "local-hermes-l20", agent_available: true, agent_model: "local-hermes-model",
        image_model: "l20/qwen-image-2.1-int8", video_model: "l20/fasth3-v2-4step",
        audio_model: "local-tts/qwen3-tts-voice-design", cloud_fallback: false,
      } },
    } },
    "GET /api/v1/projects/demo/autonomous-production/latest": { status: 200, body: { ...envelope, goal } },
    "GET /api/v1/projects/demo/events/stream": { status: 404, body: { detail: "页面级套件不回放事件流" } },
  };
}

defineRegionScenarios("原生自动制作", [
  {
    name: "当前集只授权一次目标，长目标仍可编辑并能看到启动操作",
    path: PATH,
    api: localApi(null),
    ready: async (page) => { await page.getByRole("button", { name: "自动制作", exact: true }).waitFor(); },
    act: async (page) => {
      await clearAgentOverlay(page);
      await page.getByRole("button", { name: "自动制作", exact: true }).click();
      const dialog = page.getByRole("dialog", { name: "自动制作", exact: true });
      await expect(dialog.getByRole("textbox", { name: "制作目标", exact: true })).toHaveValue("按本集现有内容自动制作至带旁白字幕的最终成片");
      await dialog.getByRole("textbox", { name: "制作目标", exact: true }).fill(GOAL.objective);
      await expect(dialog.getByRole("button", { name: "开始自动制作" })).toBeEnabled();
      await expect(dialog.getByRole("button", { name: "开始自动制作" })).toBeInViewport();
      await expect(dialog.getByText(/启动即授权 Hermes/)).toContainText("免费本地配音");
    },
  },
  {
    name: "已有活跃目标时显示元数据与启动中的会话，不另建目标",
    path: PATH,
    api: localApi(GOAL),
    ready: async (page) => { await page.getByRole("button", { name: "自动制作 · 等待任务" }).waitFor(); },
    act: async (page) => {
      await clearAgentOverlay(page);
      await page.getByRole("button", { name: "自动制作 · 等待任务" }).click();
      const dialog = page.getByRole("dialog", { name: "自动制作", exact: true });
      await expect(dialog.getByText("Hermes 正在启动，会话就绪后可从这里打开。")).toBeVisible();
      await expect(dialog.getByRole("button", { name: "停止自动制作" })).toBeInViewport();
      await expect(dialog.getByRole("textbox", { name: "制作目标", exact: true })).toHaveCount(0);
      await expect(dialog.getByRole("button", { name: "打开 Hermes 会话" })).toHaveCount(0);
    },
  },
  {
    name: "停止处理中保持只读状态，防止再次投递停止或制作请求",
    path: PATH,
    api: localApi({ ...GOAL, state: "stopping" }),
    ready: async (page) => { await page.getByRole("button", { name: "自动制作 · 正在停止" }).waitFor(); },
    act: async (page) => {
      await clearAgentOverlay(page);
      await page.getByRole("button", { name: "自动制作 · 正在停止" }).click();
      const dialog = page.getByRole("dialog", { name: "自动制作", exact: true });
      await expect(dialog.getByRole("button", { name: "停止自动制作" })).toBeDisabled();
      await expect(dialog.getByRole("button", { name: "开始自动制作" })).toHaveCount(0);
      await expect(dialog.getByRole("button", { name: "关闭", exact: true })).toBeInViewport();
    },
  },
]);
