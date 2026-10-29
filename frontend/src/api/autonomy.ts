import { requestJson } from "@/api/transport";
import type { AutonomousProductionGoal } from "@/types/autonomy";

export const AutonomyAPI = {
  latest(projectName: string, options: { signal?: AbortSignal } = {}) {
    return requestJson<{ goal: AutonomousProductionGoal | null }>(
      `/projects/${encodeURIComponent(projectName)}/autonomous-production/latest`,
      { signal: options.signal },
    );
  },
  create(projectName: string, episode: number, objective: string, clientKey: string) {
    return requestJson<{ goal: AutonomousProductionGoal }>(
      `/projects/${encodeURIComponent(projectName)}/autonomous-production`,
      { method: "POST", body: JSON.stringify({ episode_id: episode, objective, client_key: clientKey }) },
    );
  },
  retry(projectName: string, runId: string, expectedRevision: number) {
    return requestJson<{ goal: AutonomousProductionGoal }>(
      `/projects/${encodeURIComponent(projectName)}/autonomous-production/${encodeURIComponent(runId)}/retry`,
      { method: "POST", body: JSON.stringify({ expected_revision: expectedRevision }) },
    );
  },
  stop(projectName: string, runId: string) {
    return requestJson<{ goal: AutonomousProductionGoal }>(
      `/projects/${encodeURIComponent(projectName)}/autonomous-production/${encodeURIComponent(runId)}/stop`,
      { method: "POST" },
    );
  },
};
