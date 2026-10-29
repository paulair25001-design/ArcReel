export type AutonomousProductionState = "pending" | "running" | "waiting" | "stopping" | "completed" | "stopped" | "attention";

/** Public goal envelope; session events and project artifacts own production detail. */
export interface AutonomousProductionGoal {
  id: string;
  project_name: string;
  episode_id: number;
  objective: string;
  session_id: string | null;
  state: AutonomousProductionState;
  revision: number;
  dispatch_count: number;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}
