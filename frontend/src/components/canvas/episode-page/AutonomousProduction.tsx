import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Bot, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { AutonomyAPI } from "@/api/autonomy";
import { useConfirmLeave } from "@/components/shared/edit-unit/LeaveGuard";
import { TruncatedText } from "@/components/shared/TruncatedText";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogBody, DialogClose, DialogContent, DialogDescription,
  DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useAppStore } from "@/stores/app-store";
import { useAssistantStore } from "@/stores/assistant-store";
import { useConfigStatusStore } from "@/stores/config-status-store";
import type { AutonomousProductionGoal } from "@/types/autonomy";
import { errMsg } from "@/utils/async";
import { uid } from "@/utils/id";

const ACTIVE = new Set(["pending", "running", "waiting", "stopping"]);
const POLL_MILLISECONDS = 2500;
const IDLE_POLL_MILLISECONDS = 10000;

function active(goal: AutonomousProductionGoal | null): goal is AutonomousProductionGoal {
  return goal !== null && ACTIVE.has(goal.state);
}

/** Local production has one durable goal per user/project, across episode views. */
export function AutonomousProduction({ projectName, episode }: { projectName: string; episode: number }) {
  const enabled = useConfigStatusStore((state) => state.executionRuntime?.mode === "local-hermes-l20");
  return enabled ? <ProductionEntry key={`${projectName}:${episode}`} projectName={projectName} episode={episode} /> : null;
}

function ProductionEntry({ projectName, episode }: { projectName: string; episode: number }) {
  const { t } = useTranslation(["workflow", "common"]);
  const [open, setOpen] = useState(false);
  const [goal, setGoal] = useState<AutonomousProductionGoal | null>(null);
  const [loading, setLoading] = useState(true);
  const [readError, setReadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [objective, setObjective] = useState(() => t("workflow:autonomy_default_goal"));
  const readAbort = useRef<AbortController | null>(null);
  const knownGoal = useRef<AutonomousProductionGoal | null>(null);
  const attempt = useRef<{ signature: string; clientKey: string } | null>(null);
  const inputId = useId();
  const confirmLeave = useConfirmLeave();

  useEffect(() => {
    const controller = new AbortController();
    readAbort.current = controller;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let lastGoal = knownGoal.current;
    async function read() {
      try {
        const response = await AutonomyAPI.latest(projectName, { signal: controller.signal });
        if (controller.signal.aborted) return;
        lastGoal = response.goal;
        knownGoal.current = response.goal;
        setGoal(response.goal);
        setReadError(null);
      } catch (error) {
        if (controller.signal.aborted) return;
        setReadError(errMsg(error));
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          timer = setTimeout(() => void read(), active(lastGoal) ? POLL_MILLISECONDS : IDLE_POLL_MILLISECONDS);
        }
      }
    }
    void read();
    return () => {
      controller.abort();
      clearTimeout(timer);
      if (readAbort.current === controller) readAbort.current = null;
    };
  }, [projectName, refresh]);

  const visibleGoal = goal && (goal.episode_id === episode || active(goal)) ? goal : null;
  const isActive = active(visibleGoal);
  const canRetry = visibleGoal?.state === "attention";

  const start = async () => {
    const text = objective.trim();
    if (!text || busy || isActive) return;
    const signature = JSON.stringify([projectName, episode, text]);
    if (attempt.current?.signature !== signature) attempt.current = { signature, clientKey: uid() };
    const clientKey = attempt.current.clientKey;
    readAbort.current?.abort();
    setBusy(true);
    setActionError(null);
    try {
      const response = await AutonomyAPI.create(projectName, episode, text, clientKey);
      knownGoal.current = response.goal;
      setGoal(response.goal);
      attempt.current = null;
    } catch (error) {
      setActionError(errMsg(error));
    } finally {
      setBusy(false);
      // Reconcile even a lost write response before offering a fresh goal.
      setRefresh((value) => value + 1);
    }
  };

  const stop = async () => {
    if (!visibleGoal || busy || visibleGoal.state === "stopping") return;
    readAbort.current?.abort();
    setBusy(true);
    setActionError(null);
    try {
      const response = await AutonomyAPI.stop(projectName, visibleGoal.id);
      knownGoal.current = response.goal;
      setGoal(response.goal);
    } catch (error) {
      setActionError(errMsg(error));
    } finally {
      setBusy(false);
      setRefresh((value) => value + 1);
    }
  };

  const retry = async () => {
    if (!visibleGoal || visibleGoal.state !== "attention" || busy) return;
    readAbort.current?.abort();
    setBusy(true);
    setActionError(null);
    try {
      const response = await AutonomyAPI.retry(projectName, visibleGoal.id, visibleGoal.revision);
      knownGoal.current = response.goal;
      setGoal(response.goal);
    } catch (error) {
      setActionError(errMsg(error));
    } finally {
      setBusy(false);
      setLoading(true);
      setRefresh((value) => value + 1);
    }
  };

  const openSession = useCallback(() => {
    if (!visibleGoal?.session_id) return;
    const sessionId = visibleGoal.session_id;
    confirmLeave(() => {
      useAppStore.getState().setAssistantPanelOpen(true);
      useAssistantStore.getState().requestSessionOpen(projectName, sessionId);
      setOpen(false);
    });
  }, [visibleGoal, confirmLeave, projectName]);

  return (
    <>
      <Button variant="outline" size="sm" className="min-w-0 max-w-48 shrink" onClick={() => setOpen(true)}>
        <Bot aria-hidden data-icon="inline-start" />
        <TruncatedText
          text={isActive ? t("workflow:autonomy_active_title", { state: t(`workflow:autonomy_state_${visibleGoal.state}`) }) : t("workflow:autonomy_title")}
          focusable={false}
        />
      </Button>
      <Dialog open={open} onOpenChange={(next) => { if (!busy) setOpen(next); }}>
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>{t("workflow:autonomy_title")}</DialogTitle>
            <DialogDescription>{t("workflow:autonomy_description")}</DialogDescription>
          </DialogHeader>
          <DialogBody>
            <div className="space-y-4">
            {loading && !visibleGoal && <p role="status">{t("workflow:autonomy_loading")}</p>}
            {readError && <div role="alert" className="space-y-2 wrap-break-word">
              <p>{t("workflow:autonomy_read_failed")}</p>
              <details><summary>{t("workflow:technical_detail")}</summary><p>{readError}</p></details>
              <Button variant="outline" size="sm" onClick={() => setRefresh((value) => value + 1)}>{t("common:retry")}</Button>
            </div>}
            {visibleGoal && <section className="space-y-3" aria-label={t("workflow:autonomy_current_goal")}>
              <Badge variant="secondary">{t(`workflow:autonomy_state_${visibleGoal.state}`)}</Badge>
              {visibleGoal.episode_id !== episode && <p className="text-sm text-muted-foreground">{t("workflow:autonomy_other_episode")}</p>}
              <p className="text-sm whitespace-pre-wrap wrap-break-word">{visibleGoal.objective}</p>
              {isActive && !visibleGoal.session_id && <p role="status" className="text-sm text-muted-foreground">{t("workflow:autonomy_session_starting")}</p>}
              {visibleGoal.last_error && <details className="text-sm wrap-break-word"><summary>{t("workflow:technical_detail")}</summary><p className="whitespace-pre-wrap">{visibleGoal.last_error}</p></details>}
              {visibleGoal.session_id && <Button variant="outline" size="sm" onClick={openSession}>{t("workflow:autonomy_open_session")}</Button>}
            </section>}
            {!isActive && !canRetry && <div className="space-y-2">
              <Label htmlFor={inputId}>{t("workflow:autonomy_objective")}</Label>
              <Textarea id={inputId} value={objective} onChange={(event) => setObjective(event.target.value)} disabled={busy} />
              <p className="text-xs text-muted-foreground">{t("workflow:autonomy_authorization")}</p>
            </div>}
            {actionError && <p role="alert" className="text-sm wrap-break-word text-destructive">{actionError}</p>}
            </div>
          </DialogBody>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" disabled={busy} />}>{t("common:close")}</DialogClose>
            {isActive ? <Button variant="outline" disabled={busy || visibleGoal.state === "stopping"} onClick={() => void stop()}>
              {busy && <Loader2 aria-hidden data-icon="inline-start" className="animate-spin" />}
              {t("workflow:autonomy_stop")}
            </Button> : canRetry ? <Button disabled={busy || loading || Boolean(readError)} onClick={() => void retry()}>
              {busy && <Loader2 aria-hidden data-icon="inline-start" className="animate-spin" />}
              {t("workflow:autonomy_retry")}
            </Button> : <Button disabled={busy || loading || Boolean(readError) || !objective.trim()} onClick={() => void start()}>
              {busy && <Loader2 aria-hidden data-icon="inline-start" className="animate-spin" />}
              {t("workflow:autonomy_start")}
            </Button>}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
