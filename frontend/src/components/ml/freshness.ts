import type { MlOverviewResponse, MlStage } from "./useMlOverview";

export function predictionFreshness(freshness: MlOverviewResponse["freshness"]): string {
  if (freshness?.stale === true || freshness?.status === "stale") return "Stale";
  if (freshness?.status === "missing") return "Missing";
  if (freshness?.status === "fresh" && freshness.stale === false) return "Fresh";
  return "Unknown";
}

export function mlStagePresentation(stage: MlStage): MlStage {
  // A failed operation or guard is still a failure when its artifacts are old
  // or execution evidence is incomplete. Freshness is disclosed separately.
  const status = stage.status.trim().toLowerCase();
  if (["error", "warning"].includes(stage.tone)
    || ["error", "failed", "failure", "blocked", "warn", "warning", "retrain", "recalibrate"].includes(status)) return stage;
  const positiveClaim = stage.tone === "success"
    || ["success", "available", "ready", "present", "complete", "completed"].includes(status);
  if (!positiveClaim) return stage;
  if (stage.freshness_status === "stale") return { ...stage, status: "Stale", tone: "warning" };
  if (stage.execution_status === "verified") return stage;
  return { ...stage, status: stage.artifact_present === true ? "Present / unverified" : stage.artifact_present === false ? "Missing" : "Unverified", tone: "neutral" };
}

export function successfulPredictionExecution(data: MlOverviewResponse | null): string | null {
  if (data?.ok !== true || data.execution_evidence_scope !== "PREDICTION_STAGE_ONLY") return null;
  const stamp = data.last_successful_execution;
  return stamp && Number.isFinite(Date.parse(stamp)) ? stamp : null;
}
