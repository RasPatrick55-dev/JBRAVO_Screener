import type { MlOverviewResponse, MlStage } from "./useMlOverview";

export function predictionFreshness(freshness: MlOverviewResponse["freshness"]): string {
  if (freshness?.status === "missing") return "Missing";
  if (freshness?.source === "fs" || freshness?.source_authority === "diagnostic_only") return "Unknown";
  if (freshness?.stale === true || freshness?.status === "stale") return "Stale";
  if (freshness?.status === "fresh" && freshness.stale === false
    && freshness.source_authority === "postgresql") return "Fresh";
  return "Unknown";
}

export function mlStagePresentation(stage: MlStage): MlStage {
  if (stage.execution_status === "verified") stage = { ...stage, execution_status: "unverified" };
  if (stage.artifact_present === true && (stage.source === "fs" || stage.source_authority === "diagnostic_only")) {
    return { ...stage, status: "Diagnostic only", tone: "neutral", execution_status: "unverified" };
  }
  // A failed operation or guard is still a failure when its artifacts are old
  // or execution evidence is incomplete. Freshness is disclosed separately.
  const status = stage.status.trim().toLowerCase();
  if (["error", "warning"].includes(stage.tone)
    || ["error", "failed", "failure", "blocked", "warn", "warning", "retrain", "recalibrate"].includes(status)) return stage;
  const positiveClaim = stage.tone === "success"
    || ["success", "available", "ready", "present", "complete", "completed"].includes(status);
  if (!positiveClaim) return stage;
  if (stage.freshness_status === "stale") return { ...stage, status: "Stale", tone: "warning" };
  // There is no provider-bound execution receipt in this endpoint contract.
  // Older API versions incorrectly set verified from timestamp proximity.
  return { ...stage, status: stage.artifact_present === true ? "Present / unverified" : stage.artifact_present === false ? "Missing" : "Unverified", tone: "neutral", execution_status: "unverified" };
}

export function successfulPredictionExecution(_data: MlOverviewResponse | null): string | null {
  // No DB/provider-bound execution receipt is consumed by this API yet. Legacy
  // PREDICTION_STAGE_ONLY claims were inferred from local logs, not receipts.
  return null;
}
