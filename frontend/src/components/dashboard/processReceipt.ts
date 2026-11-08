export type ProcessStatus = "COMPLETED" | "FAILED" | "STALE" | "UNAVAILABLE" | "PENDING";

export interface ScheduledAttempt {
  ok?: boolean;
  status?: string;
  source?: string;
  started_utc?: string | null;
  finished_utc?: string | null;
  rc?: number | null;
  reason_code?: string | null;
  orders_submitted?: number | null;
  release_attribution?: string | null;
}

export type ProcessReceipt = {
  last_scheduled_attempt?: ScheduledAttempt | null;
  last_execution_receipt?: ProcessReceipt | null;
  ok?: boolean;
  status?: string;
  source?: string | null;
  started_utc?: string | null;
  finished_utc?: string | null;
  duration_seconds?: number | null;
  rc?: number | null;
  stage_rcs?: Record<string, number | null>;
};

export function isCurrentReceipt(receipt: ProcessReceipt | null | undefined): boolean {
  return receipt?.source === "primary_receipt" || receipt?.source === "executor_receipt";
}

export function receiptStatus(receipt: ProcessReceipt | null | undefined, now = Date.now()): ProcessStatus | undefined {
  if (!isCurrentReceipt(receipt)) return undefined;
  if (!receipt?.ok) return "UNAVAILABLE";
  if (!/(?:Z|[+-]\d{2}:\d{2})$/.test(receipt.finished_utc ?? "")) return "UNAVAILABLE";
  const end = Date.parse(receipt.finished_utc ?? "");
  if (!Number.isFinite(end) || end > now) return "UNAVAILABLE";
  if (now - end > 86400000) return "STALE";
  if (receipt.status === "FAILED" || (receipt.rc !== null && receipt.rc !== undefined && receipt.rc !== 0)) return "FAILED";
  if (receipt.status === "PENDING") return "PENDING";
  return receipt.status === "COMPLETED" && receipt.rc === 0 ? "COMPLETED" : "UNAVAILABLE";
}

export function processTone(status: ProcessStatus) {
  return status === "COMPLETED" ? "success" as const : status === "FAILED" ? "error" as const : "neutral" as const;
}
