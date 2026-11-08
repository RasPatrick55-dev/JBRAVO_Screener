import { taskEvidenceFreshness, type ScheduledAttempt, type TaskEvidenceSource } from "./processReceipt";

export default function ScheduledAttemptNotice({ attempt, taskSource }: { attempt?: ScheduledAttempt | null; taskSource?: TaskEvidenceSource }) {
  if (taskSource?.kind === "loading") return <p role="status" className="mt-3 text-[11px] text-secondary">Loading scheduled attempt and execution receipt. Their status is not yet known.</p>;
  const freshness = taskEvidenceFreshness(taskSource);
  const current = freshness === "current";
  const status = current && attempt?.ok === true ? attempt.status || "UNKNOWN"
    : freshness === "stale" && attempt?.ok === true ? "STALE EVIDENCE" : "UNAVAILABLE";
  return <div className="mt-3 space-y-1 text-[11px] text-secondary">
    <p>Last scheduled attempt: {status}</p>
    {!current ? <p>Task feed {freshness}; current attempt and receipt availability are unknown.{taskSource?.receivedAt ? ` Last received: ${taskSource.receivedAt}.` : ""}</p> : null}
    {attempt?.ok === true ? <>
      {!current ? <p>{freshness === "stale" ? "Cached" : "Unverified retained"} attempt status: {attempt.status || "UNKNOWN"}. This does not establish the current result.</p> : null}
      <p>{attempt.started_utc || "Unknown start"} to {attempt.finished_utc || "Unknown finish"} | Exit code: {attempt.rc ?? "unknown"}</p>
      <p>{attempt.reason_code ? `Reason: ${attempt.reason_code} | ` : ""}Submitted: {attempt.orders_submitted ?? "unknown"}</p>
      <p>Source: {attempt.source || "unknown"}. {attempt.release_attribution === "unverified_historical_attempt" ? "Historical attempt; not verified against the current release." : ""}</p>
    </> : null}
  </div>;
}
