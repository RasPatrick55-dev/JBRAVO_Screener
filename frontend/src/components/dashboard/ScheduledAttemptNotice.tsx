import type { ScheduledAttempt } from "./processReceipt";

export default function ScheduledAttemptNotice({ attempt }: { attempt?: ScheduledAttempt | null }) {
  return <div className="mt-3 space-y-1 text-[11px] text-secondary">
    <p>Last scheduled attempt: {attempt?.ok === true ? attempt.status || "UNKNOWN" : "UNAVAILABLE"}</p>
    {attempt?.ok === true ? <>
      <p>{attempt.started_utc || "Unknown start"} to {attempt.finished_utc || "Unknown finish"} | Exit code: {attempt.rc ?? "unknown"}</p>
      <p>{attempt.reason_code ? `Reason: ${attempt.reason_code} | ` : ""}Submitted: {attempt.orders_submitted ?? "unknown"}</p>
      <p>Source: {attempt.source || "unknown"}. {attempt.release_attribution === "unverified_historical_attempt" ? "Historical attempt; not verified against the current release." : ""}</p>
    </> : null}
  </div>;
}
