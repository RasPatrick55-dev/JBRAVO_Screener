import { useMemo } from "react";
import type { ExecuteSummaryResponse } from "./types";
import ScheduledAttemptNotice from "../dashboard/ScheduledAttemptNotice";
import { taskEvidenceFreshness, type ProcessReceipt, type ScheduledAttempt, type TaskEvidenceSource } from "../dashboard/processReceipt";
import {
  formatDateUtc,
  formatNumber,
  formatSignedCurrency,
  formatTimeUtc,
  parseNumber,
} from "./utils";

type Props = {
  summary: ExecuteSummaryResponse | null;
  isLoading: boolean;
  hasError: boolean;
  lastScheduledAttempt?: ScheduledAttempt | null;
  lastExecutionReceipt?: ProcessReceipt | null;
  taskSource?: TaskEvidenceSource;
};

const basePillClass =
  "rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] outline outline-1 outline-offset-[-1px]";

const summaryTileSurface = [
  "jbravo-panel-inner-cyan outline-cyan-400/35",
  "jbravo-panel-inner-emerald outline-emerald-400/35",
  "jbravo-panel-inner-violet outline-indigo-400/35",
  "jbravo-panel-inner-amber outline-amber-400/35",
  "jbravo-panel-inner-emerald outline-emerald-400/35",
  "jbravo-panel-inner-violet outline-rose-400/35",
  "jbravo-panel-inner-cyan outline-sky-400/35",
];

export const validatedExecutionPL = (summary: ExecuteSummaryResponse | null): number | null =>
  summary?.ok === true && summary.performance_status === "VALIDATED_PERFORMANCE" && summary.result_pl_source === "postgres_validated_performance"
    ? parseNumber(summary.result_pl_usd) : null;

export default function ExecutionSummaryCard({ summary, isLoading, hasError, lastScheduledAttempt, lastExecutionReceipt, taskSource }: Props) {
  const ready = !isLoading && !hasError && summary?.ok === true;
  const schedule = summary?.execution_schedule;
  const scheduleVerified = schedule?.status === "verified";
  const inWindow = scheduleVerified ? schedule.in_window : null;
  const windowLabel = scheduleVerified && schedule.entry_start_local && schedule.entry_end_local && schedule.timezone
    ? `${schedule.entry_start_local}-${schedule.entry_end_local} ${schedule.timezone}` : "Entry window unknown";
  const taskFreshness = taskEvidenceFreshness(taskSource);
  const runLabel = lastExecutionReceipt ? taskFreshness === "current" ? "Last execution receipt" : taskFreshness === "stale" ? "Cached execution receipt" : "Unverified execution receipt" : "Last reported run";
  const runTime = lastExecutionReceipt ? lastExecutionReceipt.ok === true ? lastExecutionReceipt.finished_utc : null : summary?.last_run_utc;

  const resultValue = ready ? validatedExecutionPL(summary) : null;
  const resultTone =
    resultValue !== null && resultValue > 0 ? "text-emerald-300" : resultValue !== null && resultValue < 0 ? "text-rose-300" : "text-slate-200";

  const tiles = useMemo(
    () => [
      {
        key: "last_run",
        label: runLabel,
        value: formatTimeUtc(runTime ?? null),
        subValue: formatDateUtc(runTime ?? null),
        tone: "text-slate-100",
      },
      {
        key: "in_window",
        label: "In window",
        value: inWindow === true ? "YES" : inWindow === false ? "NO" : "UNKNOWN",
        subValue: windowLabel,
        tone: inWindow === true ? "text-emerald-300" : "text-amber-300",
      },
      {
        key: "candidates",
        label: "Candidates",
        value: formatNumber(summary?.candidates),
        subValue: "Scanned",
        tone: "text-slate-100",
      },
      {
        key: "submitted",
        label: "Submitted",
        value: formatNumber(summary?.submitted),
        subValue: "Orders",
        tone: "text-emerald-300",
      },
      {
        key: "filled",
        label: "Filled",
        value: formatNumber(summary?.filled),
        subValue: "Completed",
        tone: "text-emerald-300",
      },
      {
        key: "rejected",
        label: "Rejected",
        value: formatNumber(summary?.rejected),
        subValue: "Blocked",
        tone: "text-rose-300",
      },
      {
        key: "result",
        label: "Result",
        value: formatSignedCurrency(resultValue),
        subValue: resultValue === null ? "Held / unknown; validated performance required" : "USD - validated eligible trades",
        tone: resultTone,
      },
    ],
    [
      resultTone,
      resultValue,
      summary?.candidates,
      summary?.filled,
      inWindow,
      windowLabel,
      runLabel,
      runTime,
      summary?.last_run_utc,
      summary?.rejected,
      summary?.result_pl_usd,
      summary?.submitted,
    ]
  );

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel jbravo-panel-cyan p-3 sm:p-5">
      <header className="flex flex-col items-start gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
        <h2 className="font-arimo text-[24px] font-semibold leading-none text-primary sm:text-[28px]">
          Execution Summary
        </h2>
        <div className="flex flex-wrap items-center gap-1 sm:justify-end">
          <span
            className={`${basePillClass} ${
              isLoading
                ? "bg-slate-500/20 text-slate-200 outline-slate-400/60"
                : "bg-slate-500/10 text-slate-400 outline-slate-500/45"
            }`}
          >
            Loading
          </span>
          <span
            className={`${basePillClass} ${
              !isLoading && !hasError && summary?.ok === true
                ? "bg-emerald-500/20 text-emerald-300 outline-emerald-400/55"
                : "bg-slate-500/10 text-slate-400 outline-slate-500/45"
            }`}
          >
            Success
          </span>
          <span
            className={`${basePillClass} ${
              hasError
                ? "bg-rose-500/20 text-rose-300 outline-rose-400/55"
                : "bg-slate-500/10 text-slate-400 outline-slate-500/45"
            }`}
          >
            Error
          </span>
        </div>
      </header>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-4 xl:grid-cols-7">
        {tiles.map((tile, index) => (
          <article
            key={tile.key}
            className={`rounded-xl px-3 py-3 jbravo-panel-inner outline outline-1 ${
              summaryTileSurface[index % summaryTileSurface.length]
            }`}
          >
            <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
              {tile.label}
            </p>
            <p className={`mt-2 font-cousine text-xl font-bold tabular-nums sm:text-2xl ${tile.tone}`}>
              {ready ? tile.value : "--"}
            </p>
            <p className="mt-1 font-cousine text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
              {ready ? tile.subValue : "--"}
            </p>
          </article>
        ))}
      </div>

      {ready ? <div className="mt-3 space-y-1 text-[11px] text-secondary">
        <p>Run metrics source: {summary?.source || "unknown"}.{lastExecutionReceipt ? ` ${taskFreshness === "current" ? "Actual" : "Last known"} receipt status: ${lastExecutionReceipt.status || "unknown"}.` : ""}</p>
        <p>Lifecycle owner: {scheduleVerified ? schedule.lifecycle_owner || "unknown" : "unknown"}.</p>
        <p>Cancellation policy: {scheduleVerified ? schedule.cancellation_policy?.replace(/_/g, " ") || "unknown" : "unknown"}. Deadline: {scheduleVerified && schedule.cancellation_deadline_status === "verified" ? schedule.cancellation_deadline || "unknown" : "unknown"}.</p>
        {scheduleVerified && schedule.legacy_cancel_after_min_is_deadline === false ? <p>Legacy cancel-after-min {schedule.legacy_cancel_after_min ?? "unknown"} is a compatibility flag, not the cancellation deadline.</p> : null}
        <p>Schedule source: {schedule?.source || "unknown"}; checked {schedule?.observed_at || "unknown"}. Session: {schedule?.session_date || "unknown"} ({schedule?.session_status || "unknown"}).</p>
        {!scheduleVerified ? <p>{schedule?.reason || "Current schedule is unverified."}</p> : null}
      </div> : null}
      <ScheduledAttemptNotice attempt={lastScheduledAttempt} taskSource={taskSource} />

      {hasError ? (
        <p className="mt-3 text-xs text-rose-300">Execution summary unavailable. Showing fallback values.</p>
      ) : null}
    </section>
  );
}

