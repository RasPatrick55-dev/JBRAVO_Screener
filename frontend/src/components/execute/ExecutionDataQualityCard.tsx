import { usePollingRead, refreshNotice } from "../ui/usePollingRead";
import { useMemo } from "react";
import type { ExecuteAuditFinding, ExecuteAuditResponse } from "./types";
import {
  auditSeverityChipClass,
  formatAgeSeconds,
  formatDateTimeUtc,
  formatNumber,
} from "./utils";

const chipClass =
  "inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1";
const parseAudit = (value: unknown): ExecuteAuditResponse => {
  const payload = value as ExecuteAuditResponse | null;
  if (!payload || typeof payload !== "object" || !Boolean(payload.findings || payload.severity_counts || payload.cross_checks)) {
    throw new Error("Data quality audit unavailable.");
  }
  return payload;
};

export default function ExecutionDataQualityCard() {
  const source = usePollingRead("/api/execute/audit?limit=200", parseAudit);
  const { data: audit, isLoading } = source;
  const hasError = Boolean(source.error);

  const findings = useMemo(
    () => (audit?.findings ?? []).slice(0, 6),
    [audit?.findings]
  );

  const fetched = formatDateTimeUtc(audit?.fetched_at_utc ?? null);
  const highCount = audit?.severity_counts?.high ?? null;
  const warningCount = audit?.severity_counts?.warning ?? null;
  const overlapCount = audit?.cross_checks?.orders_subset?.open_closed_intersection_count ?? null;
  const ordersAge = audit?.cross_checks?.freshness?.orders_newest_age_seconds ?? null;
  const logsAge = audit?.cross_checks?.freshness?.execute_logs_newest_age_seconds ?? null;
  const isHealthy = !hasError && audit?.ok === true;

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel p-3 sm:p-5">
      <header className="flex flex-col items-start gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="font-arimo text-[26px] font-semibold leading-none text-primary sm:text-[30px]">
            Data Quality
          </h2>
          <p className="mt-1 font-cousine text-xs text-slate-600 dark:text-slate-400">
            Updated {isLoading ? "--:--:--" : fetched.time} {isLoading ? "---- -- --" : fetched.date} UTC
          </p>
        </div>
        <span
          className={
            `${chipClass} ` +
            (isHealthy
              ? "bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 outline-emerald-400/55"
              : "bg-rose-500/20 text-rose-700 dark:text-rose-300 outline-rose-400/55")
          }
        >
          {isLoading ? "Loading" : hasError ? source.receivedAt ? "Stale" : "Unavailable" : isHealthy ? "Healthy" : "Issues"}
        </span>
      </header>

      <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-violet outline outline-1 outline-indigo-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">High</p>
          <p className="mt-1 font-cousine text-xl font-bold text-rose-700 dark:text-rose-300 tabular-nums">{formatNumber(highCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-amber outline outline-1 outline-amber-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Warnings</p>
          <p className="mt-1 font-cousine text-xl font-bold text-amber-700 dark:text-amber-300 tabular-nums">{formatNumber(warningCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-cyan outline outline-1 outline-cyan-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Open/Closed Overlap</p>
          <p className="mt-1 font-cousine text-xl font-bold text-sky-700 dark:text-sky-300 tabular-nums">{formatNumber(overlapCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-emerald outline outline-1 outline-emerald-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Freshness</p>
          <p className="mt-1 font-cousine text-sm font-bold text-emerald-700 dark:text-emerald-300 tabular-nums">
            Orders {formatAgeSeconds(ordersAge)}
          </p>
          <p className="font-cousine text-sm font-bold text-emerald-700 dark:text-emerald-300 tabular-nums">
            Logs {formatAgeSeconds(logsAge)}
          </p>
        </article>
      </div>

      <div className="mt-3 overflow-x-auto">
        <div className="min-w-[620px] max-h-[240px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan">
          <table className="min-w-full table-auto">
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-700/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-600 dark:text-slate-300">
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Severity
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Code
                </th>
                <th scope="col" className="px-3 py-2 text-left">
                  Message
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={3} className="px-3 py-6 text-center text-sm text-secondary">
                    Loading data quality audit...
                  </td>
                </tr>
              ) : null}

              {!isLoading && findings.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-3 py-6 text-center text-sm text-secondary">
                    {hasError ? "Current audit findings are unknown." : "No findings."}
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? findings.map((item: ExecuteAuditFinding, index) => {
                    const severity = String(item.severity || "info").toLowerCase();
                    return (
                      <tr
                        key={`${item.code || "finding"}-${index}`}
                        className="border-b border-slate-700/70 text-sm transition-colors hover:bg-slate-200/60 dark:hover:bg-slate-800/45"
                      >
                        <td className="px-3 py-2 text-left whitespace-nowrap">
                          <span className={`${chipClass} ${auditSeverityChipClass(severity)}`}>{severity}</span>
                        </td>
                        <td className="font-cousine px-3 py-2 text-left tabular-nums text-slate-600 dark:text-slate-300 whitespace-nowrap">
                          {item.code || "--"}
                        </td>
                        <td className="px-3 py-2 text-left text-slate-900 dark:text-slate-100">{item.message || "--"}</td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>

      {hasError ? (
        <p className="mt-3 text-xs text-rose-700 dark:text-rose-300">{refreshNotice(source.error, source.receivedAt)}</p>
      ) : null}
    </section>
  );
}
