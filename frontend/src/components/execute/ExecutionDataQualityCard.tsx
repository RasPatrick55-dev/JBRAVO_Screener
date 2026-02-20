import { useEffect, useMemo, useRef, useState } from "react";
import type { ExecuteAuditFinding, ExecuteAuditResponse } from "./types";
import {
  auditSeverityChipClass,
  fetchJsonNoStore,
  formatAgeSeconds,
  formatDateTimeUtc,
  formatNumber,
  parseSseJson,
} from "./utils";

const chipClass =
  "inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1";
const REFRESH_INTERVAL_MS = 20_000;
const FIRST_LOAD_GRACE_MS = 2_500;
const REQUEST_TIMEOUT_MS = 15_000;

const hasUsableAudit = (payload: ExecuteAuditResponse | null): payload is ExecuteAuditResponse => {
  if (!payload || typeof payload !== "object") {
    return false;
  }
  return Boolean(payload.findings || payload.severity_counts || payload.cross_checks);
};

export default function ExecutionDataQualityCard() {
  const [audit, setAudit] = useState<ExecuteAuditResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const hasLoadedOnceRef = useRef(false);
  const fallbackInFlightRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const applyPayload = (payload: ExecuteAuditResponse | null) => {
      if (!isMounted) {
        return;
      }

      if (hasUsableAudit(payload)) {
        setAudit(payload);
        setHasError(false);
        hasLoadedOnceRef.current = true;
      } else {
        if (!hasLoadedOnceRef.current) {
          setAudit(null);
        }
        setHasError(true);
      }
      setIsLoading(false);
    };

    const loadFallback = async () => {
      if (fallbackInFlightRef.current) {
        return;
      }
      fallbackInFlightRef.current = true;
      setIsLoading(!hasLoadedOnceRef.current);
      const payload = await fetchJsonNoStore<ExecuteAuditResponse>(
        `/api/execute/audit?limit=200&ts=${Date.now()}`,
        REQUEST_TIMEOUT_MS
      );
      applyPayload(payload);
      fallbackInFlightRef.current = false;
    };

    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      void loadFallback();
      const intervalId = window.setInterval(() => {
        void loadFallback();
      }, REFRESH_INTERVAL_MS);
      return () => {
        isMounted = false;
        window.clearInterval(intervalId);
      };
    }

    setIsLoading(!hasLoadedOnceRef.current);
    const source = new EventSource("/api/execute/audit/stream?limit=200");
    const fallbackTimer = window.setTimeout(() => {
      if (!hasLoadedOnceRef.current) {
        void loadFallback();
      }
    }, FIRST_LOAD_GRACE_MS);
    const intervalId = window.setInterval(() => {
      void loadFallback();
    }, REFRESH_INTERVAL_MS);

    source.onmessage = (event) => {
      const payload = parseSseJson<ExecuteAuditResponse>(event.data);
      if (!payload) {
        return;
      }
      window.clearTimeout(fallbackTimer);
      applyPayload(payload);
    };
    source.onerror = () => {
      if (!isMounted) {
        return;
      }
      window.clearTimeout(fallbackTimer);
      void loadFallback();
    };

    return () => {
      isMounted = false;
      window.clearTimeout(fallbackTimer);
      window.clearInterval(intervalId);
      source.close();
    };
  }, []);

  const findings = useMemo(
    () => (audit?.findings ?? []).slice(0, 6),
    [audit?.findings]
  );

  const fetched = formatDateTimeUtc(audit?.fetched_at_utc ?? null);
  const highCount = Number(audit?.severity_counts?.high ?? 0);
  const warningCount = Number(audit?.severity_counts?.warning ?? 0);
  const overlapCount = Number(audit?.cross_checks?.orders_subset?.open_closed_intersection_count ?? 0);
  const ordersAge = audit?.cross_checks?.freshness?.orders_newest_age_seconds ?? null;
  const logsAge = audit?.cross_checks?.freshness?.execute_logs_newest_age_seconds ?? null;
  const isHealthy = Boolean(audit?.ok);

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel p-3 sm:p-5">
      <header className="flex flex-col items-start gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="font-arimo text-[26px] font-semibold leading-none text-primary sm:text-[30px]">
            Data Quality
          </h2>
          <p className="mt-1 font-cousine text-xs text-slate-400">
            Updated {isLoading ? "--:--:--" : fetched.time} {isLoading ? "---- -- --" : fetched.date} UTC
          </p>
        </div>
        <span
          className={
            `${chipClass} ` +
            (isHealthy
              ? "bg-emerald-500/20 text-emerald-300 outline-emerald-400/55"
              : "bg-rose-500/20 text-rose-300 outline-rose-400/55")
          }
        >
          {isHealthy ? "Healthy" : "Issues"}
        </span>
      </header>

      <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-violet outline outline-1 outline-indigo-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">High</p>
          <p className="mt-1 font-cousine text-xl font-bold text-rose-300 tabular-nums">{formatNumber(highCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-amber outline outline-1 outline-amber-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Warnings</p>
          <p className="mt-1 font-cousine text-xl font-bold text-amber-300 tabular-nums">{formatNumber(warningCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-cyan outline outline-1 outline-cyan-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Open/Closed Overlap</p>
          <p className="mt-1 font-cousine text-xl font-bold text-sky-300 tabular-nums">{formatNumber(overlapCount)}</p>
        </article>
        <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-emerald outline outline-1 outline-emerald-400/35">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Freshness</p>
          <p className="mt-1 font-cousine text-sm font-bold text-emerald-300 tabular-nums">
            Orders {formatAgeSeconds(ordersAge)}
          </p>
          <p className="font-cousine text-sm font-bold text-emerald-300 tabular-nums">
            Logs {formatAgeSeconds(logsAge)}
          </p>
        </article>
      </div>

      <div className="mt-3 overflow-x-auto">
        <div className="min-w-[620px] max-h-[240px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan">
          <table className="min-w-full table-auto">
            <thead className="sticky top-0 z-10 bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-700/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
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
                    No findings.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? findings.map((item: ExecuteAuditFinding, index) => {
                    const severity = String(item.severity || "info").toLowerCase();
                    return (
                      <tr
                        key={`${item.code || "finding"}-${index}`}
                        className="border-b border-slate-700/70 text-sm transition-colors hover:bg-slate-800/45"
                      >
                        <td className="px-3 py-2 text-left whitespace-nowrap">
                          <span className={`${chipClass} ${auditSeverityChipClass(severity)}`}>{severity}</span>
                        </td>
                        <td className="font-cousine px-3 py-2 text-left tabular-nums text-slate-300 whitespace-nowrap">
                          {item.code || "--"}
                        </td>
                        <td className="px-3 py-2 text-left text-slate-100">{item.message || "--"}</td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>

      {hasError ? (
        <p className="mt-3 text-xs text-rose-300">Data quality audit stream unavailable.</p>
      ) : null}
    </section>
  );
}
