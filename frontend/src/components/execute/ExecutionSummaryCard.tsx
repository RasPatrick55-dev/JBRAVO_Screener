import { useEffect, useMemo, useRef, useState } from "react";
import type { ExecuteSummaryResponse } from "./types";
import {
  fetchJsonNoStore,
  formatDateUtc,
  formatNumber,
  formatSignedCurrency,
  formatTimeUtc,
  parseSseJson,
  parseNumber,
} from "./utils";

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

export default function ExecutionSummaryCard() {
  const [summary, setSummary] = useState<ExecuteSummaryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const hasConnectedRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const applyPayload = (payload: ExecuteSummaryResponse | null) => {
      if (!isMounted) {
        return;
      }
      setSummary(payload);
      setHasError(!payload);
      setIsLoading(false);
      if (payload) {
        hasConnectedRef.current = true;
      }
    };

    const loadFallback = async () => {
      setIsLoading(true);
      const payload = await fetchJsonNoStore<ExecuteSummaryResponse>(`/api/execute/summary?ts=${Date.now()}`);
      applyPayload(payload);
    };

    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      void loadFallback();
      return () => {
        isMounted = false;
      };
    }

    setIsLoading(true);
    const source = new EventSource("/api/execute/summary/stream");
    source.onmessage = (event) => {
      const payload = parseSseJson<ExecuteSummaryResponse>(event.data);
      if (!payload) {
        return;
      }
      applyPayload(payload);
    };
    source.onerror = () => {
      if (!isMounted) {
        return;
      }
      if (!hasConnectedRef.current) {
        setHasError(true);
        setIsLoading(false);
      }
    };

    return () => {
      isMounted = false;
      source.close();
    };
  }, []);

  const resultValue = parseNumber(summary?.result_pl_usd) ?? 0;
  const resultTone =
    resultValue > 0 ? "text-emerald-300" : resultValue < 0 ? "text-rose-300" : "text-slate-200";

  const tiles = useMemo(
    () => [
      {
        key: "last_run",
        label: "Last run",
        value: formatTimeUtc(summary?.last_run_utc ?? null),
        subValue: formatDateUtc(summary?.last_run_utc ?? null),
        tone: "text-slate-100",
      },
      {
        key: "in_window",
        label: "In window",
        value: summary?.in_window ? "YES" : "NO",
        subValue: "ET 07:00-09:30",
        tone: summary?.in_window ? "text-emerald-300" : "text-amber-300",
      },
      { key: "candidates", label: "Candidates", value: formatNumber(summary?.candidates), subValue: "Scanned", tone: "text-slate-100" },
      { key: "submitted", label: "Submitted", value: formatNumber(summary?.submitted), subValue: "Orders", tone: "text-emerald-300" },
      { key: "filled", label: "Filled", value: formatNumber(summary?.filled), subValue: "Completed", tone: "text-emerald-300" },
      { key: "rejected", label: "Rejected", value: formatNumber(summary?.rejected), subValue: "Blocked", tone: "text-rose-300" },
      { key: "result", label: "Result", value: formatSignedCurrency(summary?.result_pl_usd), subValue: "USD", tone: resultTone },
    ],
    [
      resultTone,
      summary?.candidates,
      summary?.filled,
      summary?.in_window,
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
              !isLoading && !hasError
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
              {summary ? tile.value : "--"}
            </p>
            <p className="mt-1 font-cousine text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
              {summary ? tile.subValue : "--"}
            </p>
          </article>
        ))}
      </div>

      {hasError ? (
        <p className="mt-3 text-xs text-rose-300">Execution summary unavailable. Showing fallback values.</p>
      ) : null}
    </section>
  );
}
