import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import type { MetricsFilter, MetricsResponse, MetricsRow } from "./types";
import { fetchNoStoreJson, metricsMatchQuery, normalizeSymbol, withTs } from "./utils";

const filterOptions: Array<{ key: MetricsFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "gate_failures", label: "Gate Failures" },
  { key: "data_issues", label: "Data Issues" },
  { key: "high_confidence", label: "High Confidence" },
];

const normalizeGate = (value: unknown): "PASS" | "FAIL" => {
  const normalized = String(value ?? "")
    .trim()
    .toUpperCase();
  return normalized === "FAIL" ? "FAIL" : "PASS";
};

const normalizeBarsComplete = (value: unknown): "YES" | "NO" => {
  const normalized = String(value ?? "")
    .trim()
    .toUpperCase();
  return normalized === "NO" ? "NO" : "YES";
};

const normalizeConfidence = (value: unknown): "Low" | "Medium" | "High" => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "high") {
    return "High";
  }
  if (normalized === "medium") {
    return "Medium";
  }
  return "Low";
};

const normalizeMetricsRow = (row: Partial<MetricsRow>): MetricsRow => {
  return {
    symbol: normalizeSymbol(row.symbol),
    score_breakdown_short: String(row.score_breakdown_short ?? "").trim() || "--",
    liquidity_gate: normalizeGate(row.liquidity_gate),
    volatility_gate: normalizeGate(row.volatility_gate),
    trend_gate: normalizeGate(row.trend_gate),
    bars_complete: normalizeBarsComplete(row.bars_complete),
    confidence: normalizeConfidence(row.confidence),
    source_label: String(row.source_label ?? "").trim() || "--",
  };
};

const gateChipClass = (value: "PASS" | "FAIL"): string => {
  if (value === "PASS") {
    return "jbravo-chip-success";
  }
  return "jbravo-chip-error";
};

const confidenceChipClass = (value: "Low" | "Medium" | "High"): string => {
  if (value === "High") {
    return "bg-emerald-100 text-emerald-700 outline outline-1 outline-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-200 dark:outline-emerald-400/45";
  }
  if (value === "Medium") {
    return "bg-amber-100 text-amber-800 outline outline-1 outline-amber-300 dark:bg-amber-500/20 dark:text-amber-200 dark:outline-amber-300/45";
  }
  return "bg-slate-100 text-slate-600 outline outline-1 outline-slate-300 dark:bg-slate-700/35 dark:text-slate-200 dark:outline-slate-500/45";
};
const REFRESH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 15_000;

type MetricsResultsCardProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function MetricsResultsCard({ onSyncStateChange }: MetricsResultsCardProps) {
  const [rows, setRows] = useState<MetricsRow[]>([]);
  const [search, setSearch] = useState("");
  const [serverQuery, setServerQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState<MetricsFilter>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => setServerQuery(search.trim()), 250);
    return () => window.clearTimeout(timeoutId);
  }, [search]);

  useEffect(() => {
    let isMounted = true;
    let inFlight = false;

    const load = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      setIsLoading(!hasLoadedRef.current);
      const params = new URLSearchParams({
        filter: selectedFilter,
        limit: "50",
      });
      if (serverQuery) {
        params.set("q", serverQuery);
      }

      try {
        const payload = await fetchNoStoreJson<MetricsResponse>(
          withTs(`/api/screener/metrics?${params.toString()}`),
          REQUEST_TIMEOUT_MS
        );
        if (!isMounted) {
          return;
        }
        const normalizedRows = (payload.rows ?? [])
          .map((row) => normalizeMetricsRow(row))
          .filter((row) => row.symbol !== "--");
        setRows(normalizedRows);
        setErrorMessage(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unable to load metrics results.";
        setErrorMessage(message);
      } finally {
        if (isMounted) {
          setIsLoading(false);
          hasLoadedRef.current = true;
        }
        inFlight = false;
      }
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [selectedFilter, serverQuery, reloadToken]);

  useEffect(() => {
    if (!onSyncStateChange) {
      return;
    }
    if (isLoading) {
      onSyncStateChange("loading");
      return;
    }
    if (errorMessage) {
      onSyncStateChange("error");
      return;
    }
    onSyncStateChange("ready");
  }, [errorMessage, isLoading, onSyncStateChange]);

  const displayedRows = useMemo(() => rows.filter((row) => metricsMatchQuery(row, search)), [rows, search]);

  return (
    <section className="rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-emerald sm:p-md" aria-label="Metrics results">
      <header className="flex items-start justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">
          Metrics Results for Screener Picks
        </h2>
      </header>

      {errorMessage ? (
        <div
          role="alert"
          className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-rose-300/60 bg-rose-50/80 px-3 py-2 text-xs text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/10 dark:text-rose-200"
        >
          <span>{errorMessage}</span>
          <button
            type="button"
            onClick={() => setReloadToken((value) => value + 1)}
            className="rounded-md px-2 py-1 font-semibold uppercase tracking-wide outline outline-1 outline-rose-300/70 transition hover:bg-rose-100/70 dark:outline-rose-400/45 dark:hover:bg-rose-500/20"
          >
            Retry
          </button>
        </div>
      ) : null}

      <div className="mt-2 flex flex-wrap items-center gap-2 sm:mt-3">
        <label className="relative min-w-0 flex-1 sm:min-w-[220px]">
          <span className="sr-only">Search metrics rows</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search symbol, score breakdown, source..."
            className="w-full rounded-lg border border-slate-300/80 bg-white/80 px-3 py-2 pr-9 text-xs text-primary outline-none transition focus:border-sky-400 dark:border-slate-600/80 dark:bg-slate-900/55 sm:text-sm"
          />
          <svg
            viewBox="0 0 24 24"
            className="pointer-events-none absolute right-2.5 top-2.5 h-4 w-4 text-secondary"
            fill="none"
          >
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M20 20l-3.5-3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </label>

        <div className="flex flex-wrap items-center gap-1.5">
          {filterOptions.map((option) => {
            const isActive = option.key === selectedFilter;
            return (
              <button
                key={option.key}
                type="button"
                onClick={() => setSelectedFilter(option.key)}
                aria-pressed={isActive}
                className={
                  "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.07em] outline outline-1 outline-offset-[-1px] transition " +
                  (isActive
                    ? "bg-emerald-100 text-emerald-700 outline-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-200 dark:outline-emerald-400/45"
                    : "text-secondary outline-slate-300/80 hover:text-primary dark:outline-slate-600/80")
                }
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-2 overflow-x-auto sm:mt-3">
        <div className="max-h-[320px] min-w-[980px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-emerald sm:max-h-[360px] sm:min-w-[1180px]">
          <table className="w-full table-fixed">
            <caption className="sr-only">
              Screener metrics including gate checks, data quality, confidence, and source.
            </caption>
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
                <th scope="col" className="px-2 py-2 text-left">
                  Symbol
                </th>
                <th scope="col" className="px-2 py-2 text-left">
                  Score Breakdown
                </th>
                <th scope="col" className="px-2 py-2 text-center">
                  Liquidity
                </th>
                <th scope="col" className="px-2 py-2 text-center">
                  Volatility
                </th>
                <th scope="col" className="px-2 py-2 text-center">
                  Trend
                </th>
                <th scope="col" className="px-2 py-2 text-center">
                  Bars Complete
                </th>
                <th scope="col" className="px-2 py-2 text-center">
                  Confidence
                </th>
                <th scope="col" className="px-2 py-2 text-left">
                  Source
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 8 }).map((_, index) => (
                    <tr key={`metrics-skeleton-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
                      <td colSpan={8} className="px-2 py-2.5">
                        <div className="h-4 w-full animate-pulse rounded bg-slate-200/75 dark:bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && displayedRows.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-3 py-8 text-center text-sm text-secondary">
                    No results available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? displayedRows.map((row, index) => (
                    <tr
                      key={`${row.symbol}-${index}`}
                      className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45 sm:text-sm"
                    >
                      <th scope="row" className="px-2 py-2.5 text-left font-semibold text-primary">
                        {row.symbol}
                      </th>
                      <td className="px-2 py-2.5 text-left font-cousine text-xs text-secondary">
                        {row.score_breakdown_short}
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${gateChipClass(
                            row.liquidity_gate
                          )}`}
                        >
                          {row.liquidity_gate}
                        </span>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${gateChipClass(
                            row.volatility_gate
                          )}`}
                        >
                          {row.volatility_gate}
                        </span>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${gateChipClass(
                            row.trend_gate
                          )}`}
                        >
                          {row.trend_gate}
                        </span>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <span
                          className={
                            "inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 " +
                            (row.bars_complete === "YES" ? "jbravo-chip-success" : "jbravo-chip-error")
                          }
                        >
                          {row.bars_complete}
                        </span>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <span
                          className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${confidenceChipClass(
                            row.confidence
                          )}`}
                        >
                          {row.confidence}
                        </span>
                      </td>
                      <td className="px-2 py-2.5 text-left text-secondary">{row.source_label}</td>
                    </tr>
                  ))
                : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
