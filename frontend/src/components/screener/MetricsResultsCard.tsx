import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import type { MetricsFilter, MetricsResponse, MetricsRow } from "./types";
import { fetchNoStoreJson, formatCurrency, formatNumber, metricsMatchQuery, withTs } from "./utils";
import { formatMetricWinRate, metricRows, strategyLabel } from "./metricsRows";

const filterOptions: Array<{ key: MetricsFilter; label: string }> = [
  { key: "all", label: "All picks" },
  { key: "with_history", label: "Completed history" },
  { key: "no_history", label: "No completed history" },
];
const REFRESH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 15_000;

type MetricsResultsCardProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function MetricsResultsCard({ onSyncStateChange }: MetricsResultsCardProps) {
  const [strategy, setStrategy] = useState("Strategy backtest");
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
        const normalizedRows = metricRows(payload);
        setStrategy(strategyLabel(payload.strategy_id!, payload.strategy_version!));
        setRows(normalizedRows);
        setErrorMessage(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unable to load metrics results.";
        setRows([]);
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

      <p className="mt-2 text-xs text-secondary">{strategy} · Completed signals · USD</p>
      <p className="mt-1 text-xs text-secondary">Costs included. Drawdown is in dollars. Current signals have no future result yet.</p>
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
            placeholder="Search symbol or strategy..."
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
          <table className="w-full min-w-[760px] border-collapse">
          <caption className="sr-only">Historical performance from each pick's exact strategy backtest.</caption>
          <thead className="sticky top-0 bg-slate-100 dark:bg-slate-900">
            <tr className="text-xs text-secondary">
              {["Symbol", "Closed trades", "Win rate", "Net P&L", "Expectancy", "Profit factor", "Drawdown (USD)", "History"].map(name =>
                <th key={name} scope="col" className="px-2 py-2 text-left">{name}</th>)}
            </tr>
          </thead>
          <tbody>
            {isLoading ? <tr><td colSpan={8} className="px-3 py-6 text-center text-sm text-secondary">Loading strategy metrics...</td></tr> : null}
            {!isLoading && !errorMessage && !displayedRows.length ? <tr><td colSpan={8} className="px-3 py-6 text-center text-sm text-secondary">
              {search || selectedFilter !== "all" ? "No matching metrics." : "No Screener picks in this batch."}
            </td></tr> : null}
            {!isLoading && displayedRows.map(row => <tr key={row.signal_id} className="border-t border-slate-200/70 text-xs text-primary dark:border-slate-700/70 sm:text-sm">
              <th scope="row" className="px-2 py-2.5 text-left" title={strategyLabel(row.strategy_id, row.strategy_version)}>{row.symbol}</th>
              <td className="px-2 py-2.5">{formatNumber(row.trades)}</td>
              <td className="px-2 py-2.5">{formatMetricWinRate(row.win_rate_pct)}</td>
              <td className="px-2 py-2.5">{formatCurrency(row.net_pnl)}</td>
              <td className="px-2 py-2.5">{formatCurrency(row.expectancy)}</td>
              <td className="px-2 py-2.5">{row.profit_factor === null ? "--" : row.profit_factor.toFixed(2)}</td>
              <td className="px-2 py-2.5">{formatCurrency(row.max_drawdown_cash)}</td>
              <td className="px-2 py-2.5 text-secondary">{row.sample_status === "NO_COMPLETED_SIGNALS" ? "No completed signals" : "Completed signals"}</td>
            </tr>)}
          </tbody>
        </table>
        </div>
      </div>
    </section>
  );
}
