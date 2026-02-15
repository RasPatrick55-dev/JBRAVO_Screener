import { useEffect, useMemo, useState } from "react";
import type { BacktestResponse, BacktestRow, BacktestWindow } from "./types";
import {
  formatNumber,
  formatPercent,
  formatRunBadge,
  formatSignedCurrency,
  normalizeSymbol,
  parseNumber,
  withTs,
} from "./utils";

const defaultWindow: BacktestWindow = "6M";

const normalizeBacktestRow = (row: Partial<BacktestRow>, activeWindow: BacktestWindow): BacktestRow => {
  const normalizedWindow = String(row.window ?? activeWindow).toUpperCase();
  const mappedWindow = ["3M", "6M", "1Y", "ALL"].includes(normalizedWindow)
    ? (normalizedWindow as BacktestWindow)
    : activeWindow;
  return {
    symbol: normalizeSymbol(row.symbol),
    window: mappedWindow,
    trades: parseNumber(row.trades),
    win_rate_pct: parseNumber(row.win_rate_pct),
    avg_return_pct: parseNumber(row.avg_return_pct),
    pl_ratio: parseNumber(row.pl_ratio),
    max_dd_pct: parseNumber(row.max_dd_pct),
    avg_hold_days: parseNumber(row.avg_hold_days),
    total_pl_usd: parseNumber(row.total_pl_usd),
  };
};

const valueTone = (value: number | null): string => {
  if (value === null || Number.isNaN(value)) {
    return "text-primary";
  }
  if (value > 0) {
    return "jbravo-text-success";
  }
  if (value < 0) {
    return "jbravo-text-error";
  }
  return "text-primary";
};

export default function BacktestResultsCard() {
  const [rows, setRows] = useState<BacktestRow[]>([]);
  const [runTsUtc, setRunTsUtc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const load = async () => {
      setIsLoading(true);
      const params = new URLSearchParams({
        window: defaultWindow,
        limit: "50",
      });

      try {
        const response = await fetch(withTs(`/api/screener/backtest?${params.toString()}`), {
          cache: "no-store",
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Request failed (${response.status})`);
        }
        const payload = (await response.json()) as BacktestResponse;
        if (!isMounted) {
          return;
        }
        const normalizedRows = (payload.rows ?? [])
          .map((row) => normalizeBacktestRow(row, defaultWindow))
          .filter((row) => row.symbol !== "--");
        setRows(normalizedRows);
        setRunTsUtc(payload.run_ts_utc ?? null);
        setErrorMessage(null);
      } catch (error) {
        if (!isMounted || controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unable to load backtest results.";
        setErrorMessage(message);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    load();
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [reloadToken]);

  const displayedRows = useMemo(() => rows, [rows]);

  return (
    <section className="rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-violet sm:p-md" aria-label="Backtest results">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">
          Backtest Results for Screener Picks
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          <span className="max-w-full truncate rounded-full px-2 py-1 font-cousine text-[10px] font-semibold uppercase tracking-[0.06em] text-secondary outline outline-1 outline-slate-300/80 dark:outline-slate-600/80 sm:px-2.5 sm:text-[11px]">
            {formatRunBadge(runTsUtc)}
          </span>
          <span className="inline-flex items-center rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.06em] ring-1 ring-inset jbravo-status-success sm:px-2.5 sm:text-[11px]">
            COMPLETE
          </span>
        </div>
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

      <div className="mt-2 overflow-x-auto sm:mt-3">
        <div className="max-h-[320px] min-w-[860px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-violet sm:max-h-[360px] sm:min-w-[940px]">
          <table className="w-full table-fixed">
            <caption className="sr-only">
              Backtest results for screener picks.
            </caption>
            <colgroup>
              <col className="w-[18%]" />
              <col className="w-[11%]" />
              <col className="w-[13%]" />
              <col className="w-[13%]" />
              <col className="w-[12%]" />
              <col className="w-[13%]" />
              <col className="w-[20%]" />
            </colgroup>
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
                <th scope="col" className="px-2 py-2 text-left">
                  Symbol
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  Trades
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  Win Rate %
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  Avg Return
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  P/L Ratio
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  Max DD
                </th>
                <th scope="col" className="px-2 py-2 text-right">
                  Total P/L
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 8 }).map((_, index) => (
                    <tr key={`backtest-skeleton-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
                      <td colSpan={7} className="px-2 py-2.5">
                        <div className="h-4 w-full animate-pulse rounded bg-slate-200/75 dark:bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && displayedRows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-3 py-8 text-center text-sm text-secondary">
                    No results available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? displayedRows.map((row, index) => (
                    <tr
                      key={`${row.symbol}-${row.window}-${index}`}
                      className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45 sm:text-sm"
                    >
                      <th scope="row" className="px-2 py-2.5 text-left font-semibold text-primary">
                        {row.symbol}
                      </th>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatNumber(row.trades)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatPercent(row.win_rate_pct)}
                      </td>
                      <td
                        className={`font-cousine px-2 py-2.5 text-right tabular-nums ${valueTone(
                          row.avg_return_pct
                        )}`}
                      >
                        {formatPercent(row.avg_return_pct, true)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatNumber(row.pl_ratio)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums jbravo-text-error">
                        {formatPercent(row.max_dd_pct)}
                      </td>
                      <td
                        className={`font-cousine px-2 py-2.5 text-right font-semibold tabular-nums ${valueTone(
                          row.total_pl_usd
                        )}`}
                      >
                        {formatSignedCurrency(row.total_pl_usd)}
                      </td>
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
