import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import type { ScreenerPickRow, ScreenerPicksResponse } from "./types";
import {
  compareNullableNumbers,
  fetchNoStoreJson,
  formatCompactNumber,
  formatCurrency,
  formatPercent,
  formatRunBadge,
  formatUtcDateTime,
  normalizeSymbol,
  parseNumber,
  withTs,
} from "./utils";

type PicksSortColumn = "rank" | "final_score" | "price" | "adv20" | "atrp";
type SortDirection = "asc" | "desc";

interface SortState {
  column: PicksSortColumn;
  direction: SortDirection;
}

const defaultSort: SortState = {
  column: "rank",
  direction: "asc",
};

const normalizePickRow = (row: Partial<ScreenerPickRow>): ScreenerPickRow => {
  return {
    rank: parseNumber(row.rank),
    symbol: normalizeSymbol(row.symbol),
    exchange: String(row.exchange ?? "--").trim() || "--",
    screened_at_utc: row.screened_at_utc ?? null,
    final_score: parseNumber(row.final_score),
    volume: parseNumber(row.volume),
    dollar_volume: parseNumber(row.dollar_volume),
    price: parseNumber(row.price),
    sma_ema_pct: parseNumber(row.sma_ema_pct),
    entry_price: parseNumber(row.entry_price),
    adv20: parseNumber(row.adv20),
    atrp: parseNumber(row.atrp),
  };
};

const sortDirectionLabel = (direction: SortDirection): "ascending" | "descending" =>
  direction === "asc" ? "ascending" : "descending";

const headerButtonClass =
  "inline-flex items-center gap-1 rounded px-1 py-0.5 transition-colors " +
  "hover:bg-slate-200/70 dark:hover:bg-slate-700/50";

const formatScore = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(3);
};
const REFRESH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 15_000;

type ScreenerPicksCardProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function ScreenerPicksCard({ onSyncStateChange }: ScreenerPicksCardProps) {
  const [rows, setRows] = useState<ScreenerPickRow[]>([]);
  const [runTsUtc, setRunTsUtc] = useState<string | null>(null);
  const [statusLabel, setStatusLabel] = useState("COMPLETE");
  const [sortState, setSortState] = useState<SortState>(defaultSort);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const hasLoadedRef = useRef(false);

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
        limit: "50",
        filter: "all",
      });

      try {
        const payload = await fetchNoStoreJson<ScreenerPicksResponse>(
          withTs(`/api/screener/picks?${params.toString()}`),
          REQUEST_TIMEOUT_MS
        );
        if (!isMounted) {
          return;
        }
        const normalizedRows = (payload.rows ?? [])
          .map((row) => normalizePickRow(row))
          .filter((row) => row.symbol !== "--");
        setRows(normalizedRows);
        setRunTsUtc(payload.run_ts_utc ?? null);
        setStatusLabel(String(payload.status ?? "COMPLETE").toUpperCase());
        setErrorMessage(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unable to load screener picks.";
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
  }, [reloadToken]);

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

  const displayedRows = useMemo(() => {
    const sorted = [...rows].sort((left, right) => {
      const multiplier = sortState.direction === "asc" ? 1 : -1;
      if (sortState.column === "rank") {
        const leftRank = left.rank ?? Number.POSITIVE_INFINITY;
        const rightRank = right.rank ?? Number.POSITIVE_INFINITY;
        return (leftRank - rightRank) * multiplier;
      }
      if (sortState.column === "final_score") {
        return compareNullableNumbers(left.final_score, right.final_score) * multiplier;
      }
      if (sortState.column === "price") {
        return compareNullableNumbers(left.price, right.price) * multiplier;
      }
      if (sortState.column === "adv20") {
        return compareNullableNumbers(left.adv20, right.adv20) * multiplier;
      }
      return compareNullableNumbers(left.atrp, right.atrp) * multiplier;
    });
    return sorted;
  }, [rows, sortState.column, sortState.direction]);

  const toggleSort = (column: PicksSortColumn) => {
    setSortState((previous) => {
      if (previous.column === column) {
        return {
          column,
          direction: previous.direction === "asc" ? "desc" : "asc",
        };
      }
      return {
        column,
        direction: column === "rank" ? "asc" : "desc",
      };
    });
  };

  const sortIcon = (column: PicksSortColumn): string => {
    if (sortState.column !== column) {
      return "<>";
    }
    return sortState.direction === "asc" ? "^" : "v";
  };

  const ariaSort = (column: PicksSortColumn): "none" | "ascending" | "descending" => {
    if (sortState.column !== column) {
      return "none";
    }
    return sortDirectionLabel(sortState.direction);
  };

  const statusChipClass =
    statusLabel === "COMPLETE"
      ? "jbravo-status-success"
      : statusLabel === "ERROR"
        ? "jbravo-status-error"
        : "bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-700/30 dark:text-slate-200 dark:ring-slate-500/40";

  return (
    <section className="rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-cyan sm:p-md" aria-label="Screener picks">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">
          Screener Picks
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          <span className="max-w-full truncate rounded-full px-2 py-1 font-cousine text-[10px] font-semibold uppercase tracking-[0.06em] text-secondary outline outline-1 outline-slate-300/80 dark:outline-slate-600/80 sm:px-2.5 sm:text-[11px]">
            {formatRunBadge(runTsUtc)}
          </span>
          <span
            className={`inline-flex items-center rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.06em] ring-1 ring-inset sm:px-2.5 sm:text-[11px] ${statusChipClass}`}
          >
            {statusLabel}
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
        <div className="max-h-[320px] min-w-[780px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan sm:max-h-[360px] sm:min-w-[860px]">
          <table className="w-full table-fixed">
            <caption className="sr-only">
              Screener picks with rank, score, price, average dollar volume, and ATR percent.
            </caption>
            <colgroup>
              <col className="w-[8%]" />
              <col className="w-[14%]" />
              <col className="w-[22%]" />
              <col className="w-[12%]" />
              <col className="w-[12%]" />
              <col className="w-[18%]" />
              <col className="w-[14%]" />
            </colgroup>
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
                <th scope="col" aria-sort={ariaSort("rank")} className="px-2 py-2 text-right">
                  <button type="button" onClick={() => toggleSort("rank")} className={headerButtonClass}>
                    Rank <span aria-hidden="true">{sortIcon("rank")}</span>
                  </button>
                </th>
                <th scope="col" className="px-2 py-2 text-left">
                  Symbol
                </th>
                <th scope="col" className="px-2 py-2 text-left">
                  Date Screened (UTC)
                </th>
                <th scope="col" aria-sort={ariaSort("final_score")} className="px-2 py-2 text-right">
                  <button
                    type="button"
                    onClick={() => toggleSort("final_score")}
                    className={headerButtonClass}
                  >
                    Final Score <span aria-hidden="true">{sortIcon("final_score")}</span>
                  </button>
                </th>
                <th scope="col" aria-sort={ariaSort("price")} className="px-2 py-2 text-right">
                  <button type="button" onClick={() => toggleSort("price")} className={headerButtonClass}>
                    Price <span aria-hidden="true">{sortIcon("price")}</span>
                  </button>
                </th>
                <th scope="col" aria-sort={ariaSort("adv20")} className="px-2 py-2 text-right">
                  <button type="button" onClick={() => toggleSort("adv20")} className={headerButtonClass}>
                    ADV20 <span aria-hidden="true">{sortIcon("adv20")}</span>
                  </button>
                </th>
                <th scope="col" aria-sort={ariaSort("atrp")} className="px-2 py-2 text-right">
                  <button
                    type="button"
                    onClick={() => toggleSort("atrp")}
                    className={headerButtonClass}
                  >
                    ATRP <span aria-hidden="true">{sortIcon("atrp")}</span>
                  </button>
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 8 }).map((_, index) => (
                    <tr key={`pick-skeleton-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
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
                      key={`${row.symbol}-${row.rank ?? index}`}
                      className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45 sm:text-sm"
                    >
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-secondary">
                        {row.rank ?? "--"}
                      </td>
                      <th scope="row" className="px-2 py-2.5 text-left font-semibold text-primary">
                        {row.symbol}
                        {row.exchange !== "--" ? (
                          <span className="ml-2 text-[11px] font-medium uppercase tracking-wide text-secondary">
                            {row.exchange}
                          </span>
                        ) : null}
                      </th>
                      <td className="font-cousine px-2 py-2.5 text-left text-secondary sm:whitespace-nowrap">
                        {formatUtcDateTime(row.screened_at_utc)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right font-semibold tabular-nums jbravo-text-success">
                        {formatScore(row.final_score)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatCurrency(row.price)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatCompactNumber(row.adv20)}
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                        {formatPercent(row.atrp, false)}
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
