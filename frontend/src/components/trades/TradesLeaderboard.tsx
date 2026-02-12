import { useMemo } from "react";
import { formatSignedCurrency } from "../dashboard/formatters";
import StockLogo from "../ui/StockLogo";
import TradesRangePills from "./TradesRangePills";
import type { LeaderRow, LeaderboardMode, RangeKey } from "./types";

interface TradesLeaderboardProps {
  rows: LeaderRow[];
  selectedRange: RangeKey;
  onRangeChange: (range: RangeKey) => void;
  mode: LeaderboardMode;
  onModeChange: (mode: LeaderboardMode) => void;
  isLoading?: boolean;
}

const MAX_LEADERBOARD_ROWS = 12;

export default function TradesLeaderboard({
  rows,
  selectedRange,
  onRangeChange,
  mode,
  onModeChange,
  isLoading = false,
}: TradesLeaderboardProps) {
  const sortedRows = useMemo(
    () => [...rows].sort((left, right) => left.rank - right.rank).slice(0, MAX_LEADERBOARD_ROWS),
    [rows]
  );

  const title = mode === "winners" ? "TOP WINNERS" : "TOP LOSERS";

  return (
    <section
      className="flex h-full min-h-0 flex-col overflow-hidden rounded-2xl bg-surface p-md shadow-card outline-subtle"
      aria-label="Leaderboard"
    >
      <h2 className="sr-only">Leaderboard</h2>

      <TradesRangePills value={selectedRange} onChange={onRangeChange} />

      <div className="mt-3 grid grid-cols-2 gap-2 rounded-lg bg-slate-100/70 p-1 dark:bg-slate-800/40">
        <button
          type="button"
          onClick={() => onModeChange("winners")}
          aria-pressed={mode === "winners"}
          className={
            "font-arimo rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] transition " +
            (mode === "winners"
              ? "bg-emerald-100 text-emerald-700 outline outline-1 outline-offset-[-1px] outline-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-300 dark:outline-emerald-300/45"
              : "text-secondary hover:text-primary")
          }
        >
          Winners
        </button>
        <button
          type="button"
          onClick={() => onModeChange("losers")}
          aria-pressed={mode === "losers"}
          className={
            "font-arimo rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] transition " +
            (mode === "losers"
              ? "bg-rose-100 text-rose-700 outline outline-1 outline-offset-[-1px] outline-rose-300 dark:bg-rose-500/20 dark:text-rose-300 dark:outline-rose-300/45"
              : "text-secondary hover:text-primary")
          }
        >
          Losers
        </button>
      </div>

      <h3 className="font-arimo mt-4 text-sm font-semibold uppercase tracking-[0.08em] text-primary">{title}</h3>

      <div className="mt-3 min-h-0 max-h-[312px] overflow-auto xl:flex-1 xl:max-h-none">
        <table className="w-full table-fixed">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="font-arimo border-b border-slate-200 text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-700">
              <th scope="col" className="w-[18%] px-2 py-2 text-center">
                Rank
              </th>
              <th scope="col" className="w-[46%] px-2 py-2 text-center">
                Symbol
              </th>
              <th scope="col" className="w-[36%] px-2 py-2 text-center">
                Total P/L
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, index) => (
                  <tr key={`leader-skeleton-${index}`}>
                    <td colSpan={3} className="border-b border-slate-200 px-2 py-3 dark:border-slate-700">
                      <div className="h-4 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
                    </td>
                  </tr>
                ))
              : null}

            {!isLoading && sortedRows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-2 py-6 text-center text-sm text-secondary">
                  No leaderboard trades yet.
                </td>
              </tr>
            ) : null}

            {!isLoading
              ? sortedRows.map((row) => {
                  const plTone = row.pl > 0 ? "jbravo-text-success" : row.pl < 0 ? "jbravo-text-error" : "text-primary";
                  return (
                    <tr key={`${row.rank}-${row.symbol}`} className="border-b border-slate-200 text-sm dark:border-slate-700">
                      <td className="font-cousine px-2 py-2.5 text-center text-secondary tabular-nums">#{row.rank}</td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center justify-center gap-2">
                          <div className="scale-90">
                            <StockLogo symbol={row.symbol} />
                          </div>
                          <span className="font-arimo font-semibold text-primary">{row.symbol}</span>
                        </div>
                      </td>
                      <td className={`font-cousine px-2 py-2.5 text-center font-bold tabular-nums ${plTone}`}>
                        {formatSignedCurrency(row.pl)}
                      </td>
                    </tr>
                  );
                })
              : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
