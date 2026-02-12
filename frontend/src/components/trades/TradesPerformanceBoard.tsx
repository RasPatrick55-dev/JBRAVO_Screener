import { useMemo } from "react";
import { formatSignedCurrency } from "../dashboard/formatters";
import type { RangeKey, RangeRowMetrics } from "./types";

interface TradesPerformanceBoardProps {
  rows: RangeRowMetrics[];
  isLoading?: boolean;
}

const percentFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const rangeOrder: Record<RangeKey, number> = {
  d: 0,
  w: 1,
  m: 2,
  y: 3,
  all: 4,
};

const metricLabelClass = "font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary";

export default function TradesPerformanceBoard({
  rows,
  isLoading = false,
}: TradesPerformanceBoardProps) {
  const sortedRows = useMemo(
    () => [...rows].sort((left, right) => rangeOrder[left.key] - rangeOrder[right.key]),
    [rows]
  );

  return (
    <section className="rounded-2xl bg-surface p-md shadow-card outline-subtle" aria-label="Performance board">
      <h2 className="sr-only">Performance board</h2>

      <div className="overflow-auto">
        <table className="w-full table-fixed">
          <colgroup>
            <col className="w-[15%]" />
            <col className="w-[14%]" />
            <col className="w-[16%]" />
            <col className="w-[21%]" />
            <col className="w-[21%]" />
            <col className="w-[13%]" />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary">
              <th scope="col" className="px-2 py-1.5 text-center">
                Range
              </th>
              <th scope="col" className="px-2 py-1.5 text-center">
                Win Rate
              </th>
              <th scope="col" className="px-2 py-1.5 text-center">
                Total P/L
              </th>
              <th scope="col" className="px-2 py-1.5 text-center">
                Top Trade
              </th>
              <th scope="col" className="px-2 py-1.5 text-center">
                Worst Loss
              </th>
              <th scope="col" className="px-2 py-1.5 text-center">
                Trades
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, index) => (
                  <tr key={`perf-skeleton-${index}`}>
                    <td
                      colSpan={6}
                      className="rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-3 dark:border-slate-700 dark:bg-slate-800/40"
                    >
                      <div className="h-5 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
                    </td>
                  </tr>
                ))
              : null}

            {!isLoading && sortedRows.length === 0 ? (
              <tr>
                <td
                  colSpan={6}
                  className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-secondary dark:border-slate-700"
                >
                  No trade performance metrics yet.
                </td>
              </tr>
            ) : null}

            {!isLoading
              ? sortedRows.map((row) => {
                  const totalTone =
                    row.totalPL > 0 ? "jbravo-text-success" : row.totalPL < 0 ? "jbravo-text-error" : "text-primary";
                  const topTone =
                    row.topTrade.pl > 0
                      ? "jbravo-text-success"
                      : row.topTrade.pl < 0
                        ? "jbravo-text-error"
                        : "text-primary";
                  const worstTone =
                    row.worstLoss.pl > 0
                      ? "jbravo-text-success"
                      : row.worstLoss.pl < 0
                        ? "jbravo-text-error"
                        : "text-primary";
                  return (
                    <tr key={row.key} className="align-middle border-b border-slate-200 dark:border-slate-700">
                      <th
                        scope="row"
                        className="font-arimo px-2 py-2.5 text-center text-xs font-semibold uppercase tracking-[0.08em] text-secondary"
                      >
                        {row.label}
                      </th>
                      <td className="px-2 py-2.5 text-center">
                        <p className={metricLabelClass}>Win Rate</p>
                        <p className="font-cousine mt-1 text-sm font-bold text-primary tabular-nums">
                          {percentFormatter.format(row.winRatePct)}%
                        </p>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <p className={metricLabelClass}>Total P/L</p>
                        <p className={`font-cousine mt-1 text-sm font-bold tabular-nums ${totalTone}`}>
                          {formatSignedCurrency(row.totalPL)}
                        </p>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <p className={metricLabelClass}>Top Trade</p>
                        <p className="font-cousine mt-1 text-sm font-bold text-primary tabular-nums">
                          {row.topTrade.symbol}{" "}
                          <span className={topTone}>{formatSignedCurrency(row.topTrade.pl)}</span>
                        </p>
                      </td>
                      <td className="px-2 py-2.5 text-center">
                        <p className={metricLabelClass}>Worst Loss</p>
                        <p className="font-cousine mt-1 text-sm font-bold text-primary tabular-nums">
                          {row.worstLoss.symbol}{" "}
                          <span className={worstTone}>{formatSignedCurrency(row.worstLoss.pl)}</span>
                        </p>
                      </td>
                      <td className="font-cousine px-2 py-2.5 text-center text-sm font-bold text-primary tabular-nums">
                        <p className={metricLabelClass}>Trades</p>
                        <p className="mt-1">{numberFormatter.format(row.tradesCount)}</p>
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
