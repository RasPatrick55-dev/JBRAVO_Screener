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

const plToneClass = (value: number): string => {
  if (value > 0) {
    return "jbravo-text-success";
  }
  if (value < 0) {
    return "jbravo-text-error";
  }
  return "text-primary";
};

function MobileMetric({ label, value, toneClass = "text-primary" }: { label: string; value: string; toneClass?: string }) {
  return (
    <div className="rounded-md px-2.5 py-2 jbravo-panel-inner jbravo-panel-inner-cyan">
      <p className={metricLabelClass}>{label}</p>
      <p className={`font-cousine mt-1 text-sm font-bold tabular-nums ${toneClass}`}>{value}</p>
    </div>
  );
}

export default function TradesPerformanceBoard({
  rows,
  isLoading = false,
}: TradesPerformanceBoardProps) {
  const sortedRows = useMemo(
    () => [...rows].sort((left, right) => rangeOrder[left.key] - rangeOrder[right.key]),
    [rows]
  );

  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-cyan" aria-label="Performance board">
      <h2 className="sr-only">Performance board</h2>

      <div className="sm:hidden">
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <div
                key={`perf-mobile-skeleton-${index}`}
                className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-cyan"
              >
                <div className="h-5 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
              </div>
            ))}
          </div>
        ) : null}

        {!isLoading && sortedRows.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-secondary dark:border-slate-700">
            No trade performance metrics yet.
          </div>
        ) : null}

        {!isLoading && sortedRows.length > 0 ? (
          <div className="overflow-hidden rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan">
            {sortedRows.map((row) => (
              <article key={`perf-mobile-${row.key}`} className="border-b border-slate-200/70 px-3 py-3 last:border-b-0 dark:border-slate-700/70">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-secondary">{row.label}</h3>
                  <span className="font-cousine text-xs font-semibold tabular-nums text-primary">
                    {numberFormatter.format(row.tradesCount)} TRADES
                  </span>
                </div>

                <div className="mt-2 grid grid-cols-2 gap-2">
                  <MobileMetric label="Win Rate" value={`${percentFormatter.format(row.winRatePct)}%`} />
                  <MobileMetric
                    label="Total P/L"
                    value={formatSignedCurrency(row.totalPL)}
                    toneClass={plToneClass(row.totalPL)}
                  />
                  <MobileMetric
                    label="Top Trade"
                    value={`${row.topTrade.symbol} ${formatSignedCurrency(row.topTrade.pl)}`}
                    toneClass={plToneClass(row.topTrade.pl)}
                  />
                  <MobileMetric
                    label="Worst Loss"
                    value={`${row.worstLoss.symbol} ${formatSignedCurrency(row.worstLoss.pl)}`}
                    toneClass={plToneClass(row.worstLoss.pl)}
                  />
                </div>
              </article>
            ))}
          </div>
        ) : null}
      </div>

      <div className="hidden overflow-auto sm:block">
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
                      className="rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan px-3 py-3"
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
                  const totalTone = plToneClass(row.totalPL);
                  const topTone = plToneClass(row.topTrade.pl);
                  const worstTone = plToneClass(row.worstLoss.pl);
                  return (
                    <tr key={row.key} className="align-middle border-b border-slate-200/70 dark:border-slate-700/70">
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
