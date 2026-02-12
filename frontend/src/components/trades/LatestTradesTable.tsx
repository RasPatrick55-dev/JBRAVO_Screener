import { formatCurrency, formatNumber, formatSignedCurrency } from "../dashboard/formatters";
import type { LatestTradeRow } from "./types";

interface LatestTradesTableProps {
  rows: LatestTradeRow[];
  isLoading?: boolean;
}

const formatDate = (value: string): string => {
  const trimmed = value.trim();
  if (!trimmed) {
    return "--";
  }
  const parsed = new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", {
    month: "numeric",
    day: "numeric",
    year: "numeric",
  });
};

export default function LatestTradesTable({ rows, isLoading = false }: LatestTradesTableProps) {
  return (
    <section className="rounded-2xl bg-surface p-md shadow-card outline-subtle" aria-label="Latest trades">
      <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">LATEST TRADES</h2>

      <div className="mt-3 max-h-[336px] overflow-auto">
        <table className="w-full table-fixed">
          <colgroup>
            <col className="w-[12%]" />
            <col className="w-[13%]" />
            <col className="w-[13%]" />
            <col className="w-[11%]" />
            <col className="w-[12%]" />
            <col className="w-[14%]" />
            <col className="w-[12%]" />
            <col className="w-[13%]" />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="font-arimo border-b border-slate-200 text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-700">
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Symbol
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Buy Date
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Sell Date
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Total Days
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Total Shares
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Avg Entry Price
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Price Sold
              </th>
              <th scope="col" className="px-1.5 py-1.5 text-center">
                Total P/L
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, index) => (
                  <tr key={`latest-skeleton-${index}`}>
                    <td colSpan={8} className="border-b border-slate-200 px-2 py-3 dark:border-slate-700">
                      <div className="h-4 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
                    </td>
                  </tr>
                ))
              : null}

            {!isLoading && rows.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-2 py-8 text-center text-sm text-secondary">
                  No recent closed trades available.
                </td>
              </tr>
            ) : null}

            {!isLoading
              ? rows.map((row, index) => {
                  const pnlTone =
                    row.totalPL > 0 ? "jbravo-text-success" : row.totalPL < 0 ? "jbravo-text-error" : "text-primary";
                  return (
                    <tr
                      key={`${row.symbol}-${row.sellDate}-${index}`}
                      className="border-b border-slate-200 text-xs dark:border-slate-700"
                    >
                      <td className="font-arimo truncate px-1.5 py-2 text-center font-semibold text-primary">{row.symbol}</td>
                      <td className="font-cousine truncate px-1.5 py-2 text-center text-secondary">{formatDate(row.buyDate)}</td>
                      <td className="font-cousine truncate px-1.5 py-2 text-center text-secondary">{formatDate(row.sellDate)}</td>
                      <td className="font-cousine px-1.5 py-2 text-center text-primary tabular-nums">
                        {formatNumber(row.totalDays)}
                      </td>
                      <td className="font-cousine px-1.5 py-2 text-center text-primary tabular-nums">
                        {formatNumber(row.totalShares)}
                      </td>
                      <td className="font-cousine px-1.5 py-2 text-center text-primary tabular-nums">
                        {formatCurrency(row.avgEntryPrice)}
                      </td>
                      <td className="font-cousine px-1.5 py-2 text-center text-primary tabular-nums">
                        {formatCurrency(row.priceSold)}
                      </td>
                      <td className={`font-cousine px-1.5 py-2 text-center font-bold tabular-nums ${pnlTone}`}>
                        {formatSignedCurrency(row.totalPL)}
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
