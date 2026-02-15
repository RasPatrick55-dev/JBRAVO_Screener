import { formatCurrency, formatNumber, formatSignedCurrency } from "../dashboard/formatters";
import type { LatestTradeRow } from "./types";

interface LatestTradesTableProps {
  rows: LatestTradeRow[];
  isLoading?: boolean;
  onSymbolSelect?: (symbol: string) => void;
  getSymbolHref?: (symbol: string) => string | null | undefined;
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

const symbolBaseClass = "font-arimo block truncate font-semibold text-primary";
const symbolDesktopTextClass = `${symbolBaseClass} text-center`;
const symbolMobileTextClass = `${symbolBaseClass} text-left`;
const symbolActionFocusClass =
  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 " +
  "focus-visible:outline-slate-400 dark:focus-visible:outline-slate-500";

const plToneClass = (value: number): string => {
  if (value > 0) {
    return "jbravo-text-success";
  }
  if (value < 0) {
    return "jbravo-text-error";
  }
  return "text-primary";
};

const renderSymbol = (
  symbol: string,
  symbolHref: string | null | undefined,
  onSymbolSelect: ((symbol: string) => void) | undefined,
  className: string
) => {
  if (symbolHref) {
    return (
      <a href={symbolHref} className={`${className} ${symbolActionFocusClass}`}>
        {symbol}
      </a>
    );
  }
  if (onSymbolSelect) {
    return (
      <button type="button" className={`${className} ${symbolActionFocusClass}`} onClick={() => onSymbolSelect(symbol)}>
        {symbol}
      </button>
    );
  }
  return <span className={className}>{symbol}</span>;
};

function MobileField({ label, value, toneClass = "text-primary" }: { label: string; value: string; toneClass?: string }) {
  return (
    <div className="rounded-md px-2.5 py-2 jbravo-panel-inner jbravo-panel-inner-emerald">
      <p className="font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary">{label}</p>
      <p className={`font-cousine mt-1 text-sm font-bold tabular-nums ${toneClass}`}>{value}</p>
    </div>
  );
}

export default function LatestTradesTable({
  rows,
  isLoading = false,
  onSymbolSelect,
  getSymbolHref,
}: LatestTradesTableProps) {
  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-emerald" aria-label="Latest trades">
      <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">LATEST TRADES</h2>

      <div className="mt-3 sm:hidden">
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <div
                key={`latest-mobile-skeleton-${index}`}
                className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-emerald"
              >
                <div className="h-5 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
              </div>
            ))}
          </div>
        ) : null}

        {!isLoading && rows.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-secondary dark:border-slate-700">
            No recent closed trades available.
          </div>
        ) : null}

        {!isLoading && rows.length > 0 ? (
          <div className="overflow-hidden rounded-xl jbravo-panel-inner jbravo-panel-inner-emerald">
            {rows.map((row, index) => {
              const symbol = row.symbol.trim().toUpperCase() || "--";
              const symbolHref = getSymbolHref?.(symbol);
              const pnlTone = plToneClass(row.totalPL);
              return (
                <article
                  key={`latest-mobile-${row.symbol}-${row.sellDate}-${index}`}
                  className="border-b border-slate-200/70 px-3 py-3 last:border-b-0 dark:border-slate-700/70"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      {renderSymbol(symbol, symbolHref, onSymbolSelect, symbolMobileTextClass)}
                    </div>
                    <span className={`font-cousine text-sm font-bold tabular-nums ${pnlTone}`}>
                      {formatSignedCurrency(row.totalPL)}
                    </span>
                  </div>

                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <MobileField label="Buy Date" value={formatDate(row.buyDate)} toneClass="text-secondary" />
                    <MobileField label="Sell Date" value={formatDate(row.sellDate)} toneClass="text-secondary" />
                    <MobileField label="Total Days" value={formatNumber(row.totalDays)} />
                    <MobileField label="Total Shares" value={formatNumber(row.totalShares)} />
                    <MobileField label="Avg Entry Price" value={formatCurrency(row.avgEntryPrice)} />
                    <MobileField label="Price Sold" value={formatCurrency(row.priceSold)} />
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}
      </div>

      <div className="mt-3 hidden max-h-[336px] overflow-auto sm:block">
        <table className="w-full table-fixed">
          <caption className="sr-only">Latest closed trades with buy/sell dates, shares, and total P/L.</caption>
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
            <tr className="font-arimo border-b border-slate-200/70 text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-700/70">
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
                    <td colSpan={8} className="border-b border-slate-200/70 px-2 py-3 dark:border-slate-700/70">
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
                  const pnlTone = plToneClass(row.totalPL);
                  const symbol = row.symbol.trim().toUpperCase() || "--";
                  const symbolHref = getSymbolHref?.(symbol);

                  return (
                    <tr
                      key={`${row.symbol}-${row.sellDate}-${index}`}
                      className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45"
                    >
                      <th scope="row" className="truncate px-1.5 py-2 text-center">
                        {renderSymbol(symbol, symbolHref, onSymbolSelect, symbolDesktopTextClass)}
                      </th>
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
