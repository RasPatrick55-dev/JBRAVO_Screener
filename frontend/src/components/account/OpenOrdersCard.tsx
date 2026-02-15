import { formatCurrency, formatNumber } from "../dashboard/formatters";
import type { OpenOrderRow } from "./types";

interface OpenOrdersCardProps {
  rows: OpenOrderRow[];
  isLoading?: boolean;
}

const formatSubmitted = (value: string): string => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value || "--";
  }
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const sideChipClass = (side: string): string => {
  const normalized = side.toLowerCase();
  if (normalized === "buy") {
    return "jbravo-chip-success";
  }
  if (normalized === "sell") {
    return "jbravo-chip-error";
  }
  return "bg-slate-100 text-slate-700 outline outline-1 outline-slate-300 dark:bg-slate-700/40 dark:text-slate-200 dark:outline-slate-600";
};

export default function OpenOrdersCard({ rows, isLoading = false }: OpenOrdersCardProps) {
  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-cyan" aria-label="Open orders">
      <header>
        <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Open Orders</h2>
        <p className="mt-1 text-xs text-secondary">Active orders from Alpaca paper account</p>
      </header>

      <div className="mt-3 space-y-2 sm:hidden">
        {isLoading
          ? Array.from({ length: 4 }).map((_, index) => (
              <div key={`open-orders-mobile-skeleton-${index}`} className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-cyan">
                <div className="h-6 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
              </div>
            ))
          : null}

        {!isLoading && rows.length === 0 ? (
          <div className="rounded-xl px-3 py-6 text-center text-sm text-secondary jbravo-panel-inner jbravo-panel-inner-cyan">
            No open orders
          </div>
        ) : null}

        {!isLoading
          ? rows.map((row, index) => (
              <article
                key={`${row.symbol}-${row.submittedAt}-mobile-${index}`}
                className="rounded-xl px-3 py-2.5 jbravo-panel-inner jbravo-panel-inner-cyan"
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="font-arimo text-sm font-semibold text-primary">{row.symbol || "--"}</h3>
                    <p className="font-cousine text-xs uppercase tracking-[0.06em] text-secondary">{row.type || "--"}</p>
                  </div>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${sideChipClass(
                      row.side
                    )}`}
                  >
                    {row.side ? `${row.side.slice(0, 1).toUpperCase()}${row.side.slice(1)}` : "--"}
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  <div>
                    <p className="font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary">Qty</p>
                    <p className="font-cousine mt-0.5 text-xs font-bold tabular-nums text-primary">{formatNumber(row.qty)}</p>
                  </div>
                  <div>
                    <p className="font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary">Price</p>
                    <p className="font-cousine mt-0.5 text-xs font-bold tabular-nums text-primary">{formatCurrency(row.priceOrStop)}</p>
                  </div>
                  <div>
                    <p className="font-arimo text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary">Submitted</p>
                    <p className="font-cousine mt-0.5 text-xs font-bold tabular-nums text-primary">{formatSubmitted(row.submittedAt)}</p>
                  </div>
                </div>
              </article>
            ))
          : null}
      </div>

      <div className="mt-3 hidden max-h-72 overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan sm:block">
        <table className="min-w-max w-full table-auto">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="font-arimo border-b border-slate-200/70 text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/70">
              <th scope="col" className="px-2 py-2 text-left">
                Symbol
              </th>
              <th scope="col" className="px-2 py-2 text-left">
                Type
              </th>
              <th scope="col" className="px-2 py-2 text-center">
                Side
              </th>
              <th scope="col" className="px-2 py-2 text-right">
                Qty
              </th>
              <th scope="col" className="px-2 py-2 text-right">
                Price
              </th>
              <th scope="col" className="px-2 py-2 text-right">
                Submitted
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 5 }).map((_, index) => (
                  <tr key={`open-orders-skeleton-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
                    <td colSpan={6} className="px-3 py-3">
                      <div className="h-4 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
                    </td>
                  </tr>
                ))
              : null}

            {!isLoading && rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-7 text-center text-sm text-secondary">
                  No open orders
                </td>
              </tr>
            ) : null}

            {!isLoading
              ? rows.map((row, index) => (
                  <tr key={`${row.symbol}-${row.submittedAt}-${index}`} className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45">
                    <th scope="row" className="font-arimo px-2 py-2 text-left font-semibold text-primary whitespace-nowrap">
                      {row.symbol || "--"}
                    </th>
                    <td className="font-cousine px-2 py-2 text-left text-primary tabular-nums uppercase whitespace-nowrap">
                      {row.type || "--"}
                    </td>
                    <td className="px-2 py-2 text-center">
                      <span
                        className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${sideChipClass(
                          row.side
                        )}`}
                      >
                        {row.side ? `${row.side.slice(0, 1).toUpperCase()}${row.side.slice(1)}` : "--"}
                      </span>
                    </td>
                    <td className="font-cousine px-2 py-2 text-right tabular-nums text-primary">
                      {formatNumber(row.qty)}
                    </td>
                    <td className="font-cousine px-2 py-2 text-right tabular-nums text-primary">
                      {formatCurrency(row.priceOrStop)}
                    </td>
                    <td className="font-cousine px-2 py-2 text-right tabular-nums text-primary whitespace-nowrap">
                      {formatSubmitted(row.submittedAt)}
                    </td>
                  </tr>
                ))
              : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
