import { formatCurrency, formatSignedCurrency } from "../dashboard/formatters";
import type { AccountPerformanceRow, AccountTotal } from "./types";

interface AccountPerformanceCardProps {
  rows: AccountPerformanceRow[];
  total: AccountTotal;
  isLoading?: boolean;
}

const toneClass = (value: number): string => {
  if (value > 0) {
    return "jbravo-text-success";
  }
  if (value < 0) {
    return "jbravo-text-error";
  }
  return "text-primary";
};

const formatPerformancePercent = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const abs = Math.abs(value);
  const digits = abs < 1 ? 2 : 1;
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) {
    return "0%";
  }
  const sign = rounded > 0 ? "+" : "-";
  return `${sign}${Math.abs(rounded).toFixed(digits)}%`;
};

const formatAsOfUtc = (value: string | null | undefined): string => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
  }
  return `${new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
    timeZone: "UTC",
  }).format(parsed)} UTC`;
};

const performanceBasisLabel = (basis: AccountTotal["performanceBasis"]): string => {
  if (basis === "live_vs_close_baselines") {
    return "Daily to yearly net change (live equity vs close baselines)";
  }
  return "Daily to yearly net change (close-to-close)";
};

const equityBasisLabel = (basis: AccountTotal["equityBasis"]): string =>
  basis === "live" ? "Live equity" : "Last close equity";

export default function AccountPerformanceCard({
  rows,
  total,
  isLoading = false,
}: AccountPerformanceCardProps) {
  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-cyan" aria-label="Account performance">
      <header>
        <div>
          <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">
            Account Performance
          </h2>
          <p className="mt-1 text-xs text-secondary">{performanceBasisLabel(total.performanceBasis)}</p>
        </div>
      </header>

      <div className="mt-3 space-y-2 sm:hidden">
        {isLoading
          ? Array.from({ length: 4 }).map((_, index) => (
              <div key={`account-perf-mobile-skeleton-${index}`} className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-cyan">
                <div className="h-4 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
              </div>
            ))
          : null}

        {!isLoading && rows.length === 0 ? (
          <div className="rounded-xl px-3 py-6 text-center text-sm text-secondary jbravo-panel-inner jbravo-panel-inner-cyan">
            No performance data available.
          </div>
        ) : null}

        {!isLoading
          ? rows.map((row) => (
              <article
                key={`account-perf-mobile-${row.period}`}
                className="rounded-xl px-3 py-2.5 jbravo-panel-inner jbravo-panel-inner-cyan"
              >
                <div className="flex items-center justify-between gap-3">
                  <h3 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary">{row.period}</h3>
                  <span className={`font-cousine text-sm font-bold tabular-nums ${toneClass(row.netChangePct)}`}>
                    {formatPerformancePercent(row.netChangePct)}
                  </span>
                </div>
                <p className={`font-cousine mt-1 text-right text-sm font-bold tabular-nums ${toneClass(row.netChangeUsd)}`}>
                  {formatSignedCurrency(row.netChangeUsd)}
                </p>
              </article>
            ))
          : null}
      </div>

      <div className="mt-3 hidden max-h-56 overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-cyan sm:block">
        <table className="w-full table-fixed">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="font-arimo border-b border-slate-200/70 text-[10px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/70">
              <th scope="col" className="px-3 py-2 text-left">
                Period
              </th>
              <th scope="col" className="px-3 py-2 text-right">
                Net Change %
              </th>
              <th scope="col" className="px-3 py-2 text-right">
                Net Change $
              </th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 4 }).map((_, index) => (
                  <tr key={`account-perf-skeleton-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
                    <td colSpan={3} className="px-3 py-3">
                      <div className="h-4 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
                    </td>
                  </tr>
                ))
              : null}

            {!isLoading && rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-7 text-center text-sm text-secondary">
                  No performance data available.
                </td>
              </tr>
            ) : null}

            {!isLoading
              ? rows.map((row) => (
                  <tr key={row.period} className="border-b border-slate-200/70 text-sm transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45">
                    <th scope="row" className="font-arimo px-3 py-2 text-left text-xs font-semibold uppercase tracking-[0.08em] text-primary">
                      {row.period}
                    </th>
                    <td className={`font-cousine px-3 py-2 text-right tabular-nums ${toneClass(row.netChangePct)}`}>
                      {formatPerformancePercent(row.netChangePct)}
                    </td>
                    <td className={`font-cousine px-3 py-2 text-right tabular-nums ${toneClass(row.netChangeUsd)}`}>
                      {formatSignedCurrency(row.netChangeUsd)}
                    </td>
                  </tr>
                ))
              : null}
          </tbody>
          {!isLoading ? (
            <tfoot>
              <tr className="border-t border-slate-200/70 bg-slate-100/60 text-sm dark:border-slate-600/70 dark:bg-slate-800/65">
                <th
                  scope="row"
                  className="font-arimo px-3 py-2 text-left text-xs font-semibold uppercase tracking-[0.08em] text-primary"
                >
                  Account Total
                </th>
                <td className={`font-cousine px-3 py-2 text-right tabular-nums ${toneClass(total.netChangePct)}`}>
                  {formatPerformancePercent(total.netChangePct)}
                </td>
                <td className={`font-cousine px-3 py-2 text-right tabular-nums ${toneClass(total.netChangeUsd)}`}>
                  {formatSignedCurrency(total.netChangeUsd)}
                </td>
              </tr>
            </tfoot>
          ) : null}
        </table>
      </div>

      <div className="mt-3 rounded-xl px-4 py-3 jbravo-panel-inner jbravo-panel-inner-cyan">
        {isLoading ? (
          <div className="space-y-2">
            <div className="h-3 w-24 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
            <div className="h-7 w-44 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
            <div className="h-4 w-36 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
          </div>
        ) : (
          <div className="text-center">
            <div className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
              Account Total
            </div>
            <div className="font-cousine mt-1 text-3xl font-bold text-primary tabular-nums">
              {formatCurrency(total.equity)}
            </div>
            <div className="mt-1 flex items-center justify-center gap-3">
              <span className={`font-cousine text-sm font-bold tabular-nums ${toneClass(total.netChangePct)}`}>
                {formatPerformancePercent(total.netChangePct)}
              </span>
              <span className={`font-cousine text-sm font-bold tabular-nums ${toneClass(total.netChangeUsd)}`}>
                {formatSignedCurrency(total.netChangeUsd)}
              </span>
            </div>
            <div className="font-arimo mt-1 text-[11px] tracking-[0.04em] text-secondary">
              {equityBasisLabel(total.equityBasis)} as of {formatAsOfUtc(total.asOfUtc)}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
