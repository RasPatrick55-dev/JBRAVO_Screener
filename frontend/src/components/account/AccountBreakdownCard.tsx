import { formatCurrency, formatPercent } from "../dashboard/formatters";
import type { AccountSummary } from "./types";

interface AccountBreakdownCardProps {
  summary: AccountSummary | null;
  isLoading?: boolean;
}

export default function AccountBreakdownCard({
  summary,
  isLoading = false,
}: AccountBreakdownCardProps) {
  const cash = summary?.cash ?? 0;
  const openPositionsValue = summary?.openPositionsValue ?? 0;
  const total = Math.max(0, cash + openPositionsValue);
  const cashPct = total > 0 ? (cash / total) * 100 : 0;
  const positionsPct = total > 0 ? (openPositionsValue / total) * 100 : 0;

  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-emerald" aria-label="Account breakdown">
      <header>
        <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Account Breakdown</h2>
        <p className="mt-1 text-xs text-secondary">Cash, gross open exposure, and allocation ratio</p>
      </header>

      {isLoading ? (
        <div className="mt-4 space-y-3">
          {Array.from({ length: 2 }).map((_, index) => (
            <div key={`account-breakdown-skeleton-${index}`} className="rounded-xl px-4 py-3 jbravo-panel-inner jbravo-panel-inner-emerald">
              <div className="h-3 w-28 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
              <div className="mt-2 h-6 w-36 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
            </div>
          ))}
          <div className="h-3 w-full animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          <div className="rounded-xl px-4 py-3 jbravo-panel-inner jbravo-panel-inner-emerald">
            <div className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">Total Cash</div>
            <div className="font-cousine mt-1 text-3xl font-bold tabular-nums text-primary">
              {formatCurrency(cash)}
            </div>
          </div>

          <div className="rounded-xl px-4 py-3 jbravo-panel-inner jbravo-panel-inner-emerald">
            <div className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
              Total Gross Open Exposure
            </div>
            <div className="font-cousine mt-1 text-3xl font-bold tabular-nums text-primary">
              {formatCurrency(openPositionsValue)}
            </div>
          </div>

          <div className="rounded-xl px-4 py-3 jbravo-panel-inner jbravo-panel-inner-emerald">
            <div className="flex items-center justify-between gap-3">
              <span className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
                Cash-to-positions ratio
              </span>
              <span className="font-cousine text-sm font-bold tabular-nums text-primary">
                {Math.round(cashPct)}/{Math.round(positionsPct)}
              </span>
            </div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-progress-rail">
              <div className="flex h-full w-full">
                <div className="h-full bg-emerald-500/80" style={{ width: `${Math.max(0, Math.min(100, cashPct))}%` }} />
                <div className="h-full bg-sky-500/80" style={{ width: `${Math.max(0, Math.min(100, positionsPct))}%` }} />
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between font-cousine text-xs tabular-nums text-secondary">
              <span>Cash {formatPercent(cashPct)}</span>
              <span>Positions {formatPercent(positionsPct)}</span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
