import CardShell from "./CardShell";
import StatusChip from "./StatusChip";
import { formatCurrency, formatNumber, formatPercent } from "./formatters";

export interface ExecuteTradesStatusCardProps {
  lastRun: { date: string; start: string; end: string; duration: string };
  ordersPlaced: number;
  totalValue: number;
  successRate: number;
  isCycleComplete: boolean;
  marketNote?: string | null;
}

const formatValue = (value: string | null | undefined) => {
  if (!value || !value.trim()) {
    return "--";
  }
  return value;
};

export default function ExecuteTradesStatusCard({
  lastRun,
  ordersPlaced,
  totalValue,
  successRate,
  isCycleComplete,
  marketNote,
}: ExecuteTradesStatusCardProps) {
  const statusIcons = {
    complete: (
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <polyline points="23 4 23 10 17 10" />
        <polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.13-3.36L23 10" />
        <path d="M20.49 15a9 9 0 0 1-14.13 3.36L1 14" />
      </svg>
    ),
    overdue: (
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5" />
        <path d="M12 16h.01" />
      </svg>
    ),
  } as const;

  const statusChip = isCycleComplete
    ? { label: "COMPLETED", tone: "success" as const, icon: statusIcons.complete }
    : { label: "OVERDUE", tone: "neutral" as const, icon: statusIcons.overdue };

  const valueStyles =
    "rounded-md border border-emerald-400/40 bg-slate-950/60 px-2 py-0.5 text-[13px] font-semibold tracking-[0.01em] text-emerald-200 tabular-nums";
  const successValueStyles =
    "rounded-md border border-emerald-400/40 bg-emerald-500/15 px-2 py-0.5 text-[13px] font-semibold tracking-[0.01em] text-emerald-100 tabular-nums";
  const lastRunStats = [
    { label: "Date", value: formatValue(lastRun?.date), valueClassName: valueStyles },
    { label: "Start Time", value: formatValue(lastRun?.start), valueClassName: valueStyles },
    { label: "End Time", value: formatValue(lastRun?.end), valueClassName: valueStyles },
    { label: "Duration", value: formatValue(lastRun?.duration), valueClassName: valueStyles },
  ];
  const ordersCount = formatNumber(ordersPlaced);
  const ordersTotal = formatCurrency(totalValue);
  const marketNoteIcon = (
    <svg
      className="h-3.5 w-3.5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M4.5 19.5 19.5 4.5" />
      <path d="M12 7v5" />
    </svg>
  );

  return (
    <CardShell className="border-emerald-400/30 bg-gradient-to-br from-slate-950/90 via-emerald-950/80 to-lime-950/70 p-4 shadow-[0_18px_40px_-24px_rgba(16,185,129,0.55)] sm:p-5">
      <div className="rounded-xl border border-emerald-400/40 bg-gradient-to-r from-emerald-950/90 via-emerald-900/90 to-teal-950/80 p-2.5 shadow-[0_0_24px_-12px_rgba(34,197,94,0.6)] sm:p-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-200">
              <svg
                className="h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M3 12h12" />
                <path d="m11 6 6 6-6 6" />
                <path d="M17 6h4v12h-4" />
              </svg>
            </div>
            <div className="flex flex-col">
              <h3 className="text-[15px] font-bold leading-[22px] text-white">
                Execute Trades Status
              </h3>
              <span className="text-[11px] text-emerald-100/80">
                Order routing & execution window
              </span>
            </div>
          </div>
          <StatusChip {...statusChip} />
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-sm" />
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-emerald-200/70">
            Last Task Run
          </span>
        </div>
        <div className="rounded-xl border border-emerald-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(34,197,94,0.45)] sm:p-3">
          <div className="flex flex-col gap-2">
            {lastRunStats.map((stat) => (
              <div key={stat.label} className="flex items-baseline gap-2">
                <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-emerald-100/70">
                  {stat.label}:
                </span>
                <span className={stat.valueClassName}>{stat.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-amber-400/40 bg-gradient-to-br from-emerald-500/20 via-lime-400/20 to-amber-400/30 p-2.5 shadow-[0_0_18px_-12px_rgba(251,191,36,0.5)] sm:p-3">
        <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-amber-200">
          Orders Filled (24h)
        </div>
        <div className="mt-2 grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-amber-300/50 bg-slate-950/60 p-2 sm:p-2.5">
            <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-amber-200/80">
              Count
            </div>
            <div className="mt-1 text-[16px] font-semibold text-amber-100">{ordersCount}</div>
          </div>
          <div className="rounded-lg border border-amber-300/50 bg-slate-950/60 p-2 sm:p-2.5">
            <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-amber-200/80">
              Total Value
            </div>
            <div className="mt-1 text-[16px] font-semibold text-amber-100">
              {ordersTotal}
            </div>
          </div>
        </div>
        {marketNote ? (
          <div className="mt-2">
            <StatusChip
              label={marketNote}
              tone="neutral"
              icon={marketNoteIcon}
              className="text-amber-200 outline-amber-400 bg-amber-400/20"
            />
          </div>
        ) : null}
      </div>

      <div className="mt-3 rounded-xl border border-emerald-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(34,197,94,0.4)] sm:p-3">
        <div className="flex items-center justify-between gap-4">
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-emerald-100/70">
            Success Rate
          </span>
          <span className={successValueStyles}>{formatPercent(successRate)}</span>
        </div>
      </div>
    </CardShell>
  );
}
