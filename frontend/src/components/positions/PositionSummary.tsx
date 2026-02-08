import { formatNumber, formatSignedCurrency } from "../dashboard/formatters";
import { positionTableGridColumns } from "./layout";

export interface PositionSummaryProps {
  totalShares: number;
  totalOpenPL: number;
  avgDaysHeld: number;
  totalCapturedPL: number;
}

const daysFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

const toneForValue = (value: number) => {
  if (value > 0) {
    return "text-accent-green";
  }
  if (value < 0) {
    return "text-accent-red";
  }
  return "text-slate-100";
};

const formatDays = (value: number) => {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return daysFormatter.format(value);
};

const summaryNumericClass = "px-3 text-center text-base leading-6 font-cousine tabular-nums";

export default function PositionSummary({
  totalShares,
  totalOpenPL,
  avgDaysHeld,
  totalCapturedPL,
}: PositionSummaryProps) {
  return (
    <>
      <div className="rounded-md border border-techno-gold techno-summary-surface techno-summary-shadow px-4 py-4 sm:hidden">
        <div className="text-sm font-semibold uppercase tracking-[0.08em] text-techno-gold">Total Summary</div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <div className="rounded-md border border-techno-gold/40 bg-slate-950/45 px-3 py-2 shadow-[inset_0_0_0_1px_rgba(245,158,11,0.08)]">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-techno-muted">Total Shares</div>
            <div className="mt-1 font-cousine text-base tabular-nums text-slate-100">{formatNumber(totalShares)}</div>
          </div>
          <div className="rounded-md border border-techno-gold/40 bg-slate-950/45 px-3 py-2 shadow-[inset_0_0_0_1px_rgba(245,158,11,0.08)]">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-techno-muted">Avg Days</div>
            <div className="mt-1 font-cousine text-base tabular-nums text-slate-100">{formatDays(avgDaysHeld)}</div>
          </div>
          <div className="rounded-md border border-techno-gold/40 bg-slate-950/45 px-3 py-2 shadow-[inset_0_0_0_1px_rgba(245,158,11,0.08)]">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-techno-muted">Open P/L</div>
            <div className={`mt-1 font-cousine text-base tabular-nums ${toneForValue(totalOpenPL)}`}>
              {formatSignedCurrency(totalOpenPL)}
            </div>
          </div>
          <div className="rounded-md border border-techno-gold/40 bg-slate-950/45 px-3 py-2 shadow-[inset_0_0_0_1px_rgba(245,158,11,0.08)]">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-techno-muted">Captured P/L</div>
            <div className={`mt-1 font-cousine text-base tabular-nums ${toneForValue(totalCapturedPL)}`}>
              {formatSignedCurrency(totalCapturedPL)}
            </div>
          </div>
        </div>
      </div>
      <div
        className={`${positionTableGridColumns} hidden items-center gap-0 divide-x divide-techno-gold rounded-md border border-techno-gold techno-summary-surface techno-summary-shadow px-4 py-3.5 text-sm font-semibold sm:grid`}
        role="row"
      >
        <div className="px-3 text-center uppercase tracking-[0.08em] text-techno-gold">Total Summary</div>
        <div className={`${summaryNumericClass} text-slate-100`}>{formatNumber(totalShares)}</div>
        <div className={`${summaryNumericClass} text-techno-muted`} aria-hidden="true">
          --
        </div>
        <div className={`${summaryNumericClass} text-techno-muted`} aria-hidden="true">
          --
        </div>
        <div className={`${summaryNumericClass} ${toneForValue(totalOpenPL)}`}>
          {formatSignedCurrency(totalOpenPL)}
        </div>
        <div className={`${summaryNumericClass} text-slate-100`}>{formatDays(avgDaysHeld)}</div>
        <div className={`${summaryNumericClass} text-techno-muted`} aria-hidden="true">
          --
        </div>
        <div className={`${summaryNumericClass} ${toneForValue(totalCapturedPL)}`}>
          {formatSignedCurrency(totalCapturedPL)}
        </div>
      </div>
    </>
  );
}
