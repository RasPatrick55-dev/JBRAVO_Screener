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
    <div
      className={`${positionTableGridColumns} items-center gap-0 divide-x divide-techno-gold rounded-md border border-techno-gold techno-summary-surface techno-summary-shadow px-4 py-3.5 text-sm font-semibold`}
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
  );
}
