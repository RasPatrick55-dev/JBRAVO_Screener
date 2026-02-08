import { formatCurrency, formatNumber, formatSignedCurrency } from "../dashboard/formatters";
import StockLogo from "../ui/StockLogo";
import { positionTableGridColumns } from "./layout";

export interface PositionRowProps {
  symbol: string;
  logoUrl: string;
  shares: number;
  entryPrice: number;
  currentPrice: number;
  openPL: number;
  daysHeld: number;
  trailingStop: number;
  capturedPL: number;
}

const plToneClass = (value: number) => {
  if (value > 0) {
    return "text-accent-green";
  }
  if (value < 0) {
    return "text-accent-red";
  }
  return "text-financial";
};

const moneyCellClass = "px-3 text-center text-base leading-6 font-cousine tabular-nums text-financial";
const pillClass =
  "inline-flex min-w-20 items-center justify-center whitespace-nowrap rounded-full bg-surface px-2 py-0.5 text-base leading-6 font-cousine tabular-nums";
const numericCellClass = "px-3 text-center text-base leading-6 font-cousine tabular-nums text-financial";

function MobileField({ label, value, toneClass = "text-financial" }: { label: string; value: string; toneClass?: string }) {
  return (
    <div className="rounded-md border border-slate-200/70 bg-slate-50/40 px-2.5 py-2">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-secondary">{label}</div>
      <div className={`mt-1 font-cousine text-sm tabular-nums ${toneClass}`}>{value}</div>
    </div>
  );
}

export default function PositionRow({
  symbol,
  logoUrl,
  shares,
  entryPrice,
  currentPrice,
  openPL,
  daysHeld,
  trailingStop,
  capturedPL,
}: PositionRowProps) {
  const symbolText = symbol.trim().toUpperCase() || "--";

  return (
    <>
      <div className="border-b border-slate-200 px-4 py-4 sm:hidden" role="row">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <StockLogo symbol={symbolText} logoUrl={logoUrl} />
            <span className="truncate text-base font-semibold text-financial">{symbolText}</span>
          </div>
          <span className={`${pillClass} ${plToneClass(openPL)} min-w-0 text-sm leading-5`}>{formatSignedCurrency(openPL)}</span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <MobileField label="Shares" value={formatNumber(shares)} />
          <MobileField label="Days" value={formatNumber(daysHeld)} />
          <MobileField label="Entry Price" value={formatCurrency(entryPrice)} />
          <MobileField label="Current Price" value={formatCurrency(currentPrice)} />
          <MobileField label="Trailing Stop" value={formatCurrency(trailingStop)} />
          <MobileField label="Captured P/L" value={formatSignedCurrency(capturedPL)} toneClass={plToneClass(capturedPL)} />
        </div>
      </div>
      <div
        className={`${positionTableGridColumns} hidden items-center gap-0 divide-x divide-slate-200/50 border-b border-slate-200 px-4 py-3.5 sm:grid`}
        role="row"
      >
        <div className="flex items-center justify-center gap-2 px-3">
          <StockLogo symbol={symbolText} logoUrl={logoUrl} />
          <span className="truncate text-sm font-semibold text-financial">{symbolText}</span>
        </div>
        <div className={numericCellClass}>{formatNumber(shares)}</div>
        <div className={moneyCellClass}>{formatCurrency(entryPrice)}</div>
        <div className={moneyCellClass}>{formatCurrency(currentPrice)}</div>
        <div className="flex justify-center px-3">
          <span className={`${pillClass} ${plToneClass(openPL)}`}>{formatSignedCurrency(openPL)}</span>
        </div>
        <div className={numericCellClass}>{formatNumber(daysHeld)}</div>
        <div className={moneyCellClass}>{formatCurrency(trailingStop)}</div>
        <div className="flex justify-center px-3">
          <span className={`${pillClass} ${plToneClass(capturedPL)}`}>{formatSignedCurrency(capturedPL)}</span>
        </div>
      </div>
    </>
  );
}
