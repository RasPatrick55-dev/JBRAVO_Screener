import CardShell from "./CardShell";
import Sparkline from "./Sparkline";
import StatusChip from "./StatusChip";
import StockLogo from "../ui/StockLogo";
import { formatCurrency, formatSignedCurrency, formatSignedPercent } from "./formatters";

export interface Position {
  symbol: string;
  logoUrl?: string;
  qty?: number;
  entryPrice?: number;
  currentPrice: number;
  sparklineData: number[];
  percentPL: number;
  dollarPL: number;
  costBasis?: number;
}

export interface MonitoringPositionsCardProps {
  positions: Position[];
}

const plToneClass = (value: number | null | undefined) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "text-slate-300";
  }
  return value >= 0 ? "text-emerald-300" : "text-rose-300";
};

const SummaryTile = ({ label, value, valueTone }: { label: string; value: string; valueTone: string }) => {
  return (
    <div className="rounded-xl border border-amber-300/40 bg-slate-950/60 p-2 shadow-sm">
      <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-amber-200/70">
        {label}
      </div>
      <div className={`mt-1 text-[13px] leading-[18px] ${valueTone}`}>{value}</div>
    </div>
  );
};

const symbolFromLogoUrl = (logoUrl: string | undefined) => {
  if (!logoUrl) {
    return null;
  }
  const tickerMatch = logoUrl.match(/\/ticker\/([^/?#]+)/i);
  if (tickerMatch?.[1]) {
    return tickerMatch[1].toUpperCase();
  }
  const apiMatch = logoUrl.match(/\/api\/logos\/([^/.?#]+)/i);
  if (apiMatch?.[1]) {
    return apiMatch[1].toUpperCase();
  }
  return null;
};

export default function MonitoringPositionsCard({ positions }: MonitoringPositionsCardProps) {
  const hasPositions = positions && positions.length > 0;
  const dollarValues = hasPositions
    ? positions.map((position) => position.dollarPL).filter((value) => Number.isFinite(value))
    : [];
  const totalDollar =
    dollarValues.length > 0 ? dollarValues.reduce((sum, value) => sum + value, 0) : null;
  const percentValues = hasPositions
    ? positions.map((position) => position.percentPL).filter((value) => Number.isFinite(value))
    : [];
  const avgPercent =
    percentValues.length > 0
      ? percentValues.reduce((sum, value) => sum + value, 0) / percentValues.length
      : null;
  const costValues = hasPositions
    ? positions
        .map((position) => position.costBasis)
        .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : [];
  const totalCost =
    costValues.length > 0 ? costValues.reduce((sum, value) => sum + value, 0) : null;
  const portfolioPercent =
    totalDollar !== null && totalCost && totalCost !== 0
      ? (totalDollar / totalCost) * 100
      : avgPercent;
  const plSparklineFor = (position: Position) => {
    if (!position.sparklineData || position.sparklineData.length === 0) {
      return [];
    }
    const entryPrice = position.entryPrice ?? Number.NaN;
    const qty = position.qty ?? Number.NaN;
    if (Number.isFinite(entryPrice) && Number.isFinite(qty)) {
      return position.sparklineData.map((price) => (price - entryPrice) * qty);
    }
    return position.sparklineData;
  };

  return (
    <CardShell className="border-emerald-400/30 bg-gradient-to-br from-slate-950/95 via-emerald-950/80 to-amber-950/70 p-4 shadow-[0_18px_40px_-24px_rgba(16,185,129,0.5)] sm:p-5">
      <div className="rounded-xl border border-emerald-400/40 bg-gradient-to-r from-slate-950/90 via-emerald-900/90 to-amber-950/80 p-2 shadow-[0_0_24px_-12px_rgba(16,185,129,0.5)] sm:p-2.5">
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
                <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <div className="flex flex-col">
              <h3 className="text-[15px] font-bold leading-[22px] text-white">
                Monitoring Positions
              </h3>
              <span className="text-[11px] text-emerald-100/80">Live portfolio exposure</span>
            </div>
          </div>
          <StatusChip label={`${positions.length} OPEN`} tone="active" />
        </div>
      </div>

      <div className="mt-3 flex flex-col gap-3">
        <div className="hidden grid-cols-[minmax(0,1fr)_auto_auto_auto_auto] items-center text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-200/70 sm:grid">
          <span>Position</span>
          <span className="text-right">Price</span>
          <span className="text-right">P/L Trend</span>
          <span className="text-right">% P/L</span>
          <span className="text-right">Total $ P/L</span>
        </div>

        <div className="flex flex-col gap-2.5">
          {hasPositions ? (
            positions.map((position, index) => {
              const trimmedSymbol = position.symbol?.trim() ?? "";
              const displaySymbol =
                (trimmedSymbol ? trimmedSymbol.toUpperCase() : "") ||
                symbolFromLogoUrl(position.logoUrl) ||
                "--";
              const toneClass = plToneClass(position.percentPL);
              const plSparkline = plSparklineFor(position);
              const plTone =
                plSparkline.length > 0 && plSparkline[plSparkline.length - 1] < 0
                  ? "rgb(248,113,113)"
                  : "rgb(52,211,153)";
              return (
                <div
                  key={`${displaySymbol}-${index}`}
                  className="rounded-xl border border-emerald-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(34,211,238,0.35)] sm:p-3"
                >
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_auto_auto_auto_auto] sm:items-center">
                    <div className="flex items-center gap-2">
                      <StockLogo symbol={displaySymbol} />
                      <div className="min-w-[44px] flex-shrink-0">
                        <div className="text-[13px] font-bold leading-[18px] text-slate-100">
                          {displaySymbol}
                        </div>
                      </div>
                    </div>
                    <div className="hidden text-right text-[12px] font-semibold leading-[16px] text-slate-100 sm:block">
                      {formatCurrency(position.currentPrice)}
                    </div>
                    <div className="hidden items-center justify-end sm:flex">
                      <Sparkline data={plSparkline} width={64} stroke={plTone} />
                    </div>
                    <div
                      className={`hidden text-right text-[12px] font-semibold leading-[16px] sm:block ${toneClass}`}
                    >
                      {formatSignedPercent(position.percentPL)}
                    </div>
                    <div
                      className={`hidden text-right text-[12px] font-semibold leading-[16px] sm:block ${toneClass}`}
                    >
                      {formatSignedCurrency(position.dollarPL)}
                    </div>
                    <div className="flex items-center justify-between sm:hidden">
                      <div className="text-[13px] font-semibold leading-[18px] text-slate-100">
                        {formatCurrency(position.currentPrice)}
                      </div>
                      <Sparkline data={plSparkline} stroke={plTone} />
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-300 sm:hidden">
                      <span>% P/L</span>
                      <span className={toneClass}>{formatSignedPercent(position.percentPL)}</span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-300 sm:hidden">
                      <span>Total $ P/L</span>
                      <span className={toneClass}>{formatSignedCurrency(position.dollarPL)}</span>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="rounded-xl border border-emerald-400/30 bg-slate-950/60 p-4 text-[13px] leading-[18px] text-slate-300 shadow-[0_0_18px_-12px_rgba(34,211,238,0.35)]">
              No open positions available.
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <SummaryTile
            label="Total P/L"
            value={formatSignedCurrency(totalDollar)}
            valueTone={plToneClass(totalDollar)}
          />
          <SummaryTile
            label="Portfolio"
            value={formatSignedPercent(portfolioPercent)}
            valueTone={plToneClass(portfolioPercent)}
          />
        </div>
      </div>
    </CardShell>
  );
}
