import MonitoringLogsPanel, { type MonitoringLogItem } from "./MonitoringLogsPanel";
import PositionRow, { PositionRowMobile, type PositionRowProps } from "./PositionRow";
import PositionSummary, { type PositionSummaryProps } from "./PositionSummary";

export interface PositionsTabProps {
  positions: PositionRowProps[];
  summary: PositionSummaryProps;
  logs: MonitoringLogItem[];
  isLoading?: boolean;
  hasError?: boolean;
}

const columnLabels = [
  "Symbol",
  "Total Shares",
  "Avg Entry Price",
  "Current Price",
  "Total P/L",
  "Days",
  "Trailing Stop",
  "Captured P/L",
];

const skeletonKey = ["skeleton-1", "skeleton-2"];
const desktopColumnWidths = ["14.634%", "12.195%", "12.195%", "12.195%", "12.195%", "12.195%", "12.195%", "12.195%"];

const desktopHeaderCellClass = (index: number) => {
  if (index === 0) {
    return "py-3 pl-7 pr-3 text-center text-xs font-semibold uppercase tracking-wide text-secondary";
  }
  if (index === columnLabels.length - 1) {
    return "border-l border-slate-200/50 py-3 pl-3 pr-7 text-center text-xs font-semibold uppercase tracking-wide text-secondary";
  }
  return "border-l border-slate-200/50 px-3 py-3 text-center text-xs font-semibold uppercase tracking-wide text-secondary";
};

const desktopLoadingCellClass = (index: number) => {
  if (index === 0) {
    return "py-3.5 pl-7 pr-3 align-middle";
  }
  if (index === columnLabels.length - 1) {
    return "border-l border-slate-200/50 py-3.5 pl-3 pr-7 align-middle";
  }
  return "border-l border-slate-200/50 px-3 py-3.5 align-middle";
};

export default function PositionsTab({
  positions,
  summary,
  logs,
  isLoading = false,
  hasError = false,
}: PositionsTabProps) {
  return (
    <section className="space-y-6" aria-label="Positions tab">
      <header className="space-y-1">
        <h1 className="text-3xl font-semibold tracking-tight text-financial sm:text-4xl">Positions</h1>
        <p className="text-sm text-secondary sm:text-base">
          Monitor your open positions, performance metrics, and trading activity
        </p>
      </header>

      {hasError ? (
        <div role="alert" className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-300/40 dark:bg-amber-500/10 dark:text-amber-200">
          Position data could not be fully loaded. Displaying available data.
        </div>
      ) : null}

      <section className="overflow-hidden rounded-xl shadow-card jbravo-panel jbravo-panel-emerald">
        <div className="overflow-x-auto sm:overflow-x-auto">
          <div className="sm:hidden">
            {isLoading ? (
              <div className="space-y-2 px-4 py-3">
                {skeletonKey.map((key) => (
                  <div key={`mobile-${key}`} className="rounded-lg px-3 py-3 jbravo-panel-inner jbravo-panel-inner-emerald">
                    <div className="h-5 w-24 rounded bg-slate-100" />
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      {Array.from({ length: 6 }).map((_, index) => (
                        <div key={`${key}-${index}`} className="h-10 rounded bg-slate-100" />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : positions.length > 0 ? (
              positions.map((position, index) => (
                <PositionRowMobile key={`${position.symbol}-${index}`} {...position} />
              ))
            ) : (
              <div className="px-4 py-8 text-sm text-secondary">No open positions available.</div>
            )}
          </div>

          <table className="hidden min-w-full table-fixed sm:table">
            <caption className="sr-only">Open positions with share count, prices, trailing stop, and profit and loss</caption>
            <colgroup>
              {desktopColumnWidths.map((width, index) => (
                <col key={columnLabels[index]} style={{ width }} />
              ))}
            </colgroup>
            <thead className="border-b border-slate-200/70 dark:border-slate-700/70">
              <tr>
                {columnLabels.map((label, index) => (
                  <th key={label} scope="col" className={desktopHeaderCellClass(index)}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                skeletonKey.map((key) => (
                  <tr key={key} className="border-b border-slate-200/70 dark:border-slate-700/70">
                    {columnLabels.map((label, index) => (
                      <td key={`${key}-${label}`} className={desktopLoadingCellClass(index)}>
                        <div className="mx-auto h-4 w-4/5 rounded bg-slate-100" aria-hidden="true" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : positions.length > 0 ? (
                positions.map((position, index) => (
                  <PositionRow key={`${position.symbol}-${index}`} {...position} />
                ))
              ) : (
                <tr>
                  <td colSpan={columnLabels.length} className="px-4 py-8 text-sm text-secondary">
                    No open positions available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <div className="sr-only" aria-live="polite">
            {isLoading ? "Loading positions" : `${positions.length} open positions loaded`}
          </div>
        </div>
      </section>

      <section className="overflow-visible rounded-xl">
        <div className="overflow-x-auto">
          <div className="min-w-full">
            <PositionSummary {...summary} />
          </div>
        </div>
      </section>

      <MonitoringLogsPanel logs={logs} />
    </section>
  );
}
