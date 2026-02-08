import MonitoringLogsPanel, { type MonitoringLogItem } from "./MonitoringLogsPanel";
import { positionTableGridColumns } from "./layout";
import PositionRow, { type PositionRowProps } from "./PositionRow";
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
        <div role="alert" className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Position data could not be fully loaded. Displaying available data.
        </div>
      ) : null}

      <section className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-card">
        <div className="overflow-x-auto sm:overflow-x-auto">
          <div className="min-w-full">
            <div
              className={`${positionTableGridColumns} hidden items-center gap-0 divide-x divide-slate-200/50 border-b border-slate-200 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-secondary sm:grid`}
              role="row"
            >
              {columnLabels.map((label) => (
                <div key={label} className="truncate px-3 text-center">
                  {label}
                </div>
              ))}
            </div>
            <div role="rowgroup">
              {isLoading ? (
                <>
                  <div className="space-y-2 px-4 py-3 sm:hidden">
                    {skeletonKey.map((key) => (
                      <div key={`mobile-${key}`} className="rounded-lg border border-slate-200 px-3 py-3">
                        <div className="h-5 w-24 rounded bg-slate-100" />
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          {Array.from({ length: 6 }).map((_, index) => (
                            <div key={`${key}-${index}`} className="h-10 rounded bg-slate-100" />
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                  {skeletonKey.map((key) => (
                    <div
                      key={key}
                      className={`${positionTableGridColumns} hidden items-center gap-0 divide-x divide-slate-200/50 border-b border-slate-200 px-4 py-3.5 sm:grid`}
                      role="row"
                    >
                      {columnLabels.map((label) => (
                        <div
                          key={`${key}-${label}`}
                          className="mx-auto h-4 w-4/5 rounded bg-slate-100"
                          aria-hidden="true"
                        />
                      ))}
                    </div>
                  ))}
                </>
              ) : positions.length > 0 ? (
                positions.map((position, index) => (
                  <PositionRow key={`${position.symbol}-${index}`} {...position} />
                ))
              ) : (
                <div className="px-4 py-8 text-sm text-secondary">No open positions available.</div>
              )}
            </div>
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
