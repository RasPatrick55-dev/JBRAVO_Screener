import type { OrderLogLevel, OrderLogRow } from "./types";

interface DailyOrderLogsCardProps {
  rows: OrderLogRow[];
  isLoading?: boolean;
}

const levelChipClass: Record<OrderLogLevel, string> = {
  success: "jbravo-chip-success",
  info: "bg-sky-100 text-sky-700 outline outline-1 outline-sky-300 dark:bg-sky-500/20 dark:text-sky-200 dark:outline-sky-300/45",
  warning: "bg-amber-100 text-amber-800 outline outline-1 outline-amber-300 dark:bg-amber-500/20 dark:text-amber-200 dark:outline-amber-300/45",
};

const formatTs = (value: string): string => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value || "--";
  }
  return parsed.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

export default function DailyOrderLogsCard({ rows, isLoading = false }: DailyOrderLogsCardProps) {
  return (
    <section className="rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-amber" aria-label="Daily order logs">
      <header>
        <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Daily Order Logs</h2>
        <p className="mt-1 text-xs text-secondary">Execution and order lifecycle events</p>
      </header>

      <div className="mt-3 h-72 overflow-y-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-amber">
        {isLoading ? (
          <div className="space-y-2 p-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={`daily-log-skeleton-${index}`} className="h-11 animate-pulse rounded bg-slate-200/80 dark:bg-slate-700/70" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <div className="flex h-full items-center justify-center px-4 text-sm text-secondary">No logs today.</div>
        ) : (
          <ul className="divide-y divide-slate-200/70 dark:divide-slate-700/70">
            {rows.map((row, index) => (
              <li key={`${row.ts}-${index}`} className="px-3 py-2.5 transition-colors hover:bg-sky-100/35 dark:hover:bg-slate-800/45">
                <div className="flex items-start justify-between gap-3">
                  <span className="font-cousine text-xs tabular-nums text-secondary">{formatTs(row.ts)}</span>
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase outline outline-1 ${
                      levelChipClass[row.level]
                    }`}
                  >
                    {row.level}
                  </span>
                </div>
                <p className="mt-1 text-sm text-primary" title={row.message}>
                  {row.message}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
