import StatusChip from "../dashboard/StatusChip";

export type MonitoringLogType = "info" | "success" | "warning";

export interface MonitoringLogItem {
  timestamp: string;
  type: MonitoringLogType;
  message: string;
}

export interface MonitoringLogsPanelProps {
  logs: MonitoringLogItem[];
}

const badgeToneClass: Record<MonitoringLogType, string> = {
  info: "bg-sky-100 text-sky-700 dark:bg-sky-500/20 dark:text-sky-200",
  success: "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200",
  warning: "bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200",
};

export default function MonitoringLogsPanel({ logs }: MonitoringLogsPanelProps) {
  return (
    <section
      className="overflow-hidden rounded-xl shadow-card jbravo-panel jbravo-panel-cyan"
      aria-label="Monitoring position logs"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200/70 px-4 py-4 dark:border-slate-700/70">
        <div>
          <h2 className="text-lg font-semibold text-financial">Monitoring Positions Logs</h2>
          <p className="text-sm text-secondary">Trailing stop and open alerts</p>
        </div>
        <StatusChip label="Always On Track" tone="success" />
      </header>
      <div className="max-h-[32rem] overflow-auto">
        <div className="divide-y divide-slate-200/70 sm:hidden dark:divide-slate-700/70">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <article key={`${log.timestamp}-${index}`} className="px-4 py-3 transition-colors hover:bg-sky-100/35 dark:hover:bg-slate-800/45">
                <div className="flex items-start justify-between gap-3">
                  <p className="font-cousine text-xs tabular-nums text-secondary">{log.timestamp}</p>
                  <span className={`${badgeToneClass[log.type]} inline-flex rounded-full px-2 py-0.5 text-xs font-semibold capitalize`}>
                    {log.type}
                  </span>
                </div>
                <p className="mt-2 text-sm text-financial" title={log.message}>
                  {log.message}
                </p>
              </article>
            ))
          ) : (
            <div className="px-4 py-6 text-center text-sm text-secondary">No monitoring alerts available.</div>
          )}
        </div>
        <table className="hidden w-full table-auto text-sm sm:table">
          <caption className="sr-only">Latest trailing stop and open-position alerts</caption>
          <colgroup>
            <col className="w-[34%] sm:w-[28%] lg:w-[22%]" />
            <col className="w-[18%] sm:w-[14%] lg:w-[10%]" />
            <col />
          </colgroup>
          <thead className="bg-slate-50/60 text-left text-xs font-semibold uppercase tracking-wide text-secondary dark:bg-slate-900/45">
            <tr>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50/60 px-4 py-3 dark:bg-slate-900/45">
                Timestamp
              </th>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50/60 px-2 py-3 text-center dark:bg-slate-900/45">
                Type
              </th>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50/60 px-4 py-3 dark:bg-slate-900/45">
                Message
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/70 dark:divide-slate-700/70">
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <tr key={`${log.timestamp}-${index}`} className="transition-colors hover:bg-sky-100/35 dark:hover:bg-slate-800/45">
                  <td className="px-4 py-3 font-cousine tabular-nums text-secondary">{log.timestamp}</td>
                  <td className="px-2 py-3 text-center">
                    <span className={`${badgeToneClass[log.type]} inline-flex rounded-full px-2 py-0.5 text-xs font-semibold capitalize`}>
                      {log.type}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <p className="truncate text-financial" title={log.message}>
                      {log.message}
                    </p>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="px-4 py-6 text-center text-sm text-secondary">
                  No monitoring alerts available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
