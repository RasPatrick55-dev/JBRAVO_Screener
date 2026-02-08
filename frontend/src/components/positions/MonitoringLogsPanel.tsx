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
  info: "bg-gray-200 text-gray-800",
  success: "bg-green-200 text-green-900",
  warning: "bg-yellow-200 text-yellow-800",
};

export default function MonitoringLogsPanel({ logs }: MonitoringLogsPanelProps) {
  return (
    <section
      className="overflow-hidden rounded-xl border border-slate-200 bg-surface shadow-card"
      aria-label="Monitoring position logs"
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-4">
        <div>
          <h2 className="text-lg font-semibold text-financial">Monitoring Positions Logs</h2>
          <p className="text-sm text-secondary">Trailing stop and open alerts</p>
        </div>
        <StatusChip label="Always On Track" tone="success" />
      </header>
      <div className="max-h-[32rem] overflow-auto">
        <table className="w-full table-auto text-sm">
          <caption className="sr-only">Latest trailing stop and open-position alerts</caption>
          <colgroup>
            <col className="w-[34%] sm:w-[28%] lg:w-[22%]" />
            <col className="w-[18%] sm:w-[14%] lg:w-[10%]" />
            <col />
          </colgroup>
          <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-secondary">
            <tr>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50 px-4 py-3">
                Timestamp
              </th>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50 px-2 py-3 text-center">
                Type
              </th>
              <th scope="col" className="sticky top-0 z-10 bg-slate-50 px-4 py-3">
                Message
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <tr key={`${log.timestamp}-${index}`}>
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
