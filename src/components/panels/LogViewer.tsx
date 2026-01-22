import type { LogEntry, LogViewerProps, StatusTone } from "../../types/ui";
import StatusBadge from "../badges/StatusBadge";

const levelTone: Record<LogEntry["level"], StatusTone> = {
  INFO: "info",
  WARN: "warning",
  ERROR: "error",
  SUCCESS: "success",
};

export default function LogViewer({ title, entries, statusLabel, actionLabel }: LogViewerProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3">
        <div className="flex items-center gap-2">
          <div className="text-sm font-semibold text-slate-800">{title}</div>
          {statusLabel ? (
            <span className="inline-flex items-center gap-1 text-xs text-emerald-600">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              {statusLabel}
            </span>
          ) : null}
        </div>
        {actionLabel ? (
          <span className="rounded-md border border-slate-200 px-2 py-1 text-[11px] font-semibold text-slate-500">
            {actionLabel}
          </span>
        ) : null}
      </div>
      <div
        role="textbox"
        aria-readonly="true"
        className="max-h-64 overflow-y-auto px-5 py-2 font-mono text-xs text-slate-700"
      >
        {entries.map((entry, index) => (
          <div
            key={`${entry.time}-${index}`}
            className="grid grid-cols-[72px_84px_1fr] items-center gap-3 border-b border-slate-100 py-2"
          >
            <div className="text-slate-400">{entry.time}</div>
            <StatusBadge label={entry.level} tone={levelTone[entry.level]} size="sm" />
            <div className="text-slate-700">{entry.message}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
