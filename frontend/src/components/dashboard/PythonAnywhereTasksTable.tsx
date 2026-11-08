import type { SourceKind } from "../ui/sourceState";
export interface PythonAnywhereTaskRow {
  name: string;
  frequency: string;
  time: string;
}

export function scheduledTaskRows(payload: unknown): PythonAnywhereTaskRow[] {
  if (!payload || typeof payload !== "object") return [];
  const data = payload as { ok?: boolean; tasks?: unknown };
  if (data.ok !== true || !Array.isArray(data.tasks)) return [];
  return data.tasks.flatMap(value => {
    if (!value || typeof value !== "object") return [];
    const row = value as Record<string, unknown>;
    if (typeof row.name !== "string" || !row.name.trim()) return [];
    const text = (value: unknown) => typeof value === "string" && value.trim() ? value.trim() : "Unknown";
    return [{ name: row.name.trim(), frequency: text(row.frequency), time: text(row.time) }];
  });
}

interface PythonAnywhereTasksTableProps {
  title?: string;
  rows: PythonAnywhereTaskRow[];
  observedAt?: string | null;
  sourceStatus?: SourceKind;
}

export default function PythonAnywhereTasksTable({
  title = "PythonAnywhere Tasks",
  rows,
  observedAt,
  sourceStatus,
}: PythonAnywhereTasksTableProps) {
  return (
    <div className="jbravo-inverse flex h-full min-w-0 flex-col rounded-2xl border border-emerald-400/30 bg-gradient-to-br from-slate-950/95 via-emerald-950/80 to-emerald-900/70 p-4 shadow-[0_18px_40px_-24px_rgba(16,185,129,0.45)] sm:p-5">
      <div className="rounded-xl border border-emerald-400/40 bg-gradient-to-r from-slate-950/90 via-emerald-950/90 to-emerald-900/80 p-2.5 shadow-[0_0_24px_-12px_rgba(16,185,129,0.55)] sm:p-3">
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
              <path d="M8 6h13" />
              <path d="M8 12h13" />
              <path d="M8 18h13" />
              <path d="M3 6h.01" />
              <path d="M3 12h.01" />
              <path d="M3 18h.01" />
            </svg>
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-bold leading-[22px] text-white">{title}</h3>
            <span className="text-[11px] text-emerald-100/80">Provider configuration, not execution status</span>
          </div>
        </div>
      </div>

      {sourceStatus ? <p role="status" className="mt-3 text-[11px] text-secondary">{sourceStatus === "loading" ? "Loading PythonAnywhere schedule." : sourceStatus === "stale" ? "Schedule feed stale. Showing last known configuration, not the current schedule." : `Schedule feed: ${sourceStatus}.`}</p> : null}
      <p className="mt-3 text-[10px] text-emerald-100/70">Provider observation: {observedAt || "unknown"}</p>
      <div className="mt-4 overflow-x-hidden rounded-xl border border-emerald-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(16,185,129,0.35)] sm:p-3">
        <table className="w-full table-fixed font-cousine text-sm text-emerald-50">
          <thead className="hidden text-left text-[11px] font-bold uppercase tracking-[0.08em] text-emerald-100/70 sm:table-header-group">
            <tr>
              <th className="pb-2 pr-4 font-semibold sm:w-[48%]">Task Name</th>
              <th className="pb-2 pr-4 font-semibold sm:w-[26%]">Frequency</th>
              <th className="pb-2 font-semibold sm:w-[26%]">Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row, index) => (
                <tr
                  key={`${row.name}-${index}`}
                  className="block border-b border-emerald-900/60 py-3 last:border-0 sm:table-row sm:py-0"
                >
                  <td className="block pb-3 text-[12px] text-emerald-100 sm:table-cell sm:py-2 sm:pr-4 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-100/70 sm:hidden">
                      Task Name
                    </span>
                    <span className="break-words">{row.name}</span>
                  </td>
                  <td className="block pb-3 text-[12px] text-emerald-200/80 sm:table-cell sm:py-2 sm:pr-4 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-100/70 sm:hidden">
                      Frequency
                    </span>
                    {row.frequency}
                  </td>
                  <td className="block text-[12px] text-emerald-50 sm:table-cell sm:py-2 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-100/70 sm:hidden">
                      Time
                    </span>
                    {row.time}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="py-4 text-center text-[12px] text-emerald-100/70">
                  {sourceStatus === "loading" ? "Loading scheduled tasks." : "Scheduled tasks unavailable or unverified. Current schedule is unknown."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
