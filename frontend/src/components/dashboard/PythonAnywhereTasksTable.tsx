interface PythonAnywhereTaskRow {
  name: string;
  frequency: string;
  time: string;
}

interface PythonAnywhereTasksTableProps {
  title?: string;
  rows: PythonAnywhereTaskRow[];
}

export default function PythonAnywhereTasksTable({
  title = "PythonAnywhere Tasks",
  rows,
}: PythonAnywhereTasksTableProps) {
  return (
    <div className="flex h-full flex-col rounded-2xl border border-emerald-400/30 bg-gradient-to-br from-slate-950/95 via-emerald-950/80 to-emerald-900/70 p-4 shadow-[0_18px_40px_-24px_rgba(16,185,129,0.45)] sm:p-5">
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
            <span className="text-[11px] text-emerald-100/80">Scheduled automation</span>
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto rounded-xl border border-emerald-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(16,185,129,0.35)] sm:p-3">
        <table className="w-full min-w-[360px] font-cousine text-sm text-emerald-50">
          <thead className="text-left text-[11px] font-bold uppercase tracking-[0.08em] text-emerald-100/70">
            <tr>
              <th className="pb-2 pr-4 font-semibold">Task Name</th>
              <th className="pb-2 pr-4 font-semibold">Frequency</th>
              <th className="pb-2 font-semibold">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-emerald-900/60">
            {rows.length ? (
              rows.map((row, index) => (
                <tr key={`${row.name}-${index}`}>
                  <td className="py-2 pr-4 align-top text-[12px] text-emerald-100">
                    {row.name}
                  </td>
                  <td className="py-2 pr-4 align-top text-[12px] text-emerald-200/80">
                    {row.frequency}
                  </td>
                  <td className="py-2 align-top text-[12px] text-emerald-50">{row.time}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="py-4 text-center text-[12px] text-emerald-100/70">
                  No tasks scheduled.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
