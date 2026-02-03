interface LatestLogsRow {
  date: string;
  category: string;
  text: string;
}

interface LatestLogsTableProps {
  title?: string;
  rows: LatestLogsRow[];
}

export default function LatestLogsTable({ title = "Latest Logs", rows }: LatestLogsTableProps) {
  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-400/30 bg-gradient-to-br from-slate-950/95 via-slate-900/85 to-slate-800/70 p-4 shadow-[0_18px_40px_-24px_rgba(148,163,184,0.4)] sm:p-5">
      <div className="rounded-xl border border-slate-500/40 bg-gradient-to-r from-slate-950/90 via-slate-900/90 to-slate-800/80 p-2.5 shadow-[0_0_24px_-12px_rgba(148,163,184,0.45)] sm:p-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-500/20 text-slate-200">
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
              <path d="M4 4h16v12H7l-3 3V4z" />
              <path d="M8 8h8" />
              <path d="M8 12h6" />
            </svg>
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-bold leading-[22px] text-white">{title}</h3>
            <span className="text-[11px] text-slate-200/70">Recent system events</span>
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-x-auto rounded-xl border border-slate-500/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(148,163,184,0.35)] sm:p-3">
        <table className="w-full min-w-[420px] font-cousine text-sm text-slate-200">
          <thead className="text-left text-[11px] font-bold uppercase tracking-[0.08em] text-slate-200/70">
            <tr>
              <th className="pb-2 pr-4 font-semibold">Date</th>
              <th className="pb-2 pr-4 font-semibold">Category</th>
              <th className="pb-2 font-semibold">Log Text</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {rows.length ? (
              rows.map((row, index) => (
                <tr key={`${row.date}-${index}`}>
                  <td className="py-2 pr-4 align-top text-[12px] text-slate-300">
                    {row.date}
                  </td>
                  <td className="py-2 pr-4 align-top text-[12px] text-slate-300">
                    {row.category}
                  </td>
                  <td className="py-2 align-top text-[12px] text-slate-100">{row.text}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="py-4 text-center text-[12px] text-slate-300">
                  No recent logs available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
