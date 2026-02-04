import { useMemo, useState } from "react";
import type { LogEntry } from "../../types/ui";

interface LatestLogsRow {
  dateTime: string;
  script: string;
  text: string;
  level?: LogEntry["level"];
}

interface LatestLogsTableProps {
  title?: string;
  rows: LatestLogsRow[];
  enableFilter?: boolean;
}

export default function LatestLogsTable({
  title = "Latest Logs",
  rows,
  enableFilter = true,
}: LatestLogsTableProps) {
  const levelTone: Record<NonNullable<LatestLogsRow["level"]>, string> = {
    INFO: "border-sky-500/30 bg-sky-500/15 text-sky-200",
    WARN: "border-amber-500/30 bg-amber-500/15 text-amber-200",
    ERROR: "border-rose-500/30 bg-rose-500/15 text-rose-200",
    SUCCESS: "border-emerald-500/30 bg-emerald-500/15 text-emerald-200",
  };
  const scriptOptions = useMemo(() => {
    const options = Array.from(new Set(rows.map((row) => row.script).filter(Boolean)));
    return ["All Scripts", ...options];
  }, [rows]);
  const [selectedScript, setSelectedScript] = useState("All Scripts");
  const filteredRows =
    selectedScript === "All Scripts"
      ? rows
      : rows.filter((row) => row.script === selectedScript);

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-400/30 bg-gradient-to-br from-slate-950/95 via-slate-900/85 to-slate-800/70 p-4 shadow-[0_18px_40px_-24px_rgba(148,163,184,0.4)] sm:p-5">
      <div className="rounded-xl border border-slate-500/40 bg-gradient-to-r from-slate-950/90 via-slate-900/90 to-slate-800/80 p-2.5 shadow-[0_0_24px_-12px_rgba(148,163,184,0.45)] sm:p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
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
          {enableFilter ? (
            <label className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-slate-200/70">
              Script
              <select
                className="rounded-lg border border-slate-500/40 bg-slate-950/80 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.06em] text-slate-100"
                value={selectedScript}
                onChange={(event) => setSelectedScript(event.target.value)}
              >
                {scriptOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
      </div>

      <div className="mt-4 max-h-[360px] overflow-x-hidden overflow-y-auto rounded-xl border border-slate-500/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(148,163,184,0.35)] sm:p-3">
        <table className="w-full table-fixed font-cousine text-sm text-slate-200">
          <thead className="hidden text-left text-[11px] font-bold uppercase tracking-[0.08em] text-slate-200/70 sm:table-header-group">
            <tr>
              <th className="pb-2 pr-4 font-semibold sm:w-[28%]">Date &amp; Time</th>
              <th className="pb-2 pr-4 font-semibold sm:w-[20%]">Script</th>
              <th className="pb-2 font-semibold sm:w-[52%]">Log Text</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.length ? (
              filteredRows.map((row, index) => (
                <tr
                  key={`${row.dateTime}-${row.script}-${index}`}
                  className="block border-b border-slate-800/70 py-3 last:border-0 sm:table-row sm:py-0"
                >
                  <td className="block pb-3 text-[12px] text-slate-300 sm:table-cell sm:py-2 sm:pr-4 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400 sm:hidden">
                      Date &amp; Time
                    </span>
                    {row.dateTime}
                  </td>
                  <td className="block pb-3 text-[12px] text-slate-300 sm:table-cell sm:py-2 sm:pr-4 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400 sm:hidden">
                      Script
                    </span>
                    <span className="break-words" title={row.script}>
                      {row.script}
                    </span>
                  </td>
                  <td className="block text-[12px] text-slate-100 sm:table-cell sm:py-2 sm:align-top">
                    <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400 sm:hidden">
                      Log Text
                    </span>
                    <div className="flex flex-wrap items-start gap-2">
                      {row.level ? (
                        <span
                          className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] ${
                            levelTone[row.level]
                          }`}
                        >
                          {row.level}
                        </span>
                      ) : null}
                      <span className="min-w-0 flex-1 break-all text-[12px] leading-relaxed text-slate-100/90 sm:break-words">
                        {row.text}
                      </span>
                    </div>
                  </td>
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
