import { useEffect, useMemo, useRef, useState } from "react";
import type { LogsChipFilter, LogsStage, ScreenerLogRow, ScreenerLogsResponse } from "./types";
import { fetchNoStoreJson, formatUtcDateTime, normalizeLogLevel, withTs } from "./utils";

interface LogsPanelProps {
  title: string;
  stage: LogsStage;
}

const chipOptions: Array<{ key: LogsChipFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "errors", label: "Errors" },
  { key: "warnings", label: "Warnings" },
  { key: "today", label: "Today" },
];
const REFRESH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 15_000;

const normalizeLogRow = (row: Partial<ScreenerLogRow>): ScreenerLogRow => {
  return {
    ts_utc: row.ts_utc ? String(row.ts_utc) : null,
    level: normalizeLogLevel(row.level),
    message: String(row.message ?? "").trim() || "(no message)",
  };
};

const levelChipClass: Record<string, string> = {
  ERROR: "bg-rose-100 text-rose-700 outline outline-1 outline-rose-300 dark:bg-rose-500/20 dark:text-rose-200 dark:outline-rose-300/45",
  WARN: "bg-amber-100 text-amber-800 outline outline-1 outline-amber-300 dark:bg-amber-500/20 dark:text-amber-200 dark:outline-amber-300/45",
  INFO: "bg-sky-100 text-sky-700 outline outline-1 outline-sky-300 dark:bg-sky-500/20 dark:text-sky-200 dark:outline-sky-300/45",
  SUCCESS:
    "bg-emerald-100 text-emerald-700 outline outline-1 outline-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-200 dark:outline-emerald-300/45",
};

const stageStyleMap: Record<
  LogsStage,
  {
    panel: string;
    inner: string;
    chipActive: string;
    rowHover: string;
  }
> = {
  screener: {
    panel: "jbravo-panel-cyan",
    inner: "jbravo-panel-inner-cyan",
    chipActive:
      "bg-sky-100 text-sky-700 outline-sky-300 dark:bg-sky-500/20 dark:text-sky-200 dark:outline-sky-300/45",
    rowHover: "hover:bg-sky-100/35 dark:hover:bg-slate-800/45",
  },
  backtest: {
    panel: "jbravo-panel-violet",
    inner: "jbravo-panel-inner-violet",
    chipActive:
      "bg-indigo-100 text-indigo-700 outline-indigo-300 dark:bg-indigo-500/20 dark:text-indigo-200 dark:outline-indigo-300/45",
    rowHover: "hover:bg-indigo-100/35 dark:hover:bg-slate-800/45",
  },
  metrics: {
    panel: "jbravo-panel-emerald",
    inner: "jbravo-panel-inner-emerald",
    chipActive:
      "bg-emerald-100 text-emerald-700 outline-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-200 dark:outline-emerald-300/45",
    rowHover: "hover:bg-emerald-100/35 dark:hover:bg-slate-800/45",
  },
};

const isTodayUtc = (value: string | null | undefined): boolean => {
  if (!value) {
    return false;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return false;
  }
  const now = new Date();
  return (
    parsed.getUTCFullYear() === now.getUTCFullYear() &&
    parsed.getUTCMonth() === now.getUTCMonth() &&
    parsed.getUTCDate() === now.getUTCDate()
  );
};

export default function LogsPanel({ title, stage }: LogsPanelProps) {
  const [rows, setRows] = useState<ScreenerLogRow[]>([]);
  const [selectedChip, setSelectedChip] = useState<LogsChipFilter>("all");
  const [sourceDetail, setSourceDetail] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const hasLoadedRef = useRef(false);

  const stageStyle = stageStyleMap[stage] ?? stageStyleMap.screener;

  useEffect(() => {
    let isMounted = true;
    let inFlight = false;

    const load = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      setIsLoading(!hasLoadedRef.current);

      const level =
        selectedChip === "errors" ? "errors" : selectedChip === "warnings" ? "warnings" : "all";
      const today = selectedChip === "today" ? "1" : "0";

      const params = new URLSearchParams({
        stage,
        limit: "200",
        level,
        today,
      });

      try {
        const payload = await fetchNoStoreJson<ScreenerLogsResponse>(
          withTs(`/api/screener/logs?${params.toString()}`),
          REQUEST_TIMEOUT_MS
        );
        if (!isMounted) {
          return;
        }
        const normalizedRows = (payload.rows ?? []).map((row) => normalizeLogRow(row));
        setRows(normalizedRows);
        setSourceDetail(String(payload.source_detail ?? payload.source ?? ""));
        setErrorMessage(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Unable to load logs.";
        setErrorMessage(message);
      } finally {
        if (isMounted) {
          setIsLoading(false);
          hasLoadedRef.current = true;
        }
        inFlight = false;
      }
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [reloadToken, selectedChip, stage]);

  const displayedRows = useMemo(() => {
    let filtered = rows;
    if (selectedChip === "errors") {
      filtered = filtered.filter((row) => normalizeLogLevel(row.level) === "ERROR");
    } else if (selectedChip === "warnings") {
      filtered = filtered.filter((row) => normalizeLogLevel(row.level) === "WARN");
    } else if (selectedChip === "today") {
      filtered = filtered.filter((row) => isTodayUtc(row.ts_utc));
    }
    return filtered;
  }, [rows, selectedChip]);

  return (
    <section className={`flex h-full min-h-0 flex-col rounded-2xl p-3 shadow-card jbravo-panel sm:p-md ${stageStyle.panel}`}>
      <header className="flex items-center justify-between gap-2">
        <h3 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">{title}</h3>
        <button
          type="button"
          onClick={() => setReloadToken((value) => value + 1)}
          aria-label={`Refresh ${title}`}
          className="inline-flex h-7 w-7 items-center justify-center rounded-md outline outline-1 outline-slate-300/80 transition hover:bg-slate-100/70 dark:outline-slate-600/80 dark:hover:bg-slate-700/40"
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 text-secondary" fill="none">
            <path
              d="M20 12a8 8 0 1 1-2.34-5.66L20 9M20 4v5h-5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </header>

      {sourceDetail ? (
        <p className="mt-1 break-all text-[10px] text-secondary sm:truncate sm:text-[11px]" title={sourceDetail}>
          Source: {sourceDetail}
        </p>
      ) : null}

      {errorMessage ? (
        <div
          role="alert"
          className="mt-2 flex items-center justify-between gap-2 rounded-md border border-rose-300/60 bg-rose-50/80 px-2 py-1.5 text-[11px] text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/10 dark:text-rose-200"
        >
          <span className="truncate">{errorMessage}</span>
          <button
            type="button"
            onClick={() => setReloadToken((value) => value + 1)}
            className="rounded px-1.5 py-0.5 font-semibold uppercase tracking-wide outline outline-1 outline-rose-300/70"
          >
            Retry
          </button>
        </div>
      ) : null}

      <div className="mt-2 flex flex-wrap items-center gap-1">
        {chipOptions.map((option) => {
          const isActive = option.key === selectedChip;
          return (
            <button
              key={option.key}
              type="button"
              onClick={() => setSelectedChip(option.key)}
              aria-pressed={isActive}
              className={
                "rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.07em] outline outline-1 outline-offset-[-1px] transition " +
                (isActive
                  ? stageStyle.chipActive
                  : "text-secondary outline-slate-300/80 hover:text-primary dark:outline-slate-600/80")
              }
            >
              {option.label}
            </button>
          );
        })}
      </div>

      <div className="mt-2 min-h-0">
        <div className={`max-h-[240px] overflow-auto rounded-xl jbravo-panel-inner sm:max-h-[280px] ${stageStyle.inner}`}>
          <table className="w-full table-fixed">
            <caption className="sr-only">
              {title} entries with UTC timestamp, level, and message.
            </caption>
            <colgroup>
              <col className="w-[31%] sm:w-[28%]" />
              <col className="w-[14%] sm:w-[13%]" />
              <col className="w-[55%] sm:w-[59%]" />
            </colgroup>
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
                <th scope="col" className="whitespace-nowrap px-2 py-1.5 text-left">
                  Time (UTC)
                </th>
                <th scope="col" className="px-2 py-1.5 text-center">
                  Level
                </th>
                <th scope="col" className="px-2 py-1.5 text-left">
                  Message
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading
                ? Array.from({ length: 6 }).map((_, index) => (
                    <tr key={`log-skeleton-${stage}-${index}`} className="border-b border-slate-200/70 dark:border-slate-700/70">
                      <td colSpan={3} className="px-2 py-2">
                        <div className="h-3.5 w-full animate-pulse rounded bg-slate-200/75 dark:bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && displayedRows.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-2 py-8 text-center text-sm text-secondary">
                    No results available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? displayedRows.map((row, index) => {
                    const level = normalizeLogLevel(row.level);
                    const formattedTs = formatUtcDateTime(row.ts_utc);
                    const [datePart, timePart] = formattedTs === "--" ? ["--", ""] : formattedTs.split(" ");
                    return (
                      <tr
                        key={`${stage}-${row.ts_utc ?? "na"}-${index}`}
                        className={`border-b border-slate-200/70 text-[10px] transition-colors dark:border-slate-700/70 sm:text-[11px] ${stageStyle.rowHover}`}
                      >
                        <td
                          title={formattedTs}
                          className="font-cousine px-2 py-2 text-left tabular-nums text-secondary"
                        >
                          <span className="block overflow-hidden text-ellipsis whitespace-nowrap leading-tight">
                            {datePart}
                          </span>
                          {timePart ? (
                            <span className="block overflow-hidden text-ellipsis whitespace-nowrap leading-tight text-secondary/90">
                              {timePart}
                            </span>
                          ) : null}
                        </td>
                        <td className="px-2 py-2 text-center">
                          <span
                            className={`inline-flex rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase sm:text-[10px] ${levelChipClass[level]}`}
                          >
                            {level}
                          </span>
                        </td>
                        <td className="break-words px-2 py-2 text-left text-primary [overflow-wrap:anywhere]">
                          {row.message}
                        </td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
