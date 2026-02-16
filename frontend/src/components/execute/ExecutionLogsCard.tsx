import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ExecuteLogLevelFilter,
  ExecuteLogRow,
  ExecuteLogStage,
  ExecuteLogsResponse,
  ExecuteLsxFilter,
} from "./types";
import {
  LSX_CHIPS,
  fetchJsonNoStore,
  filterLogsClient,
  formatDateTimeUtc,
  logLevelChipClass,
  normalizeLogLevel,
  parseSseJson,
} from "./utils";

const stageOptions: Array<{ key: ExecuteLogStage; label: string }> = [
  { key: "execute", label: "Execute" },
  { key: "monitor", label: "Monitor" },
  { key: "pipeline", label: "Pipeline" },
];

const cycleLevelFilter = (value: ExecuteLogLevelFilter): ExecuteLogLevelFilter => {
  if (value === "all") {
    return "errors";
  }
  if (value === "errors") {
    return "warnings";
  }
  return "all";
};

const levelFilterLabel = (value: ExecuteLogLevelFilter): string => {
  if (value === "errors") {
    return "Errors";
  }
  if (value === "warnings") {
    return "Warnings";
  }
  return "All";
};

export default function ExecutionLogsCard() {
  const [rows, setRows] = useState<ExecuteLogRow[]>([]);
  const [stage, setStage] = useState<ExecuteLogStage>("execute");
  const [levelFilter, setLevelFilter] = useState<ExecuteLogLevelFilter>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [lsx, setLsx] = useState<ExecuteLsxFilter>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [hasError, setHasError] = useState(false);
  const hasLoadedOnceRef = useRef(false);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setDebouncedQuery(searchQuery.trim());
    }, 200);
    return () => window.clearTimeout(timeout);
  }, [searchQuery]);

  useEffect(() => {
    let isMounted = true;

    const normalizeRows = (payload: ExecuteLogsResponse | null) =>
      (payload?.rows ?? []).map((row) => ({
        ...row,
        level: normalizeLogLevel(row.level),
      }));

    const applyPayload = (payload: ExecuteLogsResponse | null) => {
      if (!isMounted) {
        return;
      }
      setRows(normalizeRows(payload));
      setHasError(!payload);
      setIsLoading(false);
      if (payload) {
        hasLoadedOnceRef.current = true;
        setHasLoadedOnce(true);
      }
    };

    const params = new URLSearchParams({
      stage,
      limit: "200",
      level: levelFilter,
      today: "0",
      q: debouncedQuery,
    });
    if (lsx !== "all") {
      params.set("lsx", lsx);
    }

    const loadFallback = async () => {
      setIsLoading(!hasLoadedOnceRef.current);
      const params = new URLSearchParams({
        stage,
        limit: "200",
        level: levelFilter,
        today: "0",
        q: debouncedQuery,
      });
      if (lsx !== "all") {
        params.set("lsx", lsx);
      }

      const payload = await fetchJsonNoStore<ExecuteLogsResponse>(
        `/api/execute/logs?${params.toString()}&ts=${Date.now()}`
      );
      applyPayload(payload);
    };

    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      void loadFallback();
      return () => {
        isMounted = false;
      };
    }

    setIsLoading(!hasLoadedOnceRef.current);
    const source = new EventSource(`/api/execute/logs/stream?${params.toString()}`);
    source.onmessage = (event) => {
      const payload = parseSseJson<ExecuteLogsResponse>(event.data);
      if (!payload) {
        return;
      }
      applyPayload(payload);
    };
    source.onerror = () => {
      if (!isMounted) {
        return;
      }
      if (!hasLoadedOnceRef.current) {
        setHasError(true);
        setIsLoading(false);
      }
    };

    return () => {
      isMounted = false;
      source.close();
    };
  }, [debouncedQuery, levelFilter, lsx, stage]);

  const visibleRows = useMemo(
    () => filterLogsClient(rows, searchQuery, lsx),
    [lsx, rows, searchQuery]
  );

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel jbravo-panel-amber p-3 sm:p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="w-full sm:w-auto">
          <h2 className="font-arimo text-[30px] font-semibold leading-none text-primary sm:text-[34px]">
            Execution Logs
          </h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {stageOptions.map((option) => {
              const active = stage === option.key;
              return (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => setStage(option.key)}
                  aria-pressed={active}
                  className={
                    "rounded-md px-3 py-1.5 text-xs font-medium outline outline-1 outline-offset-[-1px] transition sm:px-4 sm:py-2 sm:text-sm " +
                    (active
                      ? "bg-sky-500/20 text-sky-200 outline-sky-400/60"
                      : "text-slate-300 outline-slate-600/80 hover:text-slate-100")
                  }
                >
                  {option.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex w-full flex-wrap items-center justify-end gap-2 sm:w-auto">
          <button
            type="button"
            aria-label="Search execution logs"
            onClick={() => setShowSearch((current) => !current)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md outline outline-1 outline-slate-500/55 transition hover:bg-slate-700/35"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-slate-200" fill="none">
              <path
                d="M11 19a8 8 0 1 1 5.29-2l4.35 4.36"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>

          {showSearch ? (
            <label className="relative">
              <span className="sr-only">Search logs</span>
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search"
                className="h-8 w-32 rounded-md border border-slate-600/70 bg-slate-900/70 px-2 py-1 text-xs text-slate-100 outline-none transition focus:border-sky-400 sm:w-44"
              />
            </label>
          ) : null}

          <button
            type="button"
            aria-label={`Logs level filter ${levelFilterLabel(levelFilter)}`}
            title={`Level: ${levelFilterLabel(levelFilter)}`}
            onClick={() => setLevelFilter((current) => cycleLevelFilter(current))}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md outline outline-1 outline-slate-500/55 transition hover:bg-slate-700/35"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-slate-200" fill="none">
              <path
                d="M4 5h16l-6 7v6l-4 1v-7z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>

          {LSX_CHIPS.map((chip) => {
            const active = lsx === chip.key;
            return (
              <button
                key={`logs-chip-${chip.key}`}
                type="button"
                aria-pressed={active}
                onClick={() => setLsx((current) => (current === chip.key ? "all" : chip.key))}
                className={
                  "h-8 min-w-8 rounded-md px-2 text-xs font-semibold uppercase tracking-[0.08em] outline outline-1 outline-offset-[-1px] transition " +
                  (active
                    ? "bg-sky-500/20 text-sky-200 outline-sky-400/60"
                    : "text-slate-300 outline-slate-600/80 hover:text-slate-100")
                }
              >
                {chip.label}
              </button>
            );
          })}
        </div>
      </header>

      <div className="mt-3 overflow-x-auto">
        <div className="min-w-[760px] max-h-[320px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-amber">
          <table className="min-w-full table-auto">
            <thead className="sticky top-0 z-10 bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-700/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Time (UTC)
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Level
                </th>
                <th scope="col" className="px-3 py-2 text-left">
                  Message
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading && !hasLoadedOnce
                ? Array.from({ length: 6 }).map((_, index) => (
                    <tr key={`logs-skeleton-${index}`} className="border-b border-slate-700/70">
                      <td colSpan={3} className="px-3 py-3">
                        <div className="h-4 w-full animate-pulse rounded bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && visibleRows.length === 0 ? (
                <tr>
                  <td colSpan={3} className="px-3 py-8 text-center text-sm text-secondary">
                    No execution logs available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? visibleRows.map((row, index) => {
                    const level = normalizeLogLevel(row.level);
                    const ts = formatDateTimeUtc(row.ts_utc ?? null);
                    return (
                      <tr
                        key={`${row.ts_utc ?? "log-ts"}-${index}`}
                        className="border-b border-slate-700/70 text-sm transition-colors hover:bg-slate-800/45"
                      >
                        <td className="font-cousine px-3 py-2 text-left tabular-nums text-slate-300 whitespace-nowrap">
                          <div className="flex flex-col whitespace-nowrap leading-tight">
                            <span>{ts.time}</span>
                            <span className="text-[10px] text-slate-400">{ts.date}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2 text-left whitespace-nowrap">
                          <span
                            className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1 ${logLevelChipClass(
                              level
                            )}`}
                          >
                            {level}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-left text-slate-100">{row.message || ""}</td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>

      {hasError ? (
        <p className="mt-3 text-xs text-rose-300">Execution logs endpoint unavailable. Showing fallback state.</p>
      ) : null}
    </section>
  );
}
