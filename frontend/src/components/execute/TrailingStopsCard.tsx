import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ExecuteLsxFilter,
  ExecuteStatusScope,
  ExecuteTrailingStopRow,
  ExecuteTrailingStopsResponse,
} from "./types";
import {
  LSX_CHIPS,
  cycleStatusScope,
  fetchJsonNoStore,
  filterTrailingClient,
  formatCurrency,
  formatNumber,
  normalizeTrailingStatus,
  parseSseJson,
  statusScopeLabel,
  trailingStatusChipClass,
} from "./utils";

export default function TrailingStopsCard() {
  const [rows, setRows] = useState<ExecuteTrailingStopRow[]>([]);
  const [statusScope, setStatusScope] = useState<ExecuteStatusScope>("all");
  const [lsx, setLsx] = useState<ExecuteLsxFilter>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [hasError, setHasError] = useState(false);
  const hasLoadedOnceRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const normalizeRows = (payload: ExecuteTrailingStopsResponse | null) =>
      (payload?.rows ?? []).map((row) => ({
        ...row,
        status: normalizeTrailingStatus(row.status),
      }));

    const applyPayload = (payload: ExecuteTrailingStopsResponse | null) => {
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
      status: statusScope,
      limit: "200",
      q: "",
    });
    if (lsx !== "all") {
      params.set("lsx", lsx);
    }

    const loadFallback = async () => {
      setIsLoading(!hasLoadedOnceRef.current);
      const params = new URLSearchParams({
        status: statusScope,
        limit: "200",
        q: "",
      });
      if (lsx !== "all") {
        params.set("lsx", lsx);
      }

      const payload = await fetchJsonNoStore<ExecuteTrailingStopsResponse>(
        `/api/execute/trailing_stops?${params.toString()}&ts=${Date.now()}`
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
    const source = new EventSource(`/api/execute/trailing_stops/stream?${params.toString()}`);
    source.onmessage = (event) => {
      const payload = parseSseJson<ExecuteTrailingStopsResponse>(event.data);
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
  }, [lsx, statusScope]);

  const visibleRows = useMemo(
    () => filterTrailingClient(rows, "", statusScope, lsx),
    [lsx, rows, statusScope]
  );

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel jbravo-panel-emerald p-3 sm:p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="font-arimo text-[30px] font-semibold leading-none text-primary sm:text-[34px]">
          Trailing Stops
        </h2>
        <div className="flex w-full flex-wrap items-center justify-end gap-2 sm:w-auto">
          <button
            type="button"
            aria-label={`Trailing stop status scope ${statusScopeLabel(statusScope)}`}
            title={`Status: ${statusScopeLabel(statusScope)}`}
            onClick={() => setStatusScope((current) => cycleStatusScope(current))}
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
                key={`trailing-chip-${chip.key}`}
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
        <div className="min-w-[900px] max-h-[270px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-emerald">
          <table className="min-w-full table-auto">
            <thead className="sticky top-0 z-10 bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-700/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Symbol
                </th>
                <th scope="col" className="px-3 py-2 text-right whitespace-nowrap">
                  Qty
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Trail
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Stop Price
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Status
                </th>
                <th scope="col" className="px-3 py-2 text-left whitespace-nowrap">
                  Parent/Leg
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading && !hasLoadedOnce
                ? Array.from({ length: 5 }).map((_, index) => (
                    <tr key={`trailing-skeleton-${index}`} className="border-b border-slate-700/70">
                      <td colSpan={6} className="px-3 py-3">
                        <div className="h-4 w-full animate-pulse rounded bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && visibleRows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-3 py-8 text-center text-sm text-secondary">
                    No trailing stops available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? visibleRows.map((row, index) => {
                    const status = normalizeTrailingStatus(row.status);
                    return (
                      <tr
                        key={`${row.parent_leg ?? "leg"}-${row.symbol ?? "symbol"}-${index}`}
                        className="border-b border-slate-700/70 text-sm transition-colors hover:bg-slate-800/45"
                      >
                        <th scope="row" className="font-arimo px-3 py-2 text-left font-semibold text-sky-300 whitespace-nowrap">
                          {row.symbol || "--"}
                        </th>
                        <td className="font-cousine px-3 py-2 text-right tabular-nums text-slate-100 whitespace-nowrap">
                          {formatNumber(row.qty)}
                        </td>
                        <td className="font-cousine px-3 py-2 text-left tabular-nums text-slate-100 whitespace-nowrap">
                          {row.trail || "--"}
                        </td>
                        <td className="font-cousine px-3 py-2 text-left tabular-nums text-slate-100 whitespace-nowrap">
                          {formatCurrency(row.stop_price)}
                        </td>
                        <td className="px-3 py-2 text-left whitespace-nowrap">
                          <span
                            className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1 ${trailingStatusChipClass(
                              status
                            )}`}
                          >
                            {status}
                          </span>
                        </td>
                        <td className="font-cousine px-3 py-2 text-left text-[11px] text-slate-300 tabular-nums whitespace-nowrap">
                          {row.parent_leg || "--"}
                        </td>
                      </tr>
                    );
                  })
                : null}
            </tbody>
          </table>
        </div>
      </div>

      {hasError ? (
        <p className="mt-3 text-xs text-rose-300">Trailing stops endpoint unavailable. Showing fallback state.</p>
      ) : null}
    </section>
  );
}
