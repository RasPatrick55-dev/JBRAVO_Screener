import { useMemo } from "react";
import type { ExecuteOrderRow } from "./types";
import {
  formatCurrency,
  formatDateTimeUtc,
  formatNumber,
  normalizeOrderStatus,
  normalizeSide,
  orderStatusChipClass,
  sideChipClass,
} from "./utils";

type Props = {
  rows: ExecuteOrderRow[];
  isLoading: boolean;
  hasError: boolean;
};

export default function OrdersTableCard({ rows, isLoading, hasError }: Props) {
  const visibleRows = useMemo(() => {
    const toEpochMs = (value: string | null | undefined): number | null => {
      if (!value) {
        return null;
      }
      const parsed = Date.parse(value);
      return Number.isFinite(parsed) ? parsed : null;
    };

    return [...rows].sort((left, right) => {
      const leftTs = toEpochMs(left.ts_utc);
      const rightTs = toEpochMs(right.ts_utc);
      if (leftTs !== null && rightTs !== null) {
        return rightTs - leftTs;
      }
      if (rightTs !== null) {
        return 1;
      }
      if (leftTs !== null) {
        return -1;
      }
      return 0;
    });
  }, [rows]);

  return (
    <section className="overflow-hidden rounded-2xl outline-subtle shadow-card jbravo-panel jbravo-panel-violet p-3 sm:p-5">
      <header className="flex items-start justify-between gap-3">
        <h2 className="font-arimo text-[30px] font-semibold leading-none text-primary sm:text-[34px]">Orders</h2>
      </header>

      <div className="mt-3 overflow-x-auto">
        <div className="min-w-[980px] max-h-[320px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-violet">
          <table className="min-w-full table-fixed">
            <colgroup>
              <col className="w-[112px]" />
              <col className="w-[84px]" />
              <col className="w-[74px]" />
              <col className="w-[86px]" />
              <col className="w-[58px]" />
              <col className="w-[112px]" />
              <col className="w-[92px]" />
              <col className="w-[86px]" />
              <col className="w-[276px]" />
            </colgroup>
            <thead className="sticky top-0 z-10 bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-700/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Time (UTC)
                </th>
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Symbol
                </th>
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Side
                </th>
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Type
                </th>
                <th scope="col" className="px-1.5 py-1 text-right whitespace-nowrap">
                  Qty
                </th>
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Limit/Stop/Trail
                </th>
                <th scope="col" className="px-1.5 py-1 text-left whitespace-nowrap">
                  Status
                </th>
                <th scope="col" className="px-1.5 py-1 text-right whitespace-nowrap">
                  Filled Avg
                </th>
                <th scope="col" className="px-2 py-1 text-left whitespace-nowrap">
                  Notes
                </th>
              </tr>
            </thead>
            <tbody>
              {isLoading && rows.length === 0
                ? Array.from({ length: 6 }).map((_, index) => (
                    <tr key={`orders-skeleton-${index}`} className="border-b border-slate-700/70">
                      <td colSpan={9} className="px-2 py-2">
                        <div className="h-4 w-full animate-pulse rounded bg-slate-700/70" />
                      </td>
                    </tr>
                  ))
                : null}

              {!isLoading && visibleRows.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-3 py-8 text-center text-sm text-secondary">
                    No orders available.
                  </td>
                </tr>
              ) : null}

              {!isLoading
                ? visibleRows.map((row, index) => {
                    const side = normalizeSide(row.side);
                    const status = normalizeOrderStatus(row.status);
                    const ts = formatDateTimeUtc(row.ts_utc ?? null);
                    return (
                      <tr
                        key={`${row.order_id ?? "order"}-${row.ts_utc ?? "ts"}-${index}`}
                        className="border-b border-slate-700/70 text-sm transition-colors hover:bg-slate-800/45"
                      >
                        <td className="font-cousine px-1.5 py-1 text-left tabular-nums text-slate-300 whitespace-nowrap">
                          <div className="flex flex-col whitespace-nowrap leading-tight">
                            <span>{ts.time}</span>
                            <span className="text-[10px] text-slate-400">{ts.date}</span>
                          </div>
                        </td>
                        <th scope="row" className="font-arimo px-1.5 py-1 text-left font-semibold text-sky-300 whitespace-nowrap">
                          {row.symbol || "--"}
                        </th>
                        <td className="px-1.5 py-1 text-left whitespace-nowrap">
                          <span
                            className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1 ${sideChipClass(
                              side
                            )}`}
                          >
                            {side || "--"}
                          </span>
                        </td>
                        <td className="font-cousine px-1.5 py-1 text-left text-slate-100 whitespace-nowrap">
                          {row.type || "--"}
                        </td>
                        <td className="font-cousine px-1.5 py-1 text-right tabular-nums text-slate-100 whitespace-nowrap">
                          {formatNumber(row.qty)}
                        </td>
                        <td className="font-cousine px-1.5 py-1 text-left tabular-nums text-slate-200 whitespace-nowrap">
                          {row.limit_stop_trail || "--"}
                        </td>
                        <td className="px-1.5 py-1 text-left whitespace-nowrap">
                          <span
                            className={`inline-flex rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase outline outline-1 ${orderStatusChipClass(
                              status
                            )}`}
                          >
                            {status}
                          </span>
                        </td>
                        <td className="font-cousine px-1.5 py-1 text-right tabular-nums text-slate-100 whitespace-nowrap">
                          {formatCurrency(row.filled_avg)}
                        </td>
                        <td className="px-2 py-1 text-left text-slate-300 break-words leading-tight">
                          {row.notes || ""}
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
        <p className="mt-3 text-xs text-rose-300">Orders endpoint unavailable. Showing fallback state.</p>
      ) : null}
    </section>
  );
}
