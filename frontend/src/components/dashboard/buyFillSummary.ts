export type ExecuteOrdersSummaryResponse = {
  ok?: boolean;
  orders_filled?: number | null;
  total_value?: number | null;
  since_utc?: string | null;
  until_utc?: string | null;
  source?: string | null;
  value_unit?: string | null;
  coverage?: string | null;
};

// Count and notional must describe the same complete broker window. Never use
// last-run submissions as a fallback for trailing-24-hour fills.
export function buyFillSummary(summary: ExecuteOrdersSummaryResponse | null) {
  const since = Date.parse(summary?.since_utc ?? "");
  const until = Date.parse(summary?.until_utc ?? "");
  const valid = summary?.ok === true && summary.source === "alpaca"
    && summary.value_unit === "USD_buy_fill_notional" && summary.coverage === "requested_24h_window"
    && Number.isFinite(since) && Number.isFinite(until) && until - since === 86_400_000;
  return {
    count: valid && typeof summary.orders_filled === "number" && Number.isInteger(summary.orders_filled) && summary.orders_filled >= 0
      ? summary.orders_filled : Number.NaN,
    value: valid && typeof summary.total_value === "number" && Number.isFinite(summary.total_value) && summary.total_value >= 0
      ? summary.total_value : Number.NaN,
  };
}
