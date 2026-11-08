export type ExecuteOrdersSummaryResponse = {
  ok?: boolean;
  status?: string | null;
  fill_count?: number | null;
  count_unit?: string | null;
  observed_fill_count?: number | null;
  total_value?: number | null;
  since_utc?: string | null;
  until_utc?: string | null;
  source?: string | null;
  source_endpoint?: string | null;
  account_scope?: string | null;
  value_unit?: string | null;
  coverage?: string | null;
  coverage_scope?: string | null;
  window_basis?: string | null;
  window_boundary?: string | null;
  accounting_final?: boolean;
};

// Count executions (including partial fills), not unique or completed orders.
// Both totals require the same exhausted observed fill-activity window. Legacy
// submitted-time order queries and incomplete lower bounds are not substitutes.
export function buyFillSummary(summary: ExecuteOrdersSummaryResponse | null) {
  const since = Date.parse(summary?.since_utc ?? "");
  const until = Date.parse(summary?.until_utc ?? "");
  const valid = summary?.ok === true && summary.source === "alpaca"
    && summary.source_endpoint === "/v2/account/activities/FILL"
    && summary.account_scope === "configured_paper_broker"
    && summary.count_unit === "buy_fill_activity_events"
    && summary.value_unit === "USD_buy_fill_notional"
    && summary.coverage === "requested_24h_fill_activity_window"
    && summary.coverage_scope === "observed_broker_fill_activities"
    && summary.window_basis === "transaction_time"
    && summary.window_boundary === "since_inclusive_until_exclusive"
    && (summary.status === "ready" && (summary.fill_count ?? 0) > 0
      || summary.status === "empty" && summary.fill_count === 0 && summary.total_value === 0)
    && Number.isFinite(since) && Number.isFinite(until) && until - since === 86_400_000
    && typeof summary.fill_count === "number" && Number.isInteger(summary.fill_count) && summary.fill_count >= 0
    && typeof summary.total_value === "number" && Number.isFinite(summary.total_value) && summary.total_value >= 0;
  return {
    count: valid ? summary.fill_count! : Number.NaN,
    value: valid ? summary.total_value! : Number.NaN,
  };
}
