import { describe, expect, it } from "vitest";
import { formatMetricWinRate, metricRows } from "./metricsRows";
import type { MetricsResponse, MetricsRow } from "./types";
function response(): MetricsResponse {
  const contract = { strategy_id: "trend_pullback_reclaim_qualifying_close_v2", strategy_version: "2", definition_sha256: "a".repeat(64), backtest_result_id: "b".repeat(64) };
  return { ok: true, ...contract, metrics_result_id: "c".repeat(64), rows: [{ ...contract, symbol: "SRRK", signal_id: "d".repeat(64), trades: 0, wins: 0, losses: 0, win_rate_pct: null, net_pnl: 0, expectancy: null, profit_factor: null, max_drawdown_cash: 0, sample_status: "NO_COMPLETED_SIGNALS" }] };
}
describe("strategy metric results", () => {
  it("keeps percentage points below one percent in their declared units", () => {
    expect(formatMetricWinRate(0.5)).toBe("0.5%");
    expect(formatMetricWinRate(1)).toBe("1%");
    expect(formatMetricWinRate(null)).toBe("--");
  });
  it("preserves unknown statistics instead of inventing zero win rates", () => {
    const row = metricRows(response())[0]; expect(row.trades).toBe(0); expect(row.win_rate_pct).toBeNull(); expect(row.expectancy).toBeNull(); expect(row.profit_factor).toBeNull();
  });
  it.each(["strategy_id", "strategy_version", "definition_sha256", "backtest_result_id"] as const)("rejects a row from a different %s", key => {
    const payload = response(); (payload.rows![0] as MetricsRow)[key] = "wrong"; expect(() => metricRows(payload)).toThrow("do not match");
  });
  it("shows pending metrics as unavailable even when stale rows accompany it", () => {
    expect(() => metricRows({ ...response(), ok: false, status: "METRICS_PENDING" })).toThrow("waiting");
  });
  it("accepts a strategy-bound empty batch", () => {
    expect(metricRows({ ...response(), metrics_status: "SKIPPED_NO_CANDIDATES", rows: [] })).toEqual([]);
  });
});
