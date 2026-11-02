import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { fetchNoStoreJson, metricsMatchQuery } from "./utils";
import { metricRows, formatMetricWinRate } from "./metricsRows";
import { sourceKind } from "../ui/sourceState";
import type { MetricsResponse, ScreenerPicksResponse } from "./types";

const response = (payload: unknown) => new Response(JSON.stringify(payload), { status: 200 });
function payload(): MetricsResponse {
  const binding = { strategy_id: "trend_pullback_reclaim_qualifying_close_v2", strategy_version: "2",
    definition_sha256: "a".repeat(64), backtest_result_id: "b".repeat(64) };
  return { ok: true, ...binding, metrics_result_id: "c".repeat(64), metrics_status: "COMPLETE", rows: [{ ...binding,
    symbol: "SYN", signal_id: "d".repeat(64), trades: 200, wins: 1, losses: 199, win_rate_pct: 0.5,
    net_pnl: 12.34, expectancy: 0.0617, profit_factor: 1.02, max_drawdown_cash: 2.5, sample_status: "COMPLETED_SIGNALS" }] };
}
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("PR754 Metrics and financial dashboard source merge", () => {
  it("keeps pending Metrics explicit and never returns stale rows from HTTP200 okfalse", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ ...payload(), ok: false, status: "METRICS_PENDING" })));
    await expect(fetchNoStoreJson("/api/screener/metrics?filter=all&limit=50&ts=1"))
      .rejects.toThrow("Metrics are waiting for the strategy backtest.");
  });
  it.each(["/api/screener/picks", "/api/screener/metrics-other"])("retains fail-closed shared consumers: %s", async path => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ ok: false, status: "METRICS_PENDING", rows: [{ symbol: "STALE" }] })));
    await expect(fetchNoStoreJson(path)).rejects.toThrow("PostgreSQL data unavailable; no file fallback.");
  });
  it("unavailable Metrics remain unavailable rather than successful empty", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ ok: false, status: "ERROR", rows: [] })));
    await expect(fetchNoStoreJson("/api/screener/metrics?limit=50")).rejects.toThrow("PostgreSQL data unavailable");
  });
  it("preserves strategy binding, percent units, and symbol/strategy/history search", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload())));
    const data = await fetchNoStoreJson<MetricsResponse>("/api/screener/metrics?filter=with_history");
    const row = metricRows(data)[0];
    expect(formatMetricWinRate(row.win_rate_pct)).toBe("0.5%");
    expect(metricsMatchQuery(row, "syn")).toBe(true);
    expect(metricsMatchQuery(row, "pullback")).toBe(true);
    expect(metricsMatchQuery(row, "completed_signals")).toBe(true);
    expect(metricsMatchQuery(row, "different")).toBe(false);
  });
  it("research results cannot qualify held PostgreSQL ledger performance", () => {
    expect(metricRows(payload())[0].net_pnl).toBe(12.34);
    expect(sourceKind(payload())).toBe("ready");
    expect(sourceKind({ ok: true, source: "postgres", performance_status: "HELD_UNVALIDATED_PERFORMANCE", rows: [{ trade_id: 110 }] })).toBe("held");
    const context: ScreenerPicksResponse = { ok: true, run_date: "2026-10-23", completion_status: "verified", rows: [] };
    expect(context.completion_status).toBe("verified");
  });
  it("renders native Metrics table units and strategy context using the retained component", async () => {
    vi.resetModules(); let cursor = 0;
    const values: unknown[] = ["Pullback/Reclaim v2", metricRows(payload()), "", "", "all", false, null, 0];
    vi.doMock("react", async () => {
      const actual = await vi.importActual<typeof import("react")>("react");
      return { ...actual, useState: () => [values[cursor++], () => {}], useEffect: () => {},
        useMemo: (fn: () => unknown) => fn(), useRef: (value: unknown) => ({ current: value }) };
    });
    try {
      const { default: Card } = await import("./MetricsResultsCard");
      const html = renderToStaticMarkup(<Card/>);
      expect(html).toContain("Pullback/Reclaim v2"); expect(html).toContain("0.5%"); expect(html).not.toContain(">50%");
      expect(html).toContain("$12.34"); expect(html).toContain("$2.50"); expect(html).toContain("Drawdown (USD)");
      expect(html).toContain("Current signals have no future result yet."); expect(html).toContain("exact strategy backtest");
      expect(html).not.toContain("No Screener picks in this batch.");
    } finally { vi.doUnmock("react"); vi.resetModules(); }
  });
});
