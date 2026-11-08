import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import fixture from "./strategyBacktestFixture.json";
import { backtestRows } from "../src/components/screener/backtestRows";
import { BACKTEST_PATH, BacktestResultsContent, backtestPresentation } from "../src/components/screener/BacktestResultsCard";
import { evidenceNumber, evidenceCount } from "../src/components/screener/strategyEvidence";
import type { SourceKind } from "../src/components/ui/sourceState";

const data = () => structuredClone(fixture) as Record<string, any>;
const html = (payload: Record<string, unknown> | null, kind: SourceKind = "ready") => renderToStaticMarkup(<BacktestResultsContent result={{ payload, kind }}/>);

describe("strategy evidence numeric parsing", () => {
  it.each([null, undefined, "", " ", true, false, [], [0], {}, "0x10", "Infinity", Infinity, NaN])("does not coerce %j into a financial value", value => expect(evidenceNumber(value)).toBeNull());
  it.each([[0, 0], ["0", 0], [-12.5, -12.5], [" 1.25 ", 1.25], ["1e-3", 0.001]])("preserves finite numeric evidence %j", (value, expected) => expect(evidenceNumber(value)).toBe(expected));
  it.each([-1, 1.5, "1.5", true, " ", Number.MAX_SAFE_INTEGER + 1])("does not accept %j as a completed count", value => expect(evidenceCount(value)).toBeNull());
});

describe("Backtest card current strategy API contract", () => {
  it("maps the saved public LI/HUM response without invented history windows or performance", () => {
    const rows = backtestRows(data());
    expect(rows.map(row => [row.symbol, row.strategy_id, row.strategy_version, row.closed_signals])).toEqual([
      ["LI", "double_bottom_breakout_v1", "1", 0], ["HUM", "trend_pullback_reclaim_qualifying_close_v2", "2", 0],
    ]);
    for (const row of rows) {
      expect(row.net_pnl).toBe(0); expect(row.max_drawdown_cash).toBe(0);
      expect(row.win_rate_pct).toBeNull(); expect(row.expectancy).toBeNull(); expect(row.profit_factor).toBeNull();
      expect(row.definition_sha256).toBe(""); // Unified definition is not the row's strategy definition.
    }
    const markup = html(data());
    for (const text of ["Candidate-symbol history only", "double bottom breakout v1", "Pullback/Reclaim v2", "History window: not supplied", "Drawdown (USD)", "Net P&amp;L (USD)", "Expectancy (USD)", "Profit factor", "Insufficient evidence: 0 completed signals", "Current proposals have no future result yet", "zero-sample total"]) expect(markup).toContain(text);
    expect(markup).not.toContain("6M"); expect(BACKTEST_PATH).not.toContain("window=");
    for (const legacy of ["Avg Return", "P/L Ratio", "Max DD", "75%", ">0%<"]) expect(markup).not.toContain(legacy);
  });
  it("maps nonzero canonical fields rather than legacy aliases, preserves sub-one percent and ratio precision", () => {
    const payload = data(); payload.rows = [payload.rows[0]];
    Object.assign(payload.rows[0], { closed_signals: "200", winning_signals: "1", losing_signals: "199", win_rate_pct: "0.5",
      closed_trade_net_pnl: "12.34", expectancy_net_per_closed_signal: "0.0617", max_closed_signal_drawdown_cash: "-9.25",
      profit_factor: "1.02", profit_factor_status: "DEFINED", sample_status: "HAS_COMPLETED_SIGNALS",
      net_pnl: 888, total_pl_usd: 999, pl_ratio: 888, avg_return_pct: 999, max_dd_pct: 999, trades: 999 });
    const row = backtestRows(payload)[0], markup = html(payload);
    expect(row.net_pnl).toBe(12.34); expect(row.closed_signals).toBe(200); expect(row.expectancy).toBe(0.0617);
    for (const text of ["0.5%", "+$12.34", "+$0.06", "-$9.25", "1.02"]) expect(markup).toContain(text);
    for (const text of ["50%", "999", "888", "-9.25%"]) expect(markup).not.toContain(text);
  });
  it("keeps missing and malformed numeric values unknown, including a missing canonical field with a legacy zero", () => {
    const payload = data(); payload.rows = [payload.rows[0]];
    Object.assign(payload.rows[0], { closed_signals: true, winning_signals: [], closed_trade_net_pnl: " ", net_pnl: 0,
      expectancy_net_per_closed_signal: false, max_closed_signal_drawdown_cash: {}, fees_closed_signals: null });
    const row = backtestRows(payload)[0];
    for (const key of ["closed_signals", "wins", "net_pnl", "expectancy", "max_drawdown_cash", "fees"] as const) expect(row[key]).toBeNull();
  });
  it("requires declared dollar drawdown units and never substitutes generic max_drawdown", () => {
    const payload = data(); payload.rows[0].drawdown_units = "PERCENT"; payload.rows[0].max_drawdown = 20;
    expect(backtestRows(payload)[0].max_drawdown_cash).toBeNull();
  });
  it("does not invent all-win infinity or force no-sample rates to zero", () => {
    const payload = data(); payload.rows = [payload.rows[0]];
    Object.assign(payload.rows[0], { closed_signals: 2, winning_signals: 2, profit_factor: null, profit_factor_status: "UNDEFINED_NO_LOSSES" });
    expect(html(payload)).toContain("No losses; undefined"); expect(html(payload)).not.toContain("Infinity");
    payload.rows[0].closed_signals = 0; payload.rows[0].win_rate_pct = 99; payload.rows[0].expectancy_net_per_closed_signal = 99;
    expect(backtestRows(payload)[0].win_rate_pct).toBeNull(); expect(backtestRows(payload)[0].expectancy).toBeNull();
  });
  it("supports the bound single-strategy shape without borrowing an invented report identity", () => {
    const payload = data(), row = payload.rows[0];
    Object.assign(payload, { rows: [row], strategy_id: row.strategy_id, strategy_version: row.strategy_version,
      backtest_result_id: row.backtest_result_id, window: "STRATEGY_EPISODE", evaluation_end: "2026-11-06" });
    delete row.backtest_result_id;
    expect(backtestRows(payload)[0].backtest_result_id).toBe(payload.backtest_result_id);
    expect(html(payload)).toContain("Window: strategy episode."); expect(html(payload)).toContain("Evaluation end: 2026-11-06");
  });
  it.each(["strategy_id", "strategy_version", "current_signal_id", "signal_id", "backtest_result_id"])("fails closed when a row identity is missing: %s", key => {
    const payload = data(); delete payload.rows[0][key]; expect(() => backtestRows(payload)).toThrow();
  });
  it("does not combine duplicate current signals or reused report identities across strategies", () => {
    const payload = data(); payload.rows.push(payload.rows[0]); expect(() => backtestRows(payload)).toThrow();
    const reused = data(); reused.rows[1].backtest_result_id = reused.rows[0].backtest_result_id; expect(() => backtestRows(reused)).toThrow();
  });
  it.each(["held", "unavailable", "stale"] as const)("suppresses old rows for %s instead of labeling it a successful empty", kind => {
    const markup = html(data(), kind); expect(markup).toContain('role="alert"'); expect(markup).not.toContain(">LI<");
    expect(markup).not.toContain("No Screener picks in this batch."); expect(markup).not.toContain("verified complete");
    expect(backtestPresentation({payload: data(), kind}).sync).toBe("error");
  });
  it("separates loading, empty, transport failure, and invalid contract", () => {
    expect(html(data(), "loading")).toContain("Loading strategy backtest"); expect(html(data(), "loading")).not.toContain(">LI<");
    const empty = data(); empty.rows = []; expect(html(empty, "empty")).toContain("No Screener picks in this batch.");
    const bad = data(); bad.metric_basis = "ACTUAL_PAPER"; expect(html(bad)).toContain('role="alert"'); expect(html(bad)).not.toContain(">LI<");
    expect(html(null, "unavailable")).toContain("Backtest unavailable");
  });
  it("keeps the established card theme and local horizontal scrolling at narrow widths", () => {
    const markup = html(data());
    for (const token of ["jbravo-panel-violet", "jbravo-panel-inner-violet", "min-w-0", "overflow-x-auto", "sm:p-md", "sm:max-h-[360px]", "dark:bg-slate-900/95", "break-all"]) expect(markup).toContain(token);
    expect(markup).not.toContain('<style');
  });
});
