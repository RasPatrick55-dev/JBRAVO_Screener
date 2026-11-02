import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { sourceKind, finiteNumber, canDisplay } from "./sourceState";
import SourceNotice from "./SourceNotice";
import { marketPhase } from "../header/MarketClockGroup";
import { normalizeStatsPayload } from "../trades/TradesTab";
import { normalizeLeaderboardPayload } from "../trades/leaderboardData";
import { toPositionRow } from "../../screens/PositionsMonitoring";
import PositionsTab from "../positions/PositionsTab";

describe("business source state", () => {
  it("distinguishes held HTTP-200 data from successful empty and unavailable", () => {
    expect(sourceKind({ ok: false, performance_status: "HELD_UNVALIDATED_PERFORMANCE", rows: [] })).toBe("held");
    expect(sourceKind({ ok: true, source: "postgres", rows: [] })).toBe("empty");
    expect(sourceKind({ ok: false, rows: [] })).toBe("unavailable");
    expect(sourceKind(null)).toBe("unavailable");
    expect(sourceKind({ ok: false, performance_status: "HELD_UNVALIDATED_PERFORMANCE", policy_validation_status: "UNAVAILABLE" })).toBe("unavailable");
  });
  it("never retains ready analytics as apparently validated after held response", () => {
    const previous = { ok: true, rows: [{ totalPL: 999 }] };
    expect(canDisplay(sourceKind(previous))).toBe(true);
    const current = { ok: false, performance_status: "HELD_UNVALIDATED_PERFORMANCE", rows: [] };
    expect(canDisplay(sourceKind(current))).toBe(false);
    expect(normalizeStatsPayload(current)).toEqual([]);
    const html = renderToStaticMarkup(<SourceNotice label="Validated performance" kind="held" payload={current} />);
    expect(html).toContain("HELD");
    expect(html).not.toContain("999");
  });
  it("does not convert a one-percent win rate into 100 percent", () => {
    const row = { key: "all", label: "ALL", winRatePct: 1, totalPL: 0, tradesCount: 100,
      topTrade: { symbol: "AAA", pl: 0 }, worstLoss: { symbol: "BBB", pl: -1 } };
    expect(normalizeStatsPayload({ ok: true, rows: [row] })[0].winRatePct).toBe(1);
  });
  it("preserves financial zero, fractional shares and unknowns", () => {
    expect(finiteNumber(0)).toBe(0);
    for (const value of [null, undefined, "", NaN, Infinity, true]) expect(finiteNumber(value)).toBeNull();
    const row = toPositionRow({ symbol: "AAA", qty: "0.125", entryPrice: 12, currentPrice: null, dollarPL: null });
    expect(row.shares).toBe(0.125);
    expect(row.currentPrice).toBeNull();
    expect(row.openPL).toBeNull();
    expect(row.trailingStop).toBeNull();
    expect(normalizeLeaderboardPayload({ rows: [{ symbol: "AAA", pl: null }] })).toEqual([]);
  });
  it("does not show zero exposure or empty broker confirmation on failed load", () => {
    const html = renderToStaticMarkup(<PositionsTab positions={[]} logs={[]} available={false} hasError
      logsAvailable={false} summary={{ totalShares: null, totalOpenPL: null, avgDaysHeld: null, totalCapturedPL: null }} />);
    expect(html).toContain("Position count and exposure are unknown");
    expect(html).not.toContain("Total Summary");
    expect(html).not.toContain("0 open broker positions loaded");
    expect(html).not.toContain("$0.00");
  });
  it("a logs-only failure does not hide known broker positions or their marks", () => {
    const html = renderToStaticMarkup(<PositionsTab positions={[toPositionRow({ symbol: "AAA", qty: 1.25, currentPrice: 13.5, dollarPL: 2 })]}
      logs={[]} logsAvailable={false} summary={{ totalShares: 1.25, totalOpenPL: 2, avgDaysHeld: null, totalCapturedPL: null }} />);
    expect(html).toContain("AAA"); expect(html).toContain("$13.50");
    expect(html).toContain("Monitoring logs unavailable");
  });
});

describe("authoritative market clock", () => {
  it.each(["2026-10-25T15:00:00Z", "2026-12-25T16:00:00Z", "2026-11-27T18:01:00Z"])("uses exchange closed for %s", stamp => {
    expect(marketPhase({ status: "ready", is_open: false, observed_at: stamp }, Date.parse(stamp))).toBe("Closed");
  });
  it("uses a fresh exchange open after DST and never stale wall time", () => {
    const stamp = "2026-03-09T13:30:00Z";
    expect(marketPhase({ status: "ready", is_open: true, observed_at: stamp }, Date.parse(stamp))).toBe("Market Open");
    expect(marketPhase({ status: "ready", is_open: true, observed_at: stamp }, Date.parse(stamp) + 61000)).toBe("Stale");
    expect(marketPhase(null)).toBe("Unknown");
    expect(marketPhase({ status: "ready", is_open: true })).toBe("Unknown");
  });
});
