import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import * as sources from "../src/components/ui/sourceState";
import { executionGrossReadPolicy } from "../src/components/trades/recordedGrossData";
import { strategyHistoryReadPolicy } from "../src/components/trades/strategyHistory";
import StrategyHistoryCard from "../src/components/trades/StrategyHistoryCard";
import TradesTab from "../src/components/trades/TradesTab";
import fixture from "./recordedGrossFixture.json";
import type { SourcePayload, SourceResult } from "../src/components/ui/sourceState";

const NOW = "2026-11-09T17:00:00.000Z";
const settle = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const good = (): SourcePayload => ({...structuredClone(fixture.winners), ok: true, status: "partial",
  view_scope: "EXECUTION_RECONSTRUCTED_GROSS", pnl_basis: "POSTGRES_MATCHED_EXECUTION_GROSS_V1", performance_status: "EXECUTION_GROSS_ONLY",
  fees: null, net_pnl: null, accounting_complete: false, entry_ready: false,
  execution_fingerprint: "a".repeat(64), policy_fingerprint: "b".repeat(64), input_fingerprint: "c".repeat(64),
  observed_at: NOW, source_as_of: "2026-10-27T21:00:00Z", source_freshness: "STORED_EXECUTIONS_NOT_BROKER_COMPLETENESS_PROOF",
  coverage: {included: 5, excluded: 1, unknown: 1, reasons: {QUANTITY_MISMATCH: 1}}, closed_population_count: 7});
const unavailable = (): SourcePayload => ({ok: false, status: "unavailable", source: "postgres", view_scope: "EXECUTION_RECONSTRUCTED_GROSS",
  pnl_basis: "POSTGRES_MATCHED_EXECUTION_GROSS_V1", reason: "CURRENT_EXECUTION_GROSS_INPUTS_UNAVAILABLE", rows: [], coverage: null,
  fees_status: "UNKNOWN", net_status: "UNKNOWN", net_eligible: false, accounting_complete: false, recovery_complete: false, entry_ready: false});
const response = (payload: unknown) => new Response(JSON.stringify(payload), {status: 200});
const source = (payload: SourcePayload): SourceResult => ({payload, kind: sources.sourceKind(payload), validity: sources.sourceValidity(payload), freshness: "current", receivedAt: NOW});
afterEach(() => {vi.restoreAllMocks(); vi.useRealTimers();});

describe("opt-in gross unavailable retention", () => {
  it("preserves a dated partial sample without claiming the failed response succeeded, then recovers", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW));
    const first = good(), recovered = {...good(), observed_at: "2026-11-09T17:00:40.000Z"};
    const transport = vi.fn().mockResolvedValueOnce(response(first)).mockResolvedValueOnce(response(unavailable())).mockResolvedValueOnce(response(recovered));
    let result!: SourceResult;
    const stop = sources.watchSource("/gross", value => {result = value;}, 20000, transport, 60000, 45000, executionGrossReadPolicy);
    try {
      await settle(); expect(result.kind).toBe("partial"); expect(result.receivedAt).toBe(NOW);
      await vi.advanceTimersByTimeAsync(20000);
      expect(result.kind).toBe("stale"); expect(result.validity).toBe("partial"); expect(result.freshness).toBe("stale");
      expect(result.payload).toEqual(first); expect(result.refreshStatus).toBe("unavailable");
      expect(result.receivedAt).toBe(NOW); expect(result.lastAttemptAt).toBe("2026-11-09T17:00:20.000Z");
      vi.spyOn(sources, "useSource").mockImplementation(path => path.includes("/leaderboard") ? result :
        source({ok: false, status: "unavailable", rows: []}));
      const html = renderToStaticMarkup(<TradesTab/>);
      expect(html).toContain("STALE (stale read)"); expect(html).toContain("Current source unavailable");
      expect(html).toContain("previous observation, not current performance"); expect(html).toContain("1 unknown");
      expect(html).toContain(NOW); expect(html).toContain("Before fees"); expect(html).not.toContain("No leaderboard trades yet.");
      await vi.advanceTimersByTimeAsync(20000);
      expect(result.kind).toBe("partial"); expect(result.freshness).toBe("current"); expect(result.refreshStatus).toBeUndefined();
      expect(result.receivedAt).toBe("2026-11-09T17:00:40.000Z"); expect(result.payload).toEqual(recovered);
    } finally {stop();}
  });

  it("initial unavailable has unknown count and no successful read timestamp", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW)); let result!: SourceResult;
    const stop = sources.watchSource("/gross", value => {result = value;}, 20000,
      vi.fn().mockResolvedValue(response(unavailable())), 60000, 45000, executionGrossReadPolicy);
    try {await settle(); expect(result.kind).toBe("unavailable"); expect(result.payload?.coverage).toBeNull();
      expect(result.receivedAt).toBeUndefined(); expect(result.lastAttemptAt).toBe(NOW);
    } finally {stop();}
  });

  it.each(["identity", "policy", "wrong-basis", "malformed", "null", "future", "other-reason"])("does not retain previous values for %s failure", async damage => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW));
    let next: unknown = unavailable();
    if (damage === "identity") next = {...unavailable(), broker_account_match: "mismatch"};
    if (damage === "policy") next = {...unavailable(), performance_status: "HELD_UNVALIDATED_PERFORMANCE", policy_validation_status: "VALIDATED_POLICY"};
    if (damage === "wrong-basis") next = {...good(), pnl_basis: "VALIDATED_NET"};
    if (damage === "malformed") next = {ok: true, rows: "broken"};
    if (damage === "null") next = null;
    if (damage === "future") next = {...good(), observed_at: "2027-01-01T00:00:00Z"};
    if (damage === "other-reason") next = {...unavailable(), reason: "ACCOUNT_IDENTITY_CHANGED"};
    const transport = vi.fn().mockResolvedValueOnce(response(good())).mockResolvedValueOnce(response(next));
    let result!: SourceResult;
    const stop = sources.watchSource("/gross", value => {result = value;}, 20000, transport, 60000, 45000, executionGrossReadPolicy);
    try {await settle(); await vi.advanceTimersByTimeAsync(20000);
      expect(result.kind).toBe(damage === "policy" ? "held" : "unavailable");
      expect(result.payload).toEqual(next); expect(result.receivedAt).toBe(NOW);
      expect(sources.canDisplay(result.kind)).toBe(false);
    } finally {stop();}
  });

  it("does not change ordinary sources that did not opt in", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW)); let result!: SourceResult;
    const transport = vi.fn().mockResolvedValueOnce(response(good())).mockResolvedValueOnce(response(unavailable()));
    const stop = sources.watchSource("/ordinary", value => {result = value;}, 20000, transport);
    try {await settle(); await vi.advanceTimersByTimeAsync(20000);
      expect(result.kind).toBe("unavailable"); expect(result.payload).toEqual(unavailable());
      expect(result.refreshStatus).toBeUndefined();
    } finally {stop();}
  });

  it("does not carry a successful sample to a new request observer", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW)); let first!: SourceResult, second!: SourceResult;
    const stopFirst = sources.watchSource("/gross?range=all", value => {first = value;}, 20000,
      vi.fn().mockResolvedValue(response(good())), 60000, 45000, executionGrossReadPolicy);
    await settle(); expect(first.kind).toBe("partial"); stopFirst();
    const stopSecond = sources.watchSource("/gross?range=d", value => {second = value;}, 20000,
      vi.fn().mockResolvedValue(response(unavailable())), 60000, 45000, executionGrossReadPolicy);
    try {await settle(); expect(second.kind).toBe("unavailable"); expect(second.receivedAt).toBeUndefined();
      expect(second.payload?.rows).toEqual([]);
    } finally {stopSecond();}
  });
});

describe("strategy successful read label", () => {
  const strategy = (): SourcePayload => ({ok: true, status: "empty", rows: [], schema_version: "jbravo.strategy-history.v1",
    source: "postgres_strategy_execution", trade_view: "STRATEGY_EXECUTION_HISTORY", evidence_basis: "STORED_BROKER_ORDER_OBSERVATIONS",
    performance_status: "NOT_QUALIFIED_STRATEGY_HISTORY", accounting_complete: false, fees_status: "UNKNOWN_UNALLOCATED", net_status: "UNKNOWN",
    account_scope: "configured_paper_broker", broker_account_match: "verified", freshness: "stored_observations_not_live", observed_at: NOW});
  it.each([false, true])("unavailable response does not advance successful time; preceding success=%s", async preceding => {
    vi.useFakeTimers(); vi.setSystemTime(new Date(NOW)); let result!: SourceResult;
    const failed = {ok: false, status: "unavailable", source: "postgres_strategy_execution", rows: []};
    const transport = vi.fn(); if (preceding) transport.mockResolvedValueOnce(response(strategy()));
    transport.mockResolvedValue(response(failed));
    const stop = sources.watchSource("/strategy", value => {result = value;}, 20000, transport, 60000, 45000, strategyHistoryReadPolicy);
    try {await settle(); if (preceding) await vi.advanceTimersByTimeAsync(20000);
      expect(result.kind).toBe("unavailable"); expect(result.receivedAt).toBe(preceding ? NOW : undefined);
      const html = renderToStaticMarkup(<StrategyHistoryCard result={result}/>);
      expect(html).toContain("Last successful read: "+(preceding ? "2026-11-09 17:00:00 UTC" : "Unknown"));
      expect(html).toContain("Last attempt:"); expect(html).toContain("A response is not a successful data read.");
      expect(html).toContain("strategy count and financial results are unknown"); expect(html).not.toContain("$0.00");
    } finally {stop();}
  });
});
