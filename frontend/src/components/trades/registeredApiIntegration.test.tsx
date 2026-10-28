import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import fixture from "./registeredApiFixture.json";
import { physicalHistoryRows, normalizeStatsPayload } from "./TradesTab";
import LatestTradesTable from "./LatestTradesTable";
import type { SourceResult, SourcePayload } from "../ui/sourceState";

function responses() {
  const history: SourceResult = {kind:"ready",validity:"ready",freshness:"current",receivedAt:fixture.received_at,
    payload:structuredClone(fixture.history) as SourcePayload};
  const stats: SourceResult = {kind:"held",validity:"held",freshness:"current",receivedAt:fixture.received_at,
    payload:structuredClone(fixture.stats) as SourcePayload};
  return {history,stats};
}

describe("actual backend wire -> original V11 table",()=>{
  it("shows scoped registered costs with absent executor override while validated performance remains held",()=>{
    const {history,stats}=responses();const rows=physicalHistoryRows(history,stats);
    expect(rows).toHaveLength(1);expect(rows[0].derivedObservedNet?.fees).toBe("0.01");
    expect(rows[0].derivedObservedNet?.net).toBe("-63.39");expect(rows[0].grossPL).toBe("-63.38");
    expect(rows[0].totalPL).toBeNull();expect(rows[0].priceSold).toBeNull();
    expect(stats.payload?.performance_status).toBe("HELD_UNVALIDATED_PERFORMANCE");
    expect(stats.payload?.policy_validation_status).toBe("VALIDATED_POLICY");
    expect(normalizeStatsPayload(stats.payload)).toEqual([]);
    const html=renderToStaticMarkup(<LatestTradesTable rows={rows} plLabel="Recorded P/L"/>);
    expect(html).toContain("observed-cost net -$63.39");expect(html).toContain("Allocated fees $0.01");
    expect(html).toContain("jbravo-panel-emerald");expect(html).not.toContain("Audited fill calculations");
  });
  it.each(["stats_unavailable","stats_stale","history_stale","revoked","row_drift","wrong_scope"])("fails closed on %s without losing physical history",failure=>{
    const {history,stats}=responses();
    if(failure==="stats_unavailable")stats.payload!.policy_validation_status="UNAVAILABLE";
    if(failure==="stats_stale")stats.freshness="stale";
    if(failure==="history_stale")history.freshness="stale";
    const context=history.payload!.derived_read_context as Record<string,unknown>;
    if(failure==="revoked")context.registration_status="REVOKED";
    if(failure==="row_drift")(history.payload!.rows as Record<string,unknown>[])[0].qty=31;
    if(failure==="wrong_scope")((stats.payload!.trade_policy as Record<string,unknown>).policy as Record<string,unknown>).scope="ACTIVE_RECONCILIATION";
    const rows=physicalHistoryRows(history,stats);expect(rows).toHaveLength(1);expect(rows[0].derivedObservedNet).toBeNull();
  });
});
