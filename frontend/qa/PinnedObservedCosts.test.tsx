import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import fixture from "../src/components/trades/registeredApiFixture.json";
import { physicalHistoryRows, normalizeStatsPayload } from "../src/components/trades/TradesTab";
import LatestTradesTable from "../src/components/trades/LatestTradesTable";
import type { SourcePayload, SourceResult } from "../src/components/ui/sourceState";

function responses() {
  const payload = structuredClone(fixture.history) as unknown as SourcePayload;
  const row = (payload.rows as Record<string, unknown>[])[0];
  const data = row.derived_accounting as Record<string, unknown>;
  const context = payload.derived_read_context as Record<string, unknown>;
  data.status = "APPROVED_SCOPED_OBSERVED_DERIVATION";
  data.source = "postgres_allocation_pinned_observed_costs";
  data.canonical_account_uuid_proved = false; data.global_accounting_complete = false; data.entry_ready = false;
  context.source = "postgres_current_observed_cost_read"; context.evidence_status = "PINNED_OWNER_APPROVED";
  delete context.registration_status;
  for (const binding of [data.binding, context.binding] as Record<string,unknown>[]) {
    delete binding.registration_revision; delete binding.correction_revision;
    binding.source_binding_sha256 = "ea561075fff5dc22fd245df979d95ef48be1de3568923499de1ecb258cc3b0d5";
  }
  const history: SourceResult = {kind:"ready",validity:"ready",freshness:"current",receivedAt:fixture.received_at,payload};
  const stats: SourceResult = {kind:"held",validity:"held",freshness:"current",receivedAt:fixture.received_at,payload:structuredClone(fixture.stats) as SourcePayload};
  return {history, stats, data, context, row};
}

describe("code-only pinned observed-fee derivation in original table",()=>{
  it("shows allocated cost and scoped net, preserving unknown raw values and held performance",()=>{
    const f=responses(); const rows=physicalHistoryRows(f.history,f.stats);
    expect(rows[0].derivedObservedNet).toEqual({fees:"0.01",net:"-63.39",asOf:"2026-10-25T20:54:00.000Z"});
    expect(rows[0].totalPL).toBeNull(); expect(rows[0].priceSold).toBeNull();
    expect(normalizeStatsPayload(f.stats.payload)).toEqual([]);
    expect(f.stats.payload?.performance_status).toBe("HELD_UNVALIDATED_PERFORMANCE");
    const html=renderToStaticMarkup(<LatestTradesTable rows={rows}/>);
    expect(html).toContain("observed-cost net -$63.39"); expect(html).toContain("Allocated fees $0.01");
    expect(html).not.toContain("Audited fill calculations");
  });
  it.each(["row","allocation","policy","evidence","source_binding","cutoff","revocation","stale_history","stale_stats","false_qualification","source"])("withholds on %s",failure=>{
    const f=responses();
    if(failure==="row")f.row.qty=31;
    if(failure==="allocation")f.context.allocation_state="REVOKED";
    if(failure==="policy")f.stats.payload!.policy_validation_status="UNAVAILABLE";
    if(failure==="revocation")f.context.evidence_status="WITHDRAWN";
    if(failure==="stale_history")f.history.freshness="stale";
    if(failure==="stale_stats")f.stats.freshness="stale";
    if(failure==="false_qualification")f.data.canonical_account_uuid_proved=true;
    if(failure==="source")f.context.source="unverified";
    for (const binding of [f.data.binding,f.context.binding] as Record<string,unknown>[]) {
      if(failure==="evidence")binding.evidence_file_sha256="0".repeat(64);
      if(failure==="source_binding")binding.source_binding_sha256="0".repeat(64);
      if(failure==="cutoff")binding.cutoff="2026-10-26T00:00:00Z";
    }
    const rows=physicalHistoryRows(f.history,f.stats);
    expect(rows).toHaveLength(1);expect(rows[0].derivedObservedNet).toBeNull();
  });
});
