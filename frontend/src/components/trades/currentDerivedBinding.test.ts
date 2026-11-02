import { describe, expect, it } from "vitest";
import { physicalHistoryRows } from "./TradesTab";
import type { SourceResult } from "../ui/sourceState";

function fixture() {
  const binding={trade_id:110,policy_revision:2,allocation_id:"fixture-allocation",allocation_revision:1,
    manifest_sha256:"b".repeat(64),fills_sha256:"c".repeat(64),row_fingerprint:"d".repeat(64),
    row_revision:"2026-04-17T19:05:44.607608+00:00",registration_revision:1,correction_revision:0,
    evidence_file_sha256:"6856c384cd0d76d185a2b312f020f4f9e665969ad0182f2a18925b91e3f5e01b",rule_sha256:"51f51f1b83c3e1e60d6f5df3ecf6e580e35a49db5c559735c55281c2ffd56f53",cutoff:"2026-10-25T20:54:00Z",record_sha256:"a".repeat(64)};
  const row={trade_id:110,symbol:"CRC",status:"CLOSED",qty:30,entry_price:65,exit_price:null,realized_pnl:null,
    entry_time:"2026-04-16T11:00:12.026948+00:00",exit_time:"2026-04-17T19:05:44.628183+00:00",
    updated_at:binding.row_revision,row_fingerprint:binding.row_fingerprint};
  const data={status:"APPROVED_SCOPED_DERIVED_RECORD",source:"postgres_derived_accounting",trade_id:110,
    basis:"CRC110_ALLOCATED_OBSERVED_POSTED_COST_NET_V1",rule:"posted_daily_taf_sell_share_largest_remainder_v1",
    execution_binding:"EXACT_COMMON_FIELDS_AND_THREE_ORDER_HASHES",legacy_validated_performance_eligible:false,
    record_sha256:binding.record_sha256,net_pnl:"-63.39",allocated_fee_cost:"0.01",as_of:binding.cutoff,
    binding:{...binding},bound_row:{...row}};
  const context={schema_version:1,status:"CURRENT_BINDINGS_VERIFIED",source:"postgres_current_derived_read",
    checked_at:"2026-10-25T21:00:00Z",registration_status:"ACTIVE",allocation_state:"ACTIVE",binding:{...binding}};
  const history:SourceResult={kind:"ready",validity:"ready",freshness:"current",receivedAt:"2026-10-25T21:00:01Z",
    payload:{ok:true,source:"postgres",derived_read_context:context,rows:[{...row,derived_accounting:data}]}};
  const allocation={trade_id:110,allocation_id:binding.allocation_id,state:"ACTIVE",allocation_revision:1,
    entry_quantity:"30",exit_quantity:"30",gross_result:"-63.38",financial_eligible:false,fees:null,net:null,fees_status:"UNKNOWN",net_status:"UNKNOWN"};
  const stats:SourceResult={kind:"held",validity:"held",freshness:"current",payload:{ok:false,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"VALIDATED_POLICY",trade_policy:{policy:{scope:"VALIDATED_PERFORMANCE",policy_revision:2,
    allocation_overlay:{status:"VALIDATED",policy_revision:2,active_count:1,allocations:[allocation]}}}}};
  return {history,stats,row:(history.payload!.rows as Record<string,unknown>[])[0],data,context,allocation};
}
describe("UTC holding calendar",()=>{
  it("matches backend Feb17 17:48 to Feb18 14:36 calendar-day regression",()=>{
    const f=fixture();f.row.entry_time="2026-02-17T17:48:38.459926+00:00";f.row.exit_time="2026-02-18T14:36:28.400032+00:00";
    expect(physicalHistoryRows(f.history,f.stats)[0].totalDays).toBe(1);
  });
  it.each([["2026-02-18T01:00:00Z","2026-02-18T02:00:00Z",0],
    ["2026-02-17T23:30:00-04:00","2026-02-18T23:00:00Z",0],
    ["2026-02-18T14:00:00Z","2026-02-17T15:00:00Z",null],
    [null,"2026-02-18T00:00:00Z",null],["2026-02-17T00:00:00Z",null,null],
    ["2026-02-31T00:00:00Z","2026-03-04T00:00:00Z",null]])("preserves calendar/unknown semantics %s %s",(entry,exit,expected)=>{
    const f=fixture();f.row.entry_time=entry;f.row.exit_time=exit;
    expect(physicalHistoryRows(f.history,f.stats)[0].totalDays).toBe(expected);
  });
});
describe("current derived accounting read contract",()=>{
  it("displays a matching current row, active allocation and registered record",()=>{
    const f=fixture();expect(physicalHistoryRows(f.history,f.stats)[0].derivedObservedNet?.net).toBe("-63.39");
  });
  it.each(["symbol","status","quantity","entry","exit","entry_price","recorded_pnl","row_revision","row_hash","policy_revision","allocation_revision","allocation_id","registration_revision","correction","cutoff","evidence","rule_hash","record_hash","revoked","superseded","allocation_revoked","stats_revoked","missing_context","stale_stats","stale_history","coherent_wrong_evidence","coherent_wrong_rule","coherent_wrong_cutoff"])("rejects drift or invalidation: %s",kind=>{
    const f=fixture();
    if(kind==="coherent_wrong_evidence"){f.data.binding.evidence_file_sha256="0".repeat(64);f.context.binding.evidence_file_sha256="0".repeat(64);}
    if(kind==="coherent_wrong_rule"){f.data.binding.rule_sha256="0".repeat(64);f.context.binding.rule_sha256="0".repeat(64);}
    if(kind==="coherent_wrong_cutoff"){f.data.binding.cutoff="2026-10-25T21:00:00Z";f.context.binding.cutoff=f.data.binding.cutoff;f.data.as_of=f.data.binding.cutoff;}
    if(kind==="symbol")f.row.symbol="OTHER";
    if(kind==="status")f.row.status="OPEN";
    if(kind==="quantity")f.row.qty=31;
    if(kind==="entry")f.row.entry_time="2026-04-16T11:00:12.026949+00:00";
    if(kind==="exit")f.row.exit_time=null;
    if(kind==="entry_price")f.row.entry_price=66;
    if(kind==="recorded_pnl")f.row.realized_pnl=0;
    if(kind==="row_revision")f.row.updated_at="2026-04-17T19:05:44.607609+00:00";
    if(kind==="row_hash")f.row.row_fingerprint="0".repeat(64);
    if(kind==="policy_revision")f.context.binding.policy_revision=3;
    if(kind==="allocation_revision")f.allocation.allocation_revision=2;
    if(kind==="allocation_id")f.allocation.allocation_id="changed";
    if(kind==="registration_revision")f.context.binding.registration_revision=2;
    if(kind==="correction")f.context.binding.correction_revision=1;
    if(kind==="cutoff")f.context.binding.cutoff="2026-10-25T21:00:00Z";
    if(kind==="evidence")f.context.binding.evidence_file_sha256="0".repeat(64);
    if(kind==="rule_hash")f.context.binding.rule_sha256="0".repeat(64);
    if(kind==="record_hash")f.context.binding.record_sha256="0".repeat(64);
    if(kind==="revoked")f.context.registration_status="REVOKED";
    if(kind==="superseded")f.context.registration_status="SUPERSEDED";
    if(kind==="allocation_revoked")f.context.allocation_state="REVOKED";
    if(kind==="stats_revoked")f.allocation.state="REVOKED";
    if(kind==="missing_context")delete f.history.payload!.derived_read_context;
    if(kind==="stale_stats")f.stats.freshness="stale";
    if(kind==="stale_history")f.history.freshness="stale";
    expect(physicalHistoryRows(f.history,f.stats)[0].derivedObservedNet).toBeNull();
  });
});
