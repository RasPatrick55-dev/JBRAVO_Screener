import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import AllocationGrossPanel, { allocationGrossRows } from "./AllocationGrossPanel";
import { watchSource, type SourceResult } from "../ui/sourceState";
import { physicalHistoryRows } from "./TradesTab";
import LatestTradesTable from "./LatestTradesTable";


function held(): SourceResult {
  return {kind: "held", validity: "held", freshness: "current", payload: {
    ok: false, source: "postgres", performance_status: "HELD_UNVALIDATED_PERFORMANCE", policy_validation_status: "VALIDATED_POLICY", rows: [],
    trade_policy: {policy: {scope: "VALIDATED_PERFORMANCE", policy_revision: 2, allocation_overlay: {
      status: "VALIDATED", policy_revision: 2, active_count: 1, allocations: [{
        trade_id: 110, state: "ACTIVE", allocation_revision: 1, entry_quantity: "30", exit_quantity: "30", gross_result: "-63.38",
        financial_eligible: false, fees: null, net: null, fees_status: "UNKNOWN", net_status: "UNKNOWN",
        order_ids: ["private-order-not-for-display"], activity_ids: ["private-activity-not-for-display"],
      }],
    }}},
  }};
}
function parts(result: SourceResult) {
  const policy = (result.payload!.trade_policy as {policy: Record<string, unknown>}).policy;
  const overlay = policy.allocation_overlay as {policy_revision: number; active_count: number; status: string; allocations: Record<string, unknown>[]};
  return {policy, overlay, row: overlay.allocations[0]};
}
const markup = (result: SourceResult) => renderToStaticMarkup(<AllocationGrossPanel result={result}/>);
afterEach(() => {vi.useRealTimers();});

describe("audited gross population", () => {
  it("keeps the legacy allocation adapter separate from combined gross and original ledger values", () => {
    const result=held(), before=JSON.stringify(result);
    const history:SourceResult={kind:"ready",validity:"ready",freshness:"current",payload:{ok:true,rows:[{trade_id:110,symbol:"CRC",status:"CLOSED",qty:30,realized_pnl:null}]}};
    const rows=physicalHistoryRows(history,result);
    expect(rows[0]).toMatchObject({grossPL:"-63.38",totalPL:null,derivedObservedNet:null});
    const html=renderToStaticMarkup(<LatestTradesTable rows={rows} plLabel="Recorded P/L"/>);
    expect(html).toContain("Gross -$63.38; fees/net unknown");expect(html).toContain("Recorded P/L");
    expect(html).not.toContain("private-order");expect(html).not.toContain("private-activity");expect(JSON.stringify(result)).toBe(before);
  });
  it("repeated renders are idempotent and keep fees/net unknown", () => {
    const result = held(), before = JSON.stringify(result);
    expect(markup(result)).toBe(markup(result)); expect(JSON.stringify(result)).toBe(before);
    expect(allocationGrossRows(result)?.rows).toEqual([{tradeId:110, allocationRevision:1, entryQuantity:"30", exitQuantity:"30", gross:"-63.38"}]);
    expect(markup(result)).not.toContain("$0.00");
  });
  it("preserves fractional shares and exact Decimal text beyond binary precision", () => {
    const result = held(), {row} = parts(result); row.entry_quantity = "0.1250"; row.exit_quantity = "00.125";
    row.gross_result = "12345678901234567890.123456789";
    expect(markup(result)).toContain("$12345678901234567890.123456789");
    expect(allocationGrossRows(result)?.rows[0].entryQuantity).toBe("0.125");
    row.gross_result = "0"; expect(markup(result)).toContain("$0.00");
  });
  it.each(["quantity", "revision", "policy", "duplicate", "missing", "nan", "number"])("refuses inconsistent evidence: %s", kind => {
    const result = held(), {policy, overlay, row} = parts(result);
    if (kind === "quantity") row.exit_quantity = "29";
    if (kind === "revision") overlay.policy_revision = 3;
    if (kind === "policy") policy.scope = "EXECUTION";
    if (kind === "duplicate") {overlay.allocations.push({...row}); overlay.active_count = 2;}
    if (kind === "missing") overlay.active_count = 2;
    if (kind === "nan") row.gross_result = "NaN";
    if (kind === "number") row.gross_result = -63.38;
    expect(allocationGrossRows(result)).toBeNull(); expect(markup(result)).not.toContain("-$63.38");
  });
  it.each(["fees", "net", "eligibility"])("does not infer or accept a financial qualification: %s", field => {
    const result = held(), {row} = parts(result);
    row[field === "eligibility" ? "financial_eligible" : field] = field === "eligibility" ? true : "0";
    expect(allocationGrossRows(result)).toBeNull(); expect(markup(result)).not.toContain("$0.00");
  });
  it("distinguishes a validated empty allocation set from unavailable policy", () => {
    const result = held(), {overlay} = parts(result); overlay.allocations = []; overlay.active_count = 0;
    expect(markup(result)).toContain("No active audited fill allocations");
    result.payload!.policy_validation_status = "UNAVAILABLE";
    expect(markup(result)).toContain("calculations unavailable"); expect(markup(result)).not.toContain("No active");
  });
  it("labels retained held evidence stale after failed polling and never makes net displayable", async () => {
    vi.useFakeTimers(); const events:SourceResult[] = [], original = held();
    const transport = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(original.payload), {status:200})).mockRejectedValueOnce(new Error("offline"));
    const stop = watchSource("/stats", result => events.push(result), 20000, transport);
    for (let i=0;i<12;i++) await Promise.resolve();
    await vi.advanceTimersByTimeAsync(20000);
    const last = events[events.length-1]; expect(last.validity).toBe("held"); expect(last.freshness).toBe("stale");
    expect(markup(last)).toContain("STALE"); expect(markup(last)).toContain("-$63.38"); expect(markup(last)).toContain("Not eligible"); stop();
    const refused = held(); refused.validity = "unavailable"; refused.payload!.policy_validation_status = "UNAVAILABLE";
    expect(markup(refused)).not.toContain("-$63.38");
  });
  it("keeps legacy history adapter independent when policy is unavailable, including fractions/nulls and UTC dates", () => {
    const stats:SourceResult={...held(),kind:"unavailable",validity:"unavailable",payload:null};
    const history:SourceResult={kind:"partial",validity:"partial",freshness:"stale",receivedAt:"2026-10-25T20:57:55Z",payload:{ok:true,source:"postgres",rows:[{
      trade_id:111,symbol:"FRACTION",status:"CLOSED",qty:"0.125",entry_time:"2026-04-16T23:30:00-04:00",exit_time:null,entry_price:null,exit_price:null,realized_pnl:null,
    }]}};
    const rows=physicalHistoryRows(history,stats);expect(rows[0]).toMatchObject({totalShares:0.125,buyDate:"2026-04-17T03:30:00.000Z",totalPL:null});
    const html=renderToStaticMarkup(<LatestTradesTable rows={rows}/>);expect(html).toContain("FRACTION");expect(html).toContain("4/17/2026");expect(html).not.toContain("$0.00");
  });
  it("returns no legacy rows for unavailable or successful empty samples without inventing values", () => {
    const stats=held();
    expect(physicalHistoryRows({kind:"unavailable",validity:"unavailable",freshness:"unknown",payload:null},stats)).toEqual([]);
    expect(physicalHistoryRows({kind:"empty",validity:"empty",freshness:"current",payload:{ok:true,rows:[]}},stats)).toEqual([]);
  });
  it("refuses approved-looking legacy net tags without authoritative current read bindings",()=>{
    const stats=held();
    const derived={status:"APPROVED_SCOPED_DERIVED_RECORD",source:"postgres_derived_accounting",trade_id:110,
      basis:"CRC110_ALLOCATED_OBSERVED_POSTED_COST_NET_V1",rule:"posted_daily_taf_sell_share_largest_remainder_v1",
      execution_binding:"EXACT_COMMON_FIELDS_AND_THREE_ORDER_HASHES",legacy_validated_performance_eligible:false,
      record_sha256:"a".repeat(64),net_pnl:"-63.39",allocated_fee_cost:"0.01",as_of:"2026-10-25T20:54:00Z"};
    const history:SourceResult={kind:"ready",validity:"ready",freshness:"current",payload:{ok:true,rows:[
      {trade_id:110,symbol:"CRC",status:"CLOSED",qty:30,realized_pnl:null,derived_accounting:derived},
    ]}};
    const rows=physicalHistoryRows(history,stats);expect(rows[0].derivedObservedNet).toBeNull();
    const html=renderToStaticMarkup(<LatestTradesTable rows={rows}/>);expect(html).not.toContain("observed-cost net -$63.39");expect(html).toContain("fees/net unknown");
  });
});
