import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { SourceResult } from "../ui/sourceState";
import StrategyHistoryCard from "./StrategyHistoryCard";
import { strategyHistoryView } from "./strategyHistory";
import { strategyHistoryFixture } from "../../../qa/fixtures/strategyHistoryFixture";
import wireFixture from "../../../qa/fixtures/strategyHistoryWire.synthetic.json";

const source = (): SourceResult => ({payload: strategyHistoryFixture(), kind:"ready", validity:"ready", freshness:"current", receivedAt:"2026-11-09T16:00:01Z"});
const row = (result: SourceResult) => (result.payload!.rows as Record<string, unknown>[])[0];
const render = (result: SourceResult) => renderToStaticMarkup(<StrategyHistoryCard result={result}/>);

describe("stored strategy history presentation contract", () => {
  it("shows a complete synthetic signal with source, native state, independent gross and unknown costs", () => {
    const result = source(); const view = strategyHistoryView(result);
    expect(view.kind).toBe("ready"); expect(view.rows[0]).toMatchObject({symbol:"SYNTH",state:"CLOSED",gross:80,remainingQty:0,status:"complete"});
    const html=render(result);
    for (const text of ["Strategy Trade History", "SYNTH", "CLOSED", "Full exit observed", "$80.00", "Fees: Unknown", "Net: Unknown", "2026-10-05 10:00:00 UTC", "not added to its performance totals", "stored observations, not live broker state"]) expect(html).toContain(text);
    expect(html).not.toContain("$0.00"); expect(html).not.toContain("Validated performance");
  });
  it("accepts the backend-produced synthetic v1 response without symbol-specific code", () => {
    const result={...source(),payload:wireFixture};const view=strategyHistoryView(result);
    expect(view.rows).toHaveLength(1);expect(view.rows[0]).toMatchObject({symbol:"SYNTH",gross:80,entryQty:40,exitedQty:40,remainingQty:0,status:"complete"});
    expect(render(result)).toContain("$80.00");expect(render(result)).toContain("2026-10-05 10:00:00 UTC");
    expect(render(result)).toContain("Fees: Unknown");expect(render(result)).toContain("Net: Unknown");
  });
  it.each(["schema_version", "source", "trade_view", "evidence_basis", "performance_status", "accounting_complete", "fees_status", "net_status", "account_scope", "freshness"])("fails closed for envelope mismatch: %s", key => {
    const result = source(); result.payload![key] = "unexpected";
    const view = strategyHistoryView(result); expect(view.kind).toBe("unavailable"); expect(view.rows).toEqual([]);
    expect(render(result)).not.toContain("$80.00"); expect(render(result)).toContain("not an empty trading history");
  });
  it.each(["PARTIAL_EXIT_OBSERVED", "NO_EXIT_OBSERVED", "FULL_EXIT_OBSERVED"])("preserves native state separately from %s and never invents completed P/L", completion => {
    const result=source(); Object.assign(row(result), {state:"OPEN",status:"partial",completion_status:completion,exited_qty:completion==="NO_EXIT_OBSERVED"?0:16,remaining_qty:24});
    const value = strategyHistoryView(result).rows[0]; expect(value.state).toBe("OPEN"); expect(value.completion).toBe(completion); expect(value.gross).toBeNull();
    expect(strategyHistoryView(result).kind).toBe("ready"); // Valid unfinished lifecycle, not a transport error.
    expect(render(result)).not.toContain("$80.00"); expect(render(result)).toContain("OPEN");
  });
  it("keeps an unfilled signal distinct from an open or closed position", () => {
    const result=source(); Object.assign(row(result), {state:"SUBMITTED",status:"partial",completion_status:"NO_ENTRY_FILL_OBSERVED",entry_qty:0,exited_qty:0,remaining_qty:0,entry_avg_price:null,exit_avg_price:null,entry_time:null,exit_time:null});
    const html=render(result); expect(html).toContain("SUBMITTED"); expect(html).toContain("Entry unfilled in stored observation"); expect(html).not.toContain("$0.00"); expect(html).not.toContain("Full exit observed");
  });
  it.each(["account", "basis", "fee", "net", "native_state", "incomplete", "completion"])("retains native identity but suppresses financial fields for %s", failure => {
    const result=source(), value=row(result);
    if(failure==="account") result.payload!.broker_account_match="unavailable";
    if(failure==="basis") value.evidence_basis="theoretical_fills";
    if(failure==="fee") value.fees=0;
    if(failure==="net") value.net_realized_pnl=80;
    if(failure==="native_state") value.state="OPEN";
    if(failure==="incomplete") value.status="incomplete";
    if(failure==="completion") value.completion_status="NEW_STATUS";
    const parsed=strategyHistoryView(result).rows[0]; expect(parsed.symbol).toBe("SYNTH"); expect(parsed.gross).toBeNull(); expect(parsed.entryQty).toBeNull(); expect(parsed.entryTime).toBeNull();
    expect(render(result)).toContain("Fill evidence incomplete"); expect(render(result)).not.toContain("$80.00");
  });
  it.each([null,undefined,"",false,NaN,Infinity,"12",{},[]])("does not coerce unknown or malformed numeric values (%s) to zero", value => {
    const result=source(); Object.assign(row(result),{gross_realized_pnl:value,entry_qty:value,exit_avg_price:value});
    expect(strategyHistoryView(result).rows[0]).toMatchObject({gross:null,entryQty:null,exitPrice:null});
  });
  it("preserves genuine zero gross without treating fees/net as known", () => {
    const result=source(); row(result).gross_realized_pnl=0;
    expect(strategyHistoryView(result).rows[0].gross).toBe(0); expect(render(result)).toContain("$0.00"); expect(render(result)).toContain("Net: Unknown");
  });
  it.each(["entry_qty","exited_qty","remaining_qty","entry_avg_price","exit_avg_price","entry_time","exit_time"])("does not show complete gross with missing %s",key=>{
    const result=source();row(result)[key]=null;expect(strategyHistoryView(result).rows[0].gross).toBeNull();expect(render(result)).not.toContain("$80.00");
  });
  it("treats invalid calendar timestamps as unknown rather than silently normalizing them",()=>{
    const result=source();row(result).exit_time="2026-02-31T15:00:00Z";
    expect(strategyHistoryView(result).rows[0].exitTime).toBeNull();expect(strategyHistoryView(result).rows[0].gross).toBeNull();
    result.payload!.observed_at="not-a-time";expect(strategyHistoryView(result).kind).toBe("unavailable");
  });
  it.each([
    {exited_qty:39}, {entry_time:"2026-10-06T10:00:00Z"},
    {entry_order_ids:[]}, {exit_order_ids:[]}, {entry_order_ids:null},
    {entry_order_ids:["synthetic-entry-0","synthetic-entry-0"]},
    {exit_order_ids:["synthetic-exit-0","synthetic-exit-0"]},
    {exit_order_ids:["synthetic-entry-0"]}, {entry_order_ids:[" "]},
    {exit_order_ids:["synthetic-exit-0",null]},
  ])("fails closed on inconsistent complete-row provenance: %j", mutation=>{
    const result=source();Object.assign(row(result),mutation);
    const view=strategyHistoryView(result);expect(view.kind).toBe("partial");
    expect(view.rows[0]).toMatchObject({state:"CLOSED",status:"incomplete",completion:"UNVERIFIED",gross:null,entryQty:null});
    expect(render(result)).not.toContain("$80.00");expect(render(result)).toContain("Fill evidence incomplete");
  });
  it.each(["synth"," SYNTH","SYNTH ","1SYNTH","SYN/TH","ABCDEFGHIJKLMNOPQ"])("omits invalid native symbol identity %s",symbol=>{
    const result=source();row(result).symbol=symbol;const view=strategyHistoryView(result);
    expect(view.rows).toEqual([]);expect(view.omitted).toBe(1);expect(view.kind).toBe("partial");
    expect(render(result)).not.toContain("$80.00");
  });
  it("flags malformed and duplicate identities without claiming empty", () => {
    const result=source(); result.payload!.rows=[row(result),row(result),{...row(result),signal_id:"bad",symbol:"BAD"}];
    const view=strategyHistoryView(result); expect(view.rows).toHaveLength(1); expect(view.omitted).toBe(2); expect(view.kind).toBe("partial"); expect(render(result)).toContain("Coverage is incomplete");
  });
  it("labels bounded coverage and unknown coverage rather than silently presenting all history", () => {
    const result=source(); result.payload!.has_more=true; expect(render(result)).toContain("More signals exist; older history is not shown");
    expect(render(result)).toContain("Latest 25 strategy signals requested");
    result.payload!.has_more=null; expect(render(result)).toContain("Additional-history coverage is unknown");
  });
  it("distinguishes empty, unavailable, loading and stale retained empty", () => {
    const result=source(); result.payload!.rows=[]; result.payload!.status="empty"; result.kind="empty";
    expect(render(result)).toContain("successful stored-history read"); result.freshness="stale"; result.kind="stale";
    expect(render(result)).toContain("Current history is unverified"); expect(render(result)).not.toContain("successful stored-history read");
    result.kind="unavailable"; expect(render(result)).toContain("strategy count and financial results are unknown"); expect(render(result)).not.toContain("Showing 0 signals");
    result.kind="loading"; expect(render(result)).toContain("Loading stored strategy history"); expect(render(result)).not.toContain("No strategy signals");
  });
  it("uses the existing theme and separates mobile cards from a contained scrollable table", () => {
    const html=render(source()); expect(html).toContain("jbravo-panel-emerald"); expect(html).toContain('class="mt-3 space-y-3 sm:hidden"');
    expect(html).toContain("max-w-full overflow-auto sm:block"); expect(html).toContain("min-w-[920px]"); expect(html).toContain("dark:bg-slate-900/95");
    expect(html).toContain("break-all"); expect(html).toContain("Unknown amounts are not zero");
  });
  it("retains recognition-only scope without claiming current management authority", () => {
    const result=source(); Object.assign(row(result),{ownership_scope:"ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS",ownership_basis:"STORED_SIGNAL_PAYLOAD_NOT_REVALIDATED"});
    const html=render(result);expect(html).toContain("entry recognition only (no position orders)");expect(html).toContain("current management authority has not been revalidated");
    row(result).ownership_scope="UNRECOGNIZED_PRIVATE_PAYLOAD";
    expect(render(result)).toContain("Ownership unknown");expect(render(result)).not.toContain("UNRECOGNIZED_PRIVATE_PAYLOAD");
  });
});
