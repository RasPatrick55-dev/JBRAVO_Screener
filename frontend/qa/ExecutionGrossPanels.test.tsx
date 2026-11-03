import {afterEach,describe,expect,it,vi} from "vitest";
import {renderToStaticMarkup} from "react-dom/server";
import TradesTab from "../src/components/trades/TradesTab";
import * as sources from "../src/components/ui/sourceState";
import {recordedGrossContract,normalizeRecordedGrossStats} from "../src/components/trades/recordedGrossData";
import type {SourcePayload,SourceResult} from "../src/components/ui/sourceState";
import fixture from "./executionGrossFixture.json";
import {writeFileSync} from "node:fs";

afterEach(()=>vi.restoreAllMocks());
const source=(payload:SourcePayload,kind?:SourceResult["kind"]):SourceResult=>({payload,
  kind:kind??sources.sourceKind(payload),validity:sources.sourceValidity(payload),freshness:kind==="stale"?"stale":"current",
  receivedAt:"2001-02-03T12:00:00Z"});
function render(stats:SourcePayload=fixture.stats,leaders:SourcePayload=fixture.winners,kind?:SourceResult["kind"]){
  const requested:string[]=[];
  vi.spyOn(sources,"useSource").mockImplementation(path=>{
    requested.push(path);
    if(path.includes("/history"))return source({ok:true,source:"postgres",rows:[]});
    if(path.includes("/leaderboard"))return source(leaders,kind);
    if(path.includes("basis=execution_gross"))return source(stats,kind);
    return source({ok:false,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"VALIDATED_POLICY",rows:[]});
  });
  return {html:renderToStaticMarkup(<TradesTab/>),requested};
}
describe("actual synthetic backend execution projection into original cards",()=>{
  it("shows weighted gross with partial coverage and held net, independently of the physical page",()=>{
    const {html,requested}=render();
    expect(html).toContain("Execution gross performance PARTIAL");expect(html).toContain("+$4.00");
    expect(html).toContain("2 included / 1 policy-excluded / 1 unknown");expect(html).toContain("missing linked fills");
    expect(html).toContain("50.00%");expect(html).toContain("FICT");expect(html).toContain("Qualified net: Performance held");
    expect(html).toContain("History retains original Recorded P/L");expect(html).toContain("2001-02-03");
    expect(requested).toContain("/api/trades/stats?range=all");
    expect(requested).toContain("/api/trades/stats?range=all&basis=execution_gross");
    expect(requested).toContain("/api/trades/leaderboard?range=all&mode=winners&limit=12&basis=execution_gross");
    expect(html).toContain("xl:grid-cols-3");expect(html).toContain("order-3");expect(html).toContain("order-2");
    expect(html).not.toContain("Audited fill calculations");expect(html).not.toContain("No leaderboard trades yet");
    if(process.env.JBR_REVIEW_HTML_PATH)writeFileSync(process.env.JBR_REVIEW_HTML_PATH,
      `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic component render</title></head><body><p>LOCAL SYNTHETIC COMPONENT RENDER — not browser pixel QA or live data.</p>${html}</body></html>`);
  });
  it.each(["execution_fingerprint","policy_fingerprint","pnl_basis","view_scope"])("rejects absent/mismatched %s rather than relabeling legacy values",key=>{
    const payload:SourcePayload=structuredClone(fixture.stats);payload[key]="wrong";
    expect(recordedGrossContract(payload,"execution_gross")).toBe(false);
    expect(normalizeRecordedGrossStats(payload,"execution_gross")).toEqual([]);
    expect(render(payload).html).toContain("Execution gross inputs unavailable");
  });
  it("does not accept an execution result as the legacy basis or a net-qualified result",()=>{
    expect(recordedGrossContract(fixture.stats)).toBe(false);
    const payload:SourcePayload={...fixture.stats,net_eligible:true};
    expect(recordedGrossContract(payload,"execution_gross")).toBe(false);
  });
  it("retains unknown coverage in stale empty ranks rather than successful empty",()=>{
    const payload:SourcePayload={...fixture.winners,rows:[],coverage:{included:0,excluded:1,unknown:3,reasons:{QUANTITY_MISMATCH:3}}};
    const {html}=render(fixture.stats,payload,"stale");
    expect(html).toContain("STALE (stale read)");expect(html).toContain("3 unknown");
    expect(html).toContain("No known gross results in this ranking; unresolved records may change it.");
    expect(html).not.toContain("No positive gross results in the selected period.");
  });
  it.each(["2001-02-30T12:00:00Z","2001-02-03T12:00:00+00:60",undefined])("refuses invalid or absent observation time %s",observed=>{
    expect(recordedGrossContract({...fixture.stats,observed_at:observed},"execution_gross")).toBe(false);
  });
  it("refuses future source data and preserves valid offset-equivalent timestamps",()=>{
    expect(recordedGrossContract({...fixture.stats,source_as_of:"2001-02-04T00:00:00Z"},"execution_gross")).toBe(false);
    expect(recordedGrossContract({...fixture.stats,source_as_of:"2001-02-03T06:00:00-05:00"},"execution_gross")).toBe(true);
  });
});
