// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import TradesTab from "../src/components/trades/TradesTab";
import CombinedTradesView from "../src/components/trades/CombinedTradesView";
import { combinedOverview, combinedGrossReadPolicy, combinedTimestamp, COMBINED_GROSS_PATH } from "../src/components/trades/combinedOverview";
import { watchSource, type SourceResult } from "../src/components/ui/sourceState";
import wire from "./fixtures/combinedGrossWire.synthetic.json";
import { combinedGrossPaginationFixture } from "./fixtures/combinedGrossFixture";

type Obj=Record<string,any>;
const fixture=():Obj=>structuredClone(wire);
const source=(payload=fixture()):SourceResult=>({payload,kind:payload.status === "partial" ? "partial" : payload.status === "empty" ? "empty" : "ready",validity:payload.status === "partial" ? "partial" : payload.status === "empty" ? "empty" : "ready",freshness:"current",receivedAt:payload.observed_at,lastAttemptAt:payload.observed_at});
const markup=(result=source())=>renderToStaticMarkup(<CombinedTradesView result={result}/>);
const settle=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
function unavailable(reason="source_timeout",failure_stage="database_cohort_read"):Obj {
  const p=fixture();return {...p,ok:false,status:"unavailable",reason,failure_stage,cohort_id:null,source_as_of:null,population_complete:false,
    strategy_account_binding:"unavailable",policy_validation_status:"UNAVAILABLE",coverage:null,rows:[],stats:[],rankings:{},strategy_history:null};
}
function emptyFixture():Obj {
  const p=fixture();const cov={included:0,excluded:0,unknown:0,deduplicated:0,strategy_not_closed:0,by_source:{legacy:{included:0,excluded:0,unknown:0},strategy:{included:0,excluded:0,unknown:0}}};
  p.status="empty";p.rows=[];p.coverage=cov;p.source_as_of=null;
  p.stats=p.stats.map((row:Obj)=>({...row,tradesCount:0,totalPL:"0",winRatePct:null,topTrade:{symbol:"--",pl:null},worstLoss:{symbol:"--",pl:null},coverage:structuredClone(cov)}));
  p.rankings=Object.fromEntries(Object.keys(p.rankings).map(key=>[key,{winners:[],losers:[]}]));
  p.strategy_history={...p.strategy_history,rows:[],status:"empty",source_as_of:null,has_more:false};return p;
}
let root:Root|undefined,container:HTMLDivElement|undefined;
afterEach(()=>{if(root){act(()=>root!.unmount());root=undefined;}container?.remove();container=undefined;vi.useRealTimers();vi.unstubAllGlobals();});
function mount(){container=document.createElement("div");document.body.append(container);root=createRoot(container);act(()=>root!.render(<TradesTab/>));}
function click(text:string){const button=Array.from(container!.querySelectorAll("button")).find(button=>button.textContent === text);expect(button).toBeDefined();act(()=>button!.click());}

describe("actual combined Trades contract and presentation",()=>{
  it("accepts the Python-produced immutable wire and shows the same cohort throughout",()=>{
    const p=fixture(),before=JSON.stringify(p),v=combinedOverview(p);expect(v).not.toBeNull();
    expect(v!.rows.length).toBe(p.rows.length);expect(v!.stats).toHaveLength(5);
    const html=markup(source(p));
    for(const text of ["Combined stored gross","Strategy Trade History","Latest Trades","TOP GROSS WINNERS","Legacy account binding: unverified","Fees and net P/L are unknown","accounting qualification is not established","rolling UTC 1, 7, 30 and 365 days","Original Recorded P/L","Gross cohort:"])expect(html).toContain(text);
    expect(html).not.toContain("These rows are not added");expect(html).not.toContain("observed-cost net");
    expect(JSON.stringify(p)).toBe(before);expect(markup(source(p))).toBe(html);
  });
  it("preserves full source clock precision while leaving range arithmetic to Python",()=>{
    const p=fixture(),row=p.rows.find((r:Obj)=>r.inclusion === "INCLUDED");row.exit_time="2026-11-09T14:20:25.705162671+00:00";
    expect(combinedOverview(p)!.rows.find(r=>r.rowId===row.row_id)!.sellDate).toBe(row.exit_time);
    expect(combinedTimestamp(row.exit_time)).toBe(true);expect(combinedTimestamp("2026-02-31T14:20:25Z")).toBe(false);
    expect(combinedTimestamp("2026-11-09T25:00:00Z")).toBe(false);expect(combinedTimestamp("2026-11-09T14:20:25.1234567890Z")).toBe(false);
  });
  it.each(["schema_version","source","view_scope","performance_status","qualified_net_status","net_eligible","accounting_complete","recovery_complete","entry_ready","fees","net_pnl","fees_status","net_status","strategy_account_binding","legacy_account_binding","policy_validation_status","source_freshness","population_complete","cohort_id","observed_at"])("rejects a changed envelope without retaining misleading totals: %s",key=>{
    const p=fixture();p[key]="invalid";expect(combinedOverview(p)).toBeNull();const html=markup(source(p));expect(html).toContain("Combined gross unavailable");expect(html).toContain("combined count and amounts are unknown");expect(html).not.toContain("stored candidates in this cohort");expect(html).not.toContain("Stored broker account binding: Verified");
  });
  it.each(["row_id","source_kind","signal_id","account_binding","inclusion","fees","net_pnl","price_basis","holding_days"])("rejects malformed source row %s rather than dropping it but showing totals",key=>{
    const p=fixture(),row=p.rows.find((r:Obj)=>r.source_kind === "STRATEGY_EXECUTION");row[key]="invalid";expect(combinedOverview(p)).toBeNull();
  });
  it.each([undefined,"",false,true,"NaN","Infinity","1e3",Infinity,NaN,{},[],"99999999999999999999999999999999"])("rejects malformed/overflow gross without zero coercion: %s",value=>{
    const p=fixture();p.rows.find((r:Obj)=>r.inclusion === "INCLUDED").gross_pnl=value;expect(combinedOverview(p)).toBeNull();
  });
  it("preserves genuine zero and unknown separately",()=>{
    const p=fixture(),row=p.rows.find((r:Obj)=>r.inclusion === "INCLUDED");row.gross_pnl="0";
    expect(combinedOverview(p)!.rows.find(r=>r.rowId===row.row_id)!.totalPL).toBe(0);expect(markup(source(p))).toContain("$0.00");
    row.gross_pnl=null;expect(combinedOverview(p)).toBeNull();
    const unknown=fixture().rows.find((r:Obj)=>r.inclusion === "UNKNOWN");expect(unknown.gross_pnl).toBeNull();expect(markup()).toContain("Unknown amounts are not zero");
  });
  it("retains a valid nonincluded legacy identity with unknown symbol explicitly",()=>{
    const p=fixture(),row=p.rows.find((r:Obj)=>r.source_kind === "LEGACY_EXECUTION" && r.inclusion !== "INCLUDED");row.symbol=null;
    const v=combinedOverview(p);expect(v).not.toBeNull();expect(v!.rows.find(r=>r.rowId===row.row_id)!.symbol).toBe("Unknown");
    row.inclusion="INCLUDED";expect(combinedOverview(p)).toBeNull();
  });
  it("rejects duplicate rows, range keys and symbol ranks",()=>{
    const p=fixture();p.rows.push(p.rows[0]);expect(combinedOverview(p)).toBeNull();
    const q=fixture();q.stats[1].key=q.stats[0].key;expect(combinedOverview(q)).toBeNull();
    const r=fixture();r.rankings.all.winners.push({...r.rankings.all.winners[0],rank:r.rankings.all.winners.length+1});expect(combinedOverview(r)).toBeNull();
  });
  it("rejects unknown source fields, bad counts, nonincluded money and cross-observation strategy data",()=>{
    const mutations=[(p:Obj)=>p.coverage.included++, (p:Obj)=>p.stats[0].tradesCount++, (p:Obj)=>p.stats[0].totalPL="NaN", (p:Obj)=>p.coverage.by_source.legacy.included++, (p:Obj)=>p.stats[0].totalPL=null,
      (p:Obj)=>p.rankings.all.winners[0].pl="-4", (p:Obj)=>p.strategy_history.observed_at="2000-01-01T00:00:00Z",
      (p:Obj)=>p.strategy_history.rows[0].combined_inclusion="INVALID",(p:Obj)=>p.rows.find((r:Obj)=>r.inclusion!=="INCLUDED").gross_pnl="1"];
    for(const mutation of mutations){const p=fixture();mutation(p);expect(combinedOverview(p)).toBeNull();}
  });
  it("distinguishes complete empty, unknown-only, loading and unavailable",()=>{
    const empty=emptyFixture();expect(combinedOverview(empty)).not.toBeNull();expect(markup(source(empty))).toContain("No realized trade candidates");
    expect(markup(source(empty))).toContain("$0.00");expect(markup(source(empty))).toContain("Unknown");
    const loading=markup({payload:null,kind:"loading",validity:"loading",freshness:"unknown"});expect(loading).toContain("Loading combined trade history");expect(loading).not.toContain("Combined gross unavailable");
    const p=unavailable("population_overflow");expect(combinedOverview(p)).toBeNull();expect(markup({...source(p),kind:"unavailable",validity:"unavailable"})).not.toContain("$0.00");expect(markup({...source(p),kind:"unavailable",validity:"unavailable"})).toContain("exceeds the read limit");
  });
  it("keeps open recognition-only and unfilled signals out of realized candidates",()=>{
    const p=fixture(),v=combinedOverview(p)!;
    const unfinished=p.strategy_history.rows.filter((row:Obj)=>row.combined_inclusion === "NOT_CLOSED");expect(unfinished).toHaveLength(2);
    for(const row of unfinished)expect(v.rows.some(candidate=>candidate.rowId === `strategy:${row.signal_id}`)).toBe(false);
    expect(markup()).toContain("entry recognition only (no position orders)");expect(markup()).toContain("NOT CLOSED");
    expect(v.stats.find(row=>row.key === "all")!.totalPL).toBe(44);expect(v.stats.find(row=>row.key === "all")!.tradesCount).toBe(3);
    expect(markup()).toContain("+$44.00");
  });
  it("shows unknown-only evidence without manufacturing zero totals or a win rate",()=>{
    const p=fixture(),unknown=p.rows.find((row:Obj)=>row.inclusion === "UNKNOWN");p.rows=[unknown];p.status="partial";p.source_as_of=null;
    const cov={included:0,excluded:0,unknown:1,deduplicated:0,strategy_not_closed:2,by_source:{legacy:{included:0,excluded:0,unknown:0},strategy:{included:0,excluded:0,unknown:1}}};p.coverage=cov;
    p.stats=p.stats.map((row:Obj)=>({...row,tradesCount:0,totalPL:null,winRatePct:null,topTrade:{symbol:"--",pl:null},worstLoss:{symbol:"--",pl:null},coverage:cov}));
    p.rankings=Object.fromEntries(Object.keys(p.rankings).map(key=>[key,{winners:[],losers:[]}]));
    // The nested preview is from this synthetic source population as well.
    p.strategy_history.rows=p.strategy_history.rows.filter((row:Obj)=>row.combined_inclusion !== "INCLUDED");
    expect(combinedOverview(p)).not.toBeNull();const html=markup(source(p));expect(html).toContain("unresolved records may change the ranking");expect(html).not.toContain("$0.00");expect(html).not.toContain("0.00%");
  });
  it("keeps mobile cards, contained tables and theme styling for every current section",()=>{
    const html=markup();for(const value of ["jbravo-panel-cyan","jbravo-panel-violet","jbravo-panel-emerald","xl:grid-cols-3 xl:items-stretch","order-2 xl:order-none",'data-combined-layout="mobile"','data-strategy-layout="mobile"',"max-w-full overflow-auto sm:block","dark:bg-slate-900/95"])expect(html).toContain(value);
    expect(html).not.toContain("<svg");
  });
  it("leaves qualified overview and account equity consumers on their own contracts",()=>{
    const dashboard=readFileSync("src/screens/DashboardHealth.tsx","utf8"),account=readFileSync("src/components/account/AccountTab.tsx","utf8");
    expect(dashboard).toContain('useSource("/api/trades/overview")');expect(dashboard).toContain('performance_status === "VALIDATED_PERFORMANCE"');
    expect(dashboard).not.toContain(COMBINED_GROSS_PATH);expect(account).not.toContain(COMBINED_GROSS_PATH);
    const tab=readFileSync("src/components/trades/TradesTab.tsx","utf8");
    expect(tab.match(/useSource\(/g)).toHaveLength(1);expect(tab).not.toContain('"/api/trades/stats');expect(tab).not.toContain('"/api/trades/history');
  });
});

describe("actual TradesTab coherent controls and read lifecycle",()=>{
  it("uses one request for all sections and reuses server ranking/filter results",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    const fetcher=vi.fn().mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(fixture()))));vi.stubGlobal("fetch",fetcher);
    mount();await act(settle);expect(fetcher).toHaveBeenCalledTimes(1);expect(fetcher.mock.calls[0][0]).toBe(COMBINED_GROSS_PATH);
    for(const label of ["D","W","M","Y","ALL","Losers","Winners","Losers"]){click(label);expect(fetcher).toHaveBeenCalledTimes(1);}
    const leader=container!.querySelector('[aria-label="Leaderboard"]')!;expect(leader.textContent).toContain("TOP GROSS LOSERS");
    for(const row of wire.rankings.all.losers)expect(leader.textContent).toContain(row.symbol);
    for(const row of wire.rankings.all.winners)if(!wire.rankings.all.losers.some(l=>l.symbol===row.symbol))expect(leader.textContent).not.toContain(row.symbol);
  });
  it("paginates the same immutable population without refetch and resets on a new cohort",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    const p=combinedGrossPaginationFixture();
    expect(combinedOverview(p)).not.toBeNull();
    const fresh=emptyFixture();fresh.cohort_id="c".repeat(64);
    const fetcher=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(p))).mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(fresh))));vi.stubGlobal("fetch",fetcher);
    mount();await act(settle);click("Next history page");expect(container!.textContent).toContain("Page 2");expect(container!.textContent).toContain("PAGE24");expect(fetcher).toHaveBeenCalledTimes(1);
    click("Previous history page");expect(container!.textContent).toContain("Page 1");click("Next history page");
    await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});expect(container!.textContent).toContain("Page 1");expect(container!.textContent).not.toContain("PAGE24");expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it("first-read failure and a later accounting hold remain unknown rather than retained success",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    const held={...fixture(),status:"held",performance_status:"HELD_UNVALIDATED_PERFORMANCE"};
    const fetcher=vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(new Response(JSON.stringify(fixture()))).mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(held))));vi.stubGlobal("fetch",fetcher);
    mount();await act(settle);expect(container!.textContent).toContain("combined count and amounts are unknown");expect(container!.textContent).not.toContain("$0.00");
    await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});expect(container!.textContent).toContain("Original Recorded P/L:");
    await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});expect(container!.textContent).toContain("Combined gross unavailable");expect(container!.textContent).not.toContain("Original Recorded P/L:");
  });
  it("does not publish obsolete or canceled reads after unmount and remount",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    let finish!:(response:Response)=>void;let signal:AbortSignal|undefined;
    const fetcher=vi.fn().mockImplementationOnce((_url,init)=>{signal=init.signal;return new Promise<Response>(resolve=>finish=resolve);}).mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(emptyFixture()))));vi.stubGlobal("fetch",fetcher);
    mount();await act(settle);act(()=>root!.unmount());root=undefined;expect(signal?.aborted).toBe(true);
    await act(async()=>{finish(new Response(JSON.stringify(fixture())));await settle();});container!.remove();mount();await act(settle);
    expect(container!.textContent).toContain("No realized trade candidates");expect(container!.textContent).not.toContain("Original Recorded P/L:");
  });
  it("keeps pending refresh coherent, marks timeout stale and recovers every section together",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    const fetcher=vi.fn().mockImplementationOnce(()=>Promise.resolve(new Response(JSON.stringify(fixture()))))
      .mockImplementationOnce((_url,init)=>new Promise((_resolve,reject)=>init.signal.addEventListener("abort",()=>reject(new Error("timeout")))))
      .mockImplementation(()=>{const p=emptyFixture();p.observed_at=new Date().toISOString();p.strategy_history.observed_at=p.observed_at;return Promise.resolve(new Response(JSON.stringify(p)));});vi.stubGlobal("fetch",fetcher);mount();await act(settle);
    await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});expect(container!.textContent).toContain("Original Recorded P/L");expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async()=>{await vi.advanceTimersByTimeAsync(45001);});expect(container!.textContent).toContain("Showing one retained cohort");
    await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});expect(container!.textContent).toContain("No realized trade candidates");expect(container!.textContent).not.toContain("Original Recorded P/L:");
  });
  it("hides all old financial results after malformed or unavailable HTTP200",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
    const fetcher=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(fixture()))).mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(unavailable("population_overflow")))));vi.stubGlobal("fetch",fetcher);
    mount();await act(settle);expect(container!.textContent).toContain("Original Recorded P/L");await act(async()=>{await vi.advanceTimersByTimeAsync(20000);});
    expect(container!.textContent).toContain("Combined gross unavailable");expect(container!.textContent).not.toContain("Original Recorded P/L:");click("Losers");expect(container!.textContent).toContain("Leaderboard unavailable");
  });
});

describe("combined cohort narrowly scoped retention",()=>{
  it.each(["source_busy","source_timeout"])("retains only a prior complete-contract whole observation for %s",async reason=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(wire.observed_at));const events:SourceResult[]=[];
    const failure=unavailable(reason);expect(combinedGrossReadPolicy.retainUnavailable(failure)).toBe(true);
    const fetcher=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(fixture()))).mockResolvedValueOnce(new Response(JSON.stringify(failure)));
    const stop=watchSource(COMBINED_GROSS_PATH,r=>events.push(r),20000,fetcher,60000,45000,combinedGrossReadPolicy);await settle();const first=events.at(-1)!;
    await vi.advanceTimersByTimeAsync(20000);await settle();const last=events.at(-1)!;
    expect(last.freshness).toBe("stale");expect(last.payload).toBe(first.payload);expect(last.receivedAt).toBe(first.receivedAt);expect(last.refreshStatus).toBe("unavailable");expect(markup(last)).toContain("Showing one retained cohort");stop();
  });
  it.each(["population_overflow","cohort_changed","combined_gross_unavailable","paper_context_unavailable","invalid_account_identity","database_unavailable"])("never treats %s as transient retained success",reason=>expect(combinedGrossReadPolicy.retainUnavailable(unavailable(reason))).toBe(false));
  it.each(["source","schema_version","fees","population_complete","rows","cohort_id","strategy_history","failure_stage","extra"])("refuses changed transient envelope field %s",field=>{
    const p=unavailable();p[field]="unexpected";expect(combinedGrossReadPolicy.retainUnavailable(p)).toBe(false);
  });
});
