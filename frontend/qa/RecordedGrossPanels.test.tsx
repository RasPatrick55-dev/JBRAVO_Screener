import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import TradesTab from "../src/components/trades/TradesTab";
import * as sources from "../src/components/ui/sourceState";
import { normalizeRecordedGrossStats, normalizeRecordedGrossLeaders, recordedGrossContract } from "../src/components/trades/recordedGrossData";
import { buildLeaderboardRequestPath } from "../src/components/trades/leaderboardData";
import type { SourcePayload, SourceResult } from "../src/components/ui/sourceState";
import fixture from "./recordedGrossFixture.json";

function render(stats:SourcePayload,leaders:SourcePayload,leaderResult?:SourceResult) {
  const source=(payload:SourcePayload):SourceResult=>({payload,kind:sources.sourceKind(payload),validity:sources.sourceValidity(payload),freshness:"current",receivedAt:"2026-10-27T22:00:00Z"});
  const requested:string[]=[];
  vi.spyOn(sources,"useSource").mockImplementation(path=>{
    requested.push(path);
    if(path.includes("/history"))return source({ok:true,source:"postgres",rows:[]});
    if(path.includes("/leaderboard"))return leaderResult ?? source(leaders);
    if(path.includes("basis=recorded_gross"))return source(stats);
    return source({ok:false,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"VALIDATED_POLICY",rows:[]});
  });
  return {html:renderToStaticMarkup(<TradesTab/>),requested};
}
afterEach(()=>{vi.restoreAllMocks();vi.useRealTimers();});

describe("retained gross coverage through freshness transitions",()=>{
  const settle=async()=>{for(let i=0;i<16;i++)await Promise.resolve();};
  for(const scenario of ["all-unknown","partial-empty","genuine-empty"] as const) {
    it.each(["rejection","timeout","expiry"] as const)(`${scenario} remains correctly qualified after %s`,async transition=>{
      vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-27T22:00:00Z"));
      const payload:SourcePayload=structuredClone(scenario==="genuine-empty"?fixture.empty_leaders:fixture.unknown_leaders);
      if(scenario==="partial-empty") {
        payload.coverage={included:2,excluded:0,unknown:1,reasons:{EXIT_PRICE_UNKNOWN_OR_INVALID:1}};
        payload.closed_population_count=3;
      }
      const transport=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(payload),{status:200}));
      if(transition==="rejection")transport.mockRejectedValue(new Error("offline"));
      else transport.mockImplementation(()=>new Promise<Response>(()=>{}));
      let result:SourceResult|undefined;
      const stop=sources.watchSource("/gross-leaders",next=>{result=next;},20000,transport);
      try {
        await settle();
        expect(result?.freshness).toBe("current");
        await vi.advanceTimersByTimeAsync(transition==="rejection"?20000:transition==="timeout"?30001:60001);
        expect(result?.kind).toBe("stale");expect(result?.freshness).toBe("stale");
        expect(result?.validity).toBe(scenario==="genuine-empty"?"empty":"partial");
        expect(result?.payload).toEqual(payload);
        const {html}=render(fixture.stats,payload,result);
        expect(html).toContain("Recorded gross leaderboard STALE (stale read)");
        if(scenario==="genuine-empty") {
          expect(html).toContain("No positive gross results in the selected period.");
          expect(html).not.toContain("unresolved records may change it");
        } else {
          expect(html).toContain("No known gross results in this ranking; unresolved records may change it.");
          expect(html).not.toContain("No positive gross results in the selected period.");
          expect(html).not.toContain("No negative gross results in the selected period.");
          expect(html).toContain("1 unknown");
        }
      } finally {stop();}
    });
  }
});

describe("actual gross wire to original cards",()=>{
  it("populates both red-boxed cards from whole-population gross while qualified net stays held",()=>{
    const {html,requested}=render(fixture.stats,fixture.winners);
    expect(html).toContain("Recorded gross performance READY");expect(html).toContain("5 included / 1 policy-excluded / 0 unknown");
    expect(html).toContain("Known Gross P/L");expect(html).toContain("-$51.38");expect(html).toContain("40.00%");
    expect(html).toContain("Best Gross Trade");expect(html).toContain("Worst Gross Trade");expect(html).toContain("TOP GROSS WINNERS");
    expect(html).toContain("+$7.00");expect(html).toContain("+$5.00");
    expect(html).toContain("Qualified net: Performance held; accounting qualification is not established.");
    expect(html).toContain("Before fees");expect(html).not.toContain("Audited fill calculations");
    // The physical page is empty in this fixture; neither card sums the page.
    expect(html).toContain("No recorded ledger trades in this page.");
    expect(requested).toContain("/api/trades/stats?range=all");expect(requested).toContain("/api/trades/stats?range=all&basis=recorded_gross");
    expect(requested).toContain("/api/trades/leaderboard?range=all&mode=winners&limit=12&basis=recorded_gross");
    expect(html).toContain("xl:grid-cols-3");expect(html).toContain("order-3");expect(html).toContain("jbravo-panel-cyan");expect(html).toContain("jbravo-panel-violet");
  });
  it("shows known subtotal and explicit partial coverage instead of full performance",()=>{
    const {html}=render(fixture.partial,fixture.winners);
    expect(html).toContain("Recorded gross performance PARTIAL");expect(html).toContain("5 included / 1 policy-excluded / 1 unknown");
    expect(html).toContain("1 missing/invalid exit price");
    expect(html).toContain("-$51.38");expect(html).toContain("1 unknown");
  });
  it("all unknown stays unknown, not successful zero/empty",()=>{
    const {html}=render(fixture.unknown,fixture.unknown_leaders);
    expect(html).toContain("0 included / 0 policy-excluded / 1 unknown");expect(html).toContain("unresolved records may change it");
    expect(html).not.toContain("$0.00");expect(html).not.toContain("No leaderboard trades yet.");
  });
  it("true empty has zero total, unknown win rate and no invented best trade",()=>{
    const rows=normalizeRecordedGrossStats(fixture.empty);
    expect(rows[0].totalPL).toBe(0);expect(rows[0].winRatePct).toBeNull();expect(rows[0].topTrade.pl).toBeNull();
    const {html}=render(fixture.empty,fixture.empty_leaders);
    expect(html).toContain("$0.00");expect(html).toContain("No positive gross results in the selected period.");
  });
  it.each(["basis","net","policy","source"])("rejects wrong %s contract rather than relabeling net",damage=>{
    const payload=structuredClone(fixture.stats) as SourcePayload;
    if(damage==="basis")payload.pnl_basis="VALIDATED_NET";
    if(damage==="net")payload.net_eligible=true;
    if(damage==="policy")payload.policy_validation_status="UNAVAILABLE";
    if(damage==="source")payload.source="csv";
    expect(recordedGrossContract(payload)).toBe(false);expect(normalizeRecordedGrossStats(payload)).toEqual([]);
    const {html}=render(payload,fixture.empty_leaders);expect(html).toContain("Recorded gross inputs unavailable");
  });
  it("keeps counts strict and malformed amounts unknown",()=>{
    const data=structuredClone(fixture.stats) as unknown as SourcePayload;
    const rows=data.rows as Record<string,unknown>[];rows[0].totalPL=" ";
    expect(normalizeRecordedGrossStats(data)[0].totalPL).toBeNull();
    (rows[0].coverage as Record<string,unknown>).included=false;expect(normalizeRecordedGrossStats(data)).toHaveLength(4);
    const leaders=structuredClone(fixture.winners) as unknown as SourcePayload;
    (leaders.rows as Record<string,unknown>[])[0].pl=" ";expect(normalizeRecordedGrossLeaders(leaders).rejected).toBe(1);
  });
  it("range/mode request remains explicit gross and legacy builder stays compatible",()=>{
    expect(buildLeaderboardRequestPath("m","losers",12,"recorded_gross")).toContain("mode=losers&limit=12&basis=recorded_gross");
    expect(buildLeaderboardRequestPath("m","losers",12)).not.toContain("basis");
  });
  it.each(["rows","coverage","closed_population_count"])("fails closed on absent %s",key=>{
    const data=structuredClone(fixture.stats) as unknown as SourcePayload;delete data[key];
    expect(recordedGrossContract(data)).toBe(false);
    expect(render(data,fixture.empty_leaders).html).toContain("Recorded gross inputs unavailable");
  });
});
