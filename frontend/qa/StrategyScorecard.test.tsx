import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { StrategyScorecardContent } from "../src/components/screener/StrategyScorecard";
import { strategyScorecardPresentation, STRATEGY_SCORECARD_PATH, type ScorecardSource } from "../src/components/screener/strategyScorecardRows";
import { sourceKind, watchSource, type SourceKind, type SourceResult } from "../src/components/ui/sourceState";
import generatedFixture from "./strategyScorecardFixture.json";

// Synthetic API-contract example. These values are not the user's strategy results.
function row(): Record<string, any> {
  return {strategy_id:"test_strategy_v1",strategy_name:"Synthetic strategy",strategy_version:"1",definition_sha256:"a".repeat(64),episode_id:"b".repeat(64),
    as_of:"2026-11-06",scope:"DECLARED_EPISODE_SIMULATED_PORTFOLIO_SIGNALS",evidence_status:"VERIFIED",sample_status:"OBSERVED_SAMPLE_NOT_QUALIFICATION",
    closed_signals:4,open_signals:1,unfilled_signals:2,wins:2,losses:1,breakeven:1,win_rate_pct:50,net_pnl:60,expectancy_net_per_signal:15,
    average_win:40,average_loss:-20,payoff_ratio:2,profit_factor:4,profit_factor_status:"DEFINED",
    costs:{basis:"MODELED",fees:1.5,slippage:2.5,total:4,actual_costs_available:false},
    uncertainty:{method:"SIGNAL_WEEK_CLUSTER_BOOTSTRAP",level:0.95,low:-20,high:40,signal_weeks:3,samples:400,status:"DESCRIPTIVE_NOT_VALIDATION"},
    evidence:{archive_count:2,distinct_signals:7,duplicate_observations:3},
    episode:{test_start:"2026-01-01",as_of:"2026-11-06",universe_kind:"FIXED_COHORT",cost_case:{id:"base",fee_per_share:0.01},declared_symbols:10},
  };
}
function payload(): Record<string, any> {
  return {ok:true,status:"COMPLETE",source:"postgres",metric_basis:"STRATEGY_RESEARCH_SIMULATION",rows:[row()],rows_final:1,
    limitations:["Synthetic research test only."],actual_paper:{status:"UNAVAILABLE",net_metrics_available:false,reason:"Actual costs have not been fully reconciled."},
    coverage:{selection:"LATEST_COMPLETED_ARCHIVE_PER_CONFIGURED_STRATEGY",max_groups:24,truncated:false,rejected_archives:0,reasons:[],verified_groups:1,unavailable_groups:0},
    research_qualified:false,order_submission_authorized:false};
}
const view = (data: Record<string, unknown> | null, kind: SourceKind = "ready", freshness?: ScorecardSource["freshness"]) => strategyScorecardPresentation({payload:data,kind,freshness});
const html = (data: Record<string, unknown> | null, kind: SourceKind = "ready", freshness?: ScorecardSource["freshness"]) => renderToStaticMarkup(<StrategyScorecardContent result={{payload:data,kind,freshness}}/>);

describe("strategy-wide scorecard evidence", () => {
  it("describes the latest configured-strategy snapshots and separates response caching from historical evidence", () => {
    expect(generatedFixture.coverage.selection).toBe("LATEST_COMPLETED_ARCHIVE_PER_CONFIGURED_STRATEGY");
    const markup=html(generatedFixture);
    for(const text of ["Latest archived snapshot for each configured strategy", "historical evidence, not verified current deployment", "Response generated: 11/08/2026 00:00:00 UTC", "reported cache age / TTL: 0s / 900s", "does not mean the archive was checked again"]) expect(markup).toContain(text);
    expect(markup).not.toContain("One row per strategy version");
  });
  it("retains the original generated time when a new page fetch returns an older cached response", () => {
    const data={...generatedFixture,cache_age_seconds:120.375};
    const output=view(data);expect(output.response.generated_at).toBe("2026-11-08T00:00:00.000Z");
    expect(output.response.cache_age_seconds).toBe(120.375);expect(html(data)).toContain("120.375s / 900s");
    expect(html(data)).toContain("Archive as of: 2026-11-06 (2 calendar days old)");
  });
  it.each([
    {generated_at:null,cache_age_seconds:null,cache_ttl_seconds:null},
    {generated_at:"2026-11-08T00:00:00",cache_age_seconds:true,cache_ttl_seconds:false},
    {generated_at:"2026-02-30T00:00:00Z",cache_age_seconds:-1,cache_ttl_seconds:Infinity},
    {generated_at:"bad",cache_age_seconds:" ",cache_ttl_seconds:[]},
  ])("keeps invalid or missing response/cache metadata unknown: %j", patch => {
    const data={...generatedFixture,...patch},output=view(data);
    expect(output.response).toEqual({generated_at:"",cache_age_seconds:null,cache_ttl_seconds:null});
    expect(html(data)).toContain("Response generated: unknown; reported cache age / TTL: unknown / unknown");
    expect(html(data)).not.toContain("1970");
  });
  it("renders the initial api_payload refresh-without-cache shape as pending, not completed empty", () => {
    const data={ok:false,status:"UNAVAILABLE",rows:[],rows_final:0,failure_code:"refresh_in_progress",cache_age_seconds:null,research_qualified:false,order_submission_authorized:false};
    const output=view(data,sourceKind(data),"current"),markup=html(data,sourceKind(data),"current");
    expect(output.initialRefresh).toBe(true);expect(output.sync).toBe("loading");expect(output.error).toBeNull();
    expect(markup).toContain("REFRESHING");expect(markup).toContain("Evidence refresh in progress. No snapshot is available yet.");
    expect(markup).not.toContain("No strategy episodes are available");expect(markup).not.toContain("$0.00");
    expect(view(data,"unavailable","stale").initialRefresh).toBe(false);
  });
  it("labels the api_payload expired-cache refresh shape as retained with its original generation time", () => {
    const data={...generatedFixture,status:"PARTIAL",cache_age_seconds:901.25,refresh_in_progress:true,stale:true};
    expect(sourceKind(data)).toBe("stale");
    const output=view(data,sourceKind(data),"stale"),markup=html(data,sourceKind(data),"stale");
    expect(output.retainedRefresh).toBe(true);expect(output.sync).toBe("loading");expect(output.rows).toHaveLength(2);
    expect(markup).toContain("RETAINED EVIDENCE");expect(markup).toContain("The source reported a refresh in progress; refresh completion is unverified");
    expect(markup).toContain("Response generated: 11/08/2026 00:00:00 UTC");expect(markup).toContain("901.25s / 900s");expect(markup).toContain("33.33%");
    expect(html(data,"ready","current")).toContain("RETAINED EVIDENCE");
  });
  it("does not relabel a backend-stale response as fresh just because HTTP succeeded", () => {
    const data={...generatedFixture,stale:true};
    const output=view(data,"ready","current");expect(output.rows).toHaveLength(0);expect(output.error).toContain("stale");
    expect(html(data,"ready","current")).not.toContain("33.33%");
  });
  it("retains metadata only when the expired cached response was already unavailable", () => {
    const data={...generatedFixture,ok:false,status:"PARTIAL",cache_age_seconds:61,cache_ttl_seconds:60,stale:true,refresh_in_progress:true,
      rows:generatedFixture.rows.filter(row=>row.evidence_status==="UNAVAILABLE")};
    const output=view(data,sourceKind(data),"stale");expect(output.retainedRefresh).toBe(true);expect(output.rows[0].closed_signals).toBeNull();
    expect(html(data,sourceKind(data),"stale")).toContain("RETAINED EVIDENCE");expect(html(data,sourceKind(data),"stale")).not.toContain("33.33%");
  });
  it("allows the bounded one-minute refresh without expiring at the instant polling starts", async () => {
    vi.useFakeTimers();const events:SourceResult[]=[];let complete!: (value:Response)=>void;
    const pending=new Promise<Response>(resolve=>{complete=resolve;});
    const transport=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(generatedFixture))).mockReturnValueOnce(pending);
    const stop=watchSource(STRATEGY_SCORECARD_PATH,value=>events.push(value),60_000,transport,120_000,35_000);
    try {
      await vi.advanceTimersByTimeAsync(0);expect(events.at(-1)?.kind).toBe("ready");
      await vi.advanceTimersByTimeAsync(85_000);expect(transport).toHaveBeenCalledTimes(2);expect(events.at(-1)?.kind).toBe("ready");
      complete(new Response(JSON.stringify(generatedFixture)));await vi.advanceTimersByTimeAsync(0);
      expect(events.at(-1)?.kind).toBe("ready");
      expect(strategyScorecardPresentation(events.at(-1)!).rows.find(row=>row.evidence_status==="VERIFIED")?.freshness.evidence_as_of).toBe("2026-11-06");
    } finally {stop();vi.useRealTimers();}
  });
  it("renders the exact backend-generated envelope, including unknown-version configured strategies", () => {
    const output=view(generatedFixture);expect(output.error).toBeNull();expect(output.status).toBe("PARTIAL");expect(output.rows).toHaveLength(2);
    const verified=output.rows.find(row=>row.evidence_status==="VERIFIED")!;
    expect(verified.closed_signals).toBe(3);expect(verified.win_rate_pct).toBeCloseTo(100/3);expect(verified.expectancy).toBe(0);
    expect(verified.costs.total).toBe(18);expect(verified.uncertainty.low).toBe(-40);
    const missing=output.rows.find(row=>row.evidence_status==="UNAVAILABLE")!;expect(missing.closed_signals).toBeNull();expect(missing.strategy_version).toBe("");
    const markup=html(generatedFixture);for(const text of ["33.33%","95% descriptive interval","version unknown","Strategy-wide performance unavailable","completed archive unavailable","Archive as of: 2026-11-06 (2 calendar days old)","Latest attempt: not evaluated","Warning: few signal week clusters. Uncertainty may be unstable."]) expect(markup).toContain(text);
    for(const text of ["Capital: $100,000.00; allocation: 5%; max positions: 4; top N: 3", "Modeled fee / fill: $1.00; slippage: 5.00 bps", "Coverage: complete; evaluated: 5; excluded: 2; discovered: --"]) expect(markup).toContain(text);
  });
  it("retains missing enabled strategies with unknown version and dates rather than fabricating an episode", () => {
    const data=payload();data.rows=[{...row(),evidence_status:"UNAVAILABLE",strategy_version:null,definition_sha256:null,episode_id:null,as_of:null,episode:null}];
    const output=view(data);expect(output.error).toBeNull();expect(output.rows[0].closed_signals).toBeNull();
    const markup=html(data);expect(markup).toContain("version unknown");expect(markup).toContain("Archive as of: unknown");expect(markup).toContain("Latest attempt: unknown");
    expect(markup).not.toContain("vnull");
  });
  it("shows retained archive age separately from the unknown latest source attempt", () => {
    const data=payload();data.rows[0].freshness={basis:"LATEST_COMPLETED_ARCHIVE_ONLY",evidence_as_of:"2026-11-06",source_age_calendar_days:2,latest_attempt_status:"NOT_EVALUATED",latest_attempt_at:null,retained_historical_evidence:true};
    const markup=html(data);expect(markup).toContain("Archive as of: 2026-11-06 (2 calendar days old)");expect(markup).toContain("Latest attempt: not evaluated");expect(markup).not.toContain("current archive");
  });
  it("displays exact distinct completed counts, net metrics, separate payoff and profit factor, costs and uncertainty", () => {
    const markup = html(payload());
    for (const text of ["Synthetic strategy v1", "verified ledger", "W / L / BE", "2 / 1 / 1", "50%", "+$15.00", "Net P&amp;L +$60.00", "$40.00 / -$20.00", "2.00", "4.00", "Fees $1.50", "Slippage $2.50", "Actual: unknown", "-$20.00 to $40.00", "95% descriptive interval", "signal week cluster bootstrap", "Window: 2026-01-01 to 2026-11-06", "cost case: base", "Actual paper performance: unavailable", "Actual costs have not been fully reconciled", "not research qualification"]) expect(markup).toContain(text);
    expect(markup).not.toContain("75%"); expect(markup).not.toContain("PASS"); expect(markup).not.toContain("FAIL");
    expect(view(payload()).rows[0].evidence.duplicate_observations).toBe(3);
  });
  it("does not average, sum, or merge overlapping episodes, versions or strategy definitions", () => {
    const data=payload(); data.rows.push({...row(),strategy_id:"second_strategy_v1",episode_id:"c".repeat(64),win_rate_pct:0.5,expectancy_net_per_signal:-35});
    data.rows.push({...row(),strategy_id:"third_strategy_v2",strategy_version:"2",definition_sha256:"d".repeat(64),win_rate_pct:30});
    const output=view(data); expect(output.rows).toHaveLength(3); expect(output.rows.map(r=>r.win_rate_pct)).toEqual([50,0.5,30]);
    const markup=html(data); expect(markup).toContain("must not be summed or averaged"); expect(markup).toContain("0.5%");
    expect(markup).not.toContain("26.83%");
  });
  it("rejects duplicate episode identities rather than rendering repeat statistics", () => {
    const data=payload(); data.rows.push(row()); const output=view(data);
    expect(output.error).toContain("Duplicate strategy episodes"); expect(output.rows).toHaveLength(0);
  });
  it("rejects multiple historical episodes for the same configured strategy under the latest-only contract", () => {
    const data=payload();data.rows.push({...row(),episode_id:"c".repeat(64)});
    expect(view(data).error).toContain("Duplicate configured strategy snapshots");expect(view(data).rows).toHaveLength(0);
  });
  it.each(["UNAVAILABLE","CONFLICT","PARTIAL","UNRECOGNIZED"])("suppresses numerical performance on %s rows even if stale numeric values leak in", evidence_status => {
    const data=payload(); data.rows[0].evidence_status=evidence_status; data.status="PARTIAL";
    const projected=view(data).rows[0];
    for (const field of ["closed_signals","wins","losses","breakeven","win_rate_pct","net_pnl","expectancy","profit_factor","payoff_ratio"] as const) expect(projected[field]).toBeNull();
    expect(projected.costs.total).toBeNull(); expect(projected.uncertainty.low).toBeNull();
    const markup=html(data); expect(markup).toContain("Strategy-wide performance unavailable"); expect(markup).not.toContain("+$15.00"); expect(markup).not.toContain("50%");
  });
  it("renders informative unknown-only PostgreSQL rows when no complete lifecycle can be verified", () => {
    const data=payload(); data.ok=false;data.status="UNAVAILABLE";data.rows[0].evidence_status="UNAVAILABLE";data.rows[0].failure_code="lifecycle_binding_unavailable";
    const output=view(data,"unavailable","current"); expect(output.rows).toHaveLength(1);expect(output.sync).toBe("error");
    const markup=html(data,"unavailable","current");
    for (const text of ["Synthetic strategy v1", "Evidence limitation: lifecycle binding unavailable", "Strategy-wide performance unavailable", 'role="alert"']) expect(markup).toContain(text);
    expect(markup).not.toContain("+$15.00"); expect(markup).not.toContain("No strategy episodes are available");
    expect(view(data,"unavailable","stale").rows).toHaveLength(0);
  });
  it("keeps verified simulation separate from unavailable actual paper accounting in mixed evidence", () => {
    const data=payload(); data.status="PARTIAL";data.rows.push({...row(),strategy_id:"unavailable_strategy",episode_id:"c".repeat(64),evidence_status:"UNAVAILABLE",failure_code:"provider_budget_exceeded"});
    const output=view(data);expect(output.rows[0].expectancy).toBe(15);expect(output.rows[1].expectancy).toBeNull();
    const markup=html(data);expect(markup).toContain("PARTIAL EVIDENCE");expect(markup).toContain("Actual paper performance: unavailable");expect(markup).toContain("provider budget exceeded");
  });
  it("keeps verified empty sample totals zero but undefined rates, expectancy, ratios and intervals unknown", () => {
    const data=payload();Object.assign(data.rows[0],{closed_signals:0,wins:0,losses:0,breakeven:0,net_pnl:0,sample_status:"NO_CLOSED_SIGNALS",profit_factor_status:"NO_CLOSED_SIGNALS"});
    data.rows[0].costs={basis:"MODELED",fees:0,slippage:0,total:0};
    const output=view(data).rows[0];expect(output.closed_signals).toBe(0);expect(output.net_pnl).toBe(0);expect(output.costs.total).toBe(0);
    for(const field of ["expectancy","win_rate_pct","payoff_ratio","profit_factor","average_win","average_loss"] as const) expect(output[field]).toBeNull();
    expect(output.uncertainty.low).toBeNull();const markup=html(data);expect(markup).toContain("Insufficient evidence: 0 completed signals");expect(markup).not.toContain(">0%<");
  });
  it("distinguishes all-loss zero win rate and zero profit factor from undefined all-win profit factor", () => {
    const data=payload();Object.assign(data.rows[0],{wins:0,losses:4,breakeven:0,win_rate_pct:0,profit_factor:0,payoff_ratio:null,average_win:null});
    expect(view(data).rows[0].win_rate_pct).toBe(0);expect(view(data).rows[0].profit_factor).toBe(0);expect(html(data)).toContain("0%");
    Object.assign(data.rows[0],{wins:4,losses:0,win_rate_pct:100,profit_factor:null,profit_factor_status:"UNDEFINED_NO_LOSSES"});
    expect(view(data).rows[0].profit_factor).toBeNull();expect(html(data)).toContain("undefined no losses");expect(html(data)).not.toContain("Infinity");
  });
  it("accepts numeric API strings without interpreting fractional percentages as ratios", () => {
    const data=payload();Object.assign(data.rows[0],{closed_signals:"200",win_rate_pct:"0.5",net_pnl:"0",expectancy_net_per_signal:"-0.002",profit_factor:"1.02"});
    expect(view(data).rows[0].closed_signals).toBe(200);expect(view(data).rows[0].net_pnl).toBe(0);expect(html(data)).toContain("0.5%");expect(html(data)).toContain("1.02");
  });
  it("does not turn booleans, blanks, arrays, or nonfinite values into sample counts or dollars", () => {
    const data=payload();Object.assign(data.rows[0],{closed_signals:true,net_pnl:" ",wins:[],losses:false,breakeven:Infinity});data.rows[0].costs.total={};
    const output=view(data).rows[0];expect(output.closed_signals).toBeNull();expect(output.wins).toBeNull();expect(output.net_pnl).toBeNull();expect(output.costs.total).toBeNull();expect(output.expectancy).toBeNull();
  });
  it.each([{status:"INSUFFICIENT_SIGNAL_WEEKS"},{level:true},{low:Infinity},{low:50,high:40},{level:95},{low:null}])("keeps invalid or insufficient uncertainty unknown: %j", patch => {
    const data=payload();Object.assign(data.rows[0].uncertainty,patch);expect(view(data).rows[0].uncertainty.low).toBeNull();expect(html(data)).not.toContain("95% descriptive interval");
  });
  it("discloses bounded coverage and rejected archives instead of implying exhaustive strategy history", () => {
    const data=payload();Object.assign(data.coverage,{truncated:true,rejected_archives:2,reasons:["archive_size_limit"]});
    const markup=html(data);expect(markup).toContain("Limited to 24 groups; more history is not shown");expect(markup).toContain("Rejected archives: 2");expect(markup).toContain("archive size limit");
  });
  it.each(["held","unavailable","stale"] as const)("suppresses older metrics on %s", kind => {
    const output=view(payload(),kind),markup=html(payload(),kind);expect(output.sync).toBe("error");expect(output.rows).toHaveLength(0);expect(markup).not.toContain("+$15.00");expect(markup).not.toContain("No strategy episodes are available");
  });
  it("separates loading, genuine empty response, and an unsupported contract", () => {
    expect(html(null,"loading")).toContain("Loading strategy-wide evidence");
    const data=payload();data.rows=[];data.status="EMPTY";expect(html(data,"empty")).toContain("This is not a zero-performance result");
    data.metric_basis="ACTUAL_PAPER";expect(view(data).error).toBeTruthy();expect(html(data)).not.toContain("No strategy episodes are available");
  });
  it("requires identities before presenting verified metrics", () => {
    for (const key of ["strategy_id","strategy_version","definition_sha256","episode_id"]) {
      const data=payload();delete data.rows[0][key];expect(view(data).error).toBeTruthy();expect(view(data).rows).toHaveLength(0);
    }
  });
  it("preserves theme tokens and contained responsive tables with no global CSS or action controls", () => {
    const markup=html(payload());for (const token of ["jbravo-panel-emerald","jbravo-panel-inner-emerald","min-w-0","overflow-x-auto","flex-wrap","sm:p-md","break-all"]) expect(markup).toContain(token);
    expect(markup).not.toContain("<style");expect(markup).not.toContain("Execute");expect(STRATEGY_SCORECARD_PATH).toBe("/api/strategies/scorecard");
    const source=readFileSync("src/components/screener/StrategyScorecard.tsx","utf8");expect(source).toContain("60_000, 35_000, 120_000");
    const tab=readFileSync("src/components/screener/ScreenerTab.tsx","utf8");expect(tab.indexOf("<StrategyScorecard")).toBeGreaterThan(tab.indexOf("<MetricsResultsCard"));expect(tab.indexOf("<StrategyScorecard")).toBeLessThan(tab.indexOf("<LogsBoard"));
  });
});
