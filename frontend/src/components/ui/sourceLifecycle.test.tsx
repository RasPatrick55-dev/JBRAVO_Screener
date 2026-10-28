import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { watchSource, canDisplay, type SourceResult } from "./sourceState";
import { computePipelineScore, pipelineStatusFromScore } from "../../screens/DashboardHealth";
import MonitoringPositionsCard from "../dashboard/MonitoringPositionsCard";
import { parseTradingStatus, buildNavbarBadges } from "../navbar/liveStatus";
import AccountBreakdownCard from "../account/AccountBreakdownCard";
import AccountPerformanceCard from "../account/AccountPerformanceCard";
import { fetchNoStoreJson } from "../screener/utils";
const response = (payload: unknown) => new Response(JSON.stringify(payload), {status: 200});
const settle = async () => { for (let i=0;i<12;i++) await Promise.resolve(); };
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });
describe("async source boundaries", () => {
  it("clears prior metrics after held HTTP200", async () => {
    vi.useFakeTimers(); const events: SourceResult[] = [];
    const transport = vi.fn().mockResolvedValueOnce(response({ok:true, rows:[{totalPL:999}]})).mockResolvedValueOnce(response({ok:false, performance_status:"HELD_UNVALIDATED_PERFORMANCE",rows:[]}));
    const stop = watchSource("/stats", result=>events.push(result), 20000, transport);
    await settle(); expect(events[events.length-1].kind).toBe("ready");
    await vi.advanceTimersByTimeAsync(20000); await settle();
    expect(events[events.length-1].kind).toBe("held"); expect(events[events.length-1].payload?.rows).toEqual([]); stop();
  });
  it("suppresses obsolete request generation", async () => {
    let resolve!: (value:Response)=>void; const events:SourceResult[]=[];
    const pending=new Promise<Response>(done=>{resolve=done;});
    const stop=watchSource("/old",value=>events.push(value),20000,vi.fn().mockReturnValue(pending));
    stop(); const count=events.length; resolve(response({ok:true,rows:[{trade_id:8}]})); await settle(); expect(events).toHaveLength(count);
    const current:SourceResult[]=[]; const stopNew=watchSource("/new",value=>current.push(value),20000,vi.fn().mockResolvedValue(response({ok:true,rows:[]})));
    await settle(); expect(current[current.length-1].kind).toBe("empty"); stopNew();
  });
  it("logs deadline does not erase broker marks", async () => {
    vi.useFakeTimers(); const positions:SourceResult[]=[],logs:SourceResult[]=[];
    const stopPositions=watchSource("/positions",v=>positions.push(v),20000,vi.fn().mockResolvedValue(response({ok:true,positions:[{symbol:"AAA",currentPrice:13.5}]})));
    const logFetch:typeof fetch=(_path,init)=>new Promise<Response>((_resolve,reject)=>init?.signal?.addEventListener("abort",()=>reject(new Error("deadline"))));
    const stopLogs=watchSource("/logs",v=>logs.push(v),20000,logFetch);
    await settle(); await vi.advanceTimersByTimeAsync(10001);
    expect(logs[logs.length-1].kind).toBe("unavailable"); expect(positions[positions.length-1].kind).toBe("ready");
    expect(positions[positions.length-1].payload?.positions).toEqual([{symbol:"AAA",currentPrice:13.5}]); stopPositions();stopLogs();
  });
  it("marks retained data stale on failed polling",async()=>{
    vi.useFakeTimers();const events:SourceResult[]=[];
    const transport=vi.fn().mockResolvedValueOnce(response({ok:true,positions:[{symbol:"AAA"}]})).mockRejectedValueOnce(new Error("offline"));
    const stop=watchSource("/positions",v=>events.push(v),20000,transport);await settle();await vi.advanceTimersByTimeAsync(20000);
    expect(events[events.length-1].kind).toBe("stale");stop();
  });
});
describe("account, screener and health",()=>{
  it("does not fabricate account zero amounts or allocations",()=>{
    const html=renderToStaticMarkup(<><AccountBreakdownCard summary={null}/><AccountPerformanceCard rows={[{period:"Daily",netChangePct:null,netChangeUsd:null}]} total={{equity:null,netChangePct:null,netChangeUsd:null}}/></>);
    expect(html).toContain("--");expect(html).not.toContain("$0.00");expect(html).not.toContain("0/0");
  });
  it("rejects business failure behind HTTP200",async()=>{
    vi.stubGlobal("fetch",vi.fn().mockResolvedValue(response({ok:false,rows:[]})));
    await expect(fetchNoStoreJson("/api/screener/picks")).rejects.toThrow("PostgreSQL data unavailable");
  });
  it("missing or stale checks cannot produce full health",()=>{
    expect(computePipelineScore({pipeline_rc:0,trading_ok:true,data_ok:true})).toBeNull();
    expect(computePipelineScore({pipeline_rc:0,trading_ok:true,data_ok:true,freshness:{freshness_level:"gray",age_seconds:99999}})).toBeNull();
    expect(computePipelineScore({pipeline_rc:0,trading_ok:true,freshness:{freshness_level:"green",age_seconds:1}})).toBeNull();
    const score=computePipelineScore({pipeline_rc:0,trading_ok:true,data_ok:true,freshness:{freshness_level:"green",age_seconds:1}});
    expect(score).toBe(100);expect(pipelineStatusFromScore(score).label).toBe("Computational/API checks passed");
  });
});


describe("overview broker exposure and saved connection labels", () => {
  it("unavailable exposure is unknown instead of confirmed zero", () => {
    const html = renderToStaticMarkup(<MonitoringPositionsCard positions={[]} available={false} sourceStatus="unavailable"/>);
    expect(html).toContain("EXPOSURE UNKNOWN");
    expect(html).toContain("Broker position count and exposure are unknown");
    expect(html).not.toContain("0 OPEN"); expect(html).not.toContain("$0.00");
  });
  it("successful broker empty is distinct from unavailable", () => {
    const html = renderToStaticMarkup(<MonitoringPositionsCard positions={[]} available sourceStatus="empty"/>);
    expect(html).toContain("0 OPEN"); expect(html).toContain("No open broker positions"); expect(html).toContain("$0.00");
  });
  it("partial P/L or missing cost basis never creates a complete portfolio aggregate", () => {
    const html = renderToStaticMarkup(<MonitoringPositionsCard positions={[
      {symbol:"AAA",currentPrice:13.5,percentPL:10,dollarPL:4,costBasis:40,sparklineData:[]},
      {symbol:"BBB",currentPrice:NaN,percentPL:NaN,dollarPL:NaN,sparklineData:[]}
    ]} available sourceStatus="partial" observedAt="2026-10-25T16:00:00Z"/>);
    expect(html).toContain("$13.50"); expect(html).toContain("partial");
    const summary = html.slice(html.lastIndexOf("Total P/L"));
    expect(summary).not.toContain("$4.00"); expect(summary).toContain(">--</div>");
    expect(summary).not.toContain("10%");
  });
  it("stale or missing diagnostics cannot label a stored API check green", () => {
    for (const freshness of [undefined, {age_seconds:99999,freshness_level:"gray"}, {age_seconds:-1,freshness_level:"green"}, {age_seconds:NaN,freshness_level:"green"}]) {
      const status = parseTradingStatus({trading_ok:true, freshness});
      expect(status).toBeNull(); expect(buildNavbarBadges(status)?.[1]?.label).toBe("API check unknown/stale");
      expect(buildNavbarBadges(status)?.[1]?.showDot).toBe(false);
    }
    expect(parseTradingStatus({trading_ok:true,freshness:{age_seconds:1,freshness_level:"green"}})).toBeNull();
    expect(parseTradingStatus({connection_check:{source:"connection_health.json",status:"fresh",trading_ok:true,
      checked_at:new Date(Date.now()-1000).toISOString(),freshness:{age_seconds:1,freshness_level:"green"}}})).toBe(true);
    expect(buildNavbarBadges(true)?.[1]?.label).toBe("Stored API check");
  });
});

// Independent review regressions recorded before v2 behavioral fixes.
describe("independent review regressions", () => {
  it.each([50,1,.5,0])("dashboard win rate preserves percent units %s", async value => {
    const {normalizeWinRate}=await import("../../screens/DashboardHealth");
    expect(normalizeWinRate(value)).toBe(value);
  });
  it("backtest loading cannot claim completed",async()=>{
    const {default:Backtest}=await import("../screener/BacktestResultsCard");
    const html=renderToStaticMarkup(<Backtest/>);
    expect(html).not.toContain(">COMPLETE<");
    expect(html).not.toMatch(/>\s*COMPLETE\s*</);
    expect(html).toContain("LOADING");
  });
  it("unavailable open orders cannot assert a genuine empty broker result",async()=>{
    const {default:Orders}=await import("../account/OpenOrdersCard");
    const props={rows:[],isLoading:false,sourceKind:"unavailable"} as any;
    const html=renderToStaticMarkup(<Orders {...props}/>);
    expect(html).not.toContain("No open orders");
    expect(html).toContain("Open order count is unknown");
  });
  it("an expired successful broker snapshot is stale even after HTTP success",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T16:00:00Z"));
    const events:SourceResult[]=[];
    const stop=watchSource("/positions",v=>events.push(v),20000,vi.fn().mockResolvedValue(response({ok:true,observed_at:"2026-10-25T15:50:00Z",positions:[]})));
    try {await settle();expect(events[events.length-1].kind).toBe("stale");} finally {stop();}
  });
});

describe("review financial consumers and async page wiring", () => {
  it("Execute hides operational P/L, held values and failed refreshes but preserves validated zero", async () => {
    const {default:Card,validatedExecutionPL}=await import("../execute/ExecutionSummaryCard");
    const operational={ok:true,result_pl_usd:1234,performance_status:"VALIDATED_PERFORMANCE",result_pl_source:"postgres:executor_runs"};
    expect(validatedExecutionPL(operational)).toBeNull();
    const invalid=renderToStaticMarkup(<Card summary={operational} isLoading={false} hasError={false}/>);
    expect(invalid).not.toContain("1,234");expect(invalid).toContain("Held / unknown");
    const validated={...operational,result_pl_usd:0,result_pl_source:"postgres_validated_performance"};
    expect(validatedExecutionPL(validated)).toBe(0);
    expect(renderToStaticMarkup(<Card summary={validated} isLoading={false} hasError={false}/>)).toContain("$0.00");
    expect(renderToStaticMarkup(<Card summary={validated} isLoading={false} hasError/>)).not.toContain("$0.00");
    const utils=await import("../execute/utils");expect(utils.parseNumber("")).toBeNull();expect(utils.formatNumber(.125)).toBe("0.125");
  });
  it.each(["loading","unavailable","held","stale","ready"] as const)("completion badge does not claim verified completion for %s",async kind=>{
    const {BacktestCompletionBadge:Badge}=await import("../screener/BacktestResultsCard");
    const html=renderToStaticMarkup(<Badge kind={kind} payload={{ok:true,status:"UNKNOWN",completion_status:"unverified"}}/>);
    expect(html).not.toContain("jbravo-status-success");expect(html).not.toMatch(/>COMPLETE</);
    expect(html).toContain(kind==="ready"?"UNVERIFIED":kind.toUpperCase());
  });
  it("open-order failure and empty last-known transitions cannot assert current zero",async()=>{
    vi.useFakeTimers();const {default:Orders}=await import("../account/OpenOrdersCard");
    const events:SourceResult[]=[];
    const transport=vi.fn().mockResolvedValueOnce(response({ok:true,rows:[]})).mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(response({ok:false,rows:[]}));
    const stop=watchSource("/orders",v=>events.push(v),20000,transport);
    try{
      await settle();expect(events[events.length-1].kind).toBe("empty");
      expect(renderToStaticMarkup(<Orders rows={[]} sourceKind="empty"/>)).toContain("No open orders (confirmed broker read)");
      await vi.advanceTimersByTimeAsync(20000);
      expect(events[events.length-1].kind).toBe("stale");
      expect(renderToStaticMarkup(<Orders rows={[]} sourceKind="stale"/>)).not.toContain("No open orders");
      await vi.advanceTimersByTimeAsync(20000);
      expect(events[events.length-1].kind).toBe("unavailable");
      expect(renderToStaticMarkup(<Orders rows={[]} sourceKind="unavailable"/>)).toContain("Open order count is unknown");
    }finally{stop();}
  });
  it("a fresh successful snapshot expires while a subsequent transport ignores abort",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T16:00:00Z"));const events:SourceResult[]=[];
    const transport=vi.fn().mockResolvedValueOnce(response({ok:true,observed_at:"2026-10-25T16:00:00Z",positions:[]})).mockReturnValue(new Promise(()=>{}));
    const stop=watchSource("/positions",v=>events.push(v),20000,transport);
    try{await settle();expect(events[events.length-1].kind).toBe("empty");await vi.advanceTimersByTimeAsync(61000);expect(events[events.length-1].kind).toBe("stale");}finally{stop();}
  });
  it("Dashboard publishes broker positions before unrelated logs finish and marks an expired success stale",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T16:00:00Z"));vi.resetModules();
    const captured:any[]=[],cleanups:(()=>void)[]=[];
    vi.doMock("react",async()=>{
      const actual=await vi.importActual<typeof import("react")>("react");
      return {...actual,useState:(initial:any)=>[typeof initial==="function"?initial():initial,(value:any)=>captured.push(value)],
        useEffect:(effect:()=>void|(()=>void))=>{const cleanup=effect();if(cleanup)cleanups.push(cleanup);},useMemo:(factory:()=>unknown)=>factory()};
    });
    vi.stubGlobal("window",globalThis);
    vi.stubGlobal("fetch",vi.fn((path:string)=>path.includes("logs")||path.includes(".log")?new Promise(()=>{}):Promise.resolve(response(path.includes("positions/monitoring")?{ok:true,observed_at:"2026-10-25T16:00:00Z",positions:[{symbol:"AAA",currentPrice:13.5}]}:{ok:true}))));
    try{
      const {default:Dashboard}=await import("../../screens/DashboardHealth");Dashboard({});
      await settle();
      expect(captured.some(value=>value?.kind==="ready"&&value?.payload?.positions?.[0]?.currentPrice===13.5)).toBe(true);
      await vi.advanceTimersByTimeAsync(61000);
      expect(captured.some(value=>value?.kind==="stale"&&value?.payload?.positions?.[0]?.symbol==="AAA")).toBe(true);
    }finally{cleanups.forEach(cleanup=>cleanup());vi.doUnmock("react");vi.resetModules();}
  });
});

describe("v3 independent review state regressions",()=>{
  it.each(["rejection","timeout"])("combined held performance survives %s without suppressing independent history/exposure",async failure=>{
    vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T16:00:00Z"));
    const combined={ok:true,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",rows:[{trade_id:9}],metrics:{net_pnl:77777},open_positions:{count:999,unrealized_pnl:9999}};
    const transport=vi.fn().mockResolvedValueOnce(response(combined));
    if(failure==="rejection")transport.mockRejectedValueOnce(new Error("offline"));else transport.mockReturnValueOnce(new Promise(()=>{}));
    const held:SourceResult[]=[],history:SourceResult[]=[],broker:SourceResult[]=[];
    const stops=[watchSource("/overview",v=>held.push(v),20000,transport),
      watchSource("/history",v=>history.push(v),20000,vi.fn().mockResolvedValue(response({ok:true,source:"postgres",rows:[{trade_id:9}],performance_status:"NOT_QUALIFIED_PHYSICAL_HISTORY"}))),
      watchSource("/positions",v=>broker.push(v),20000,vi.fn().mockResolvedValue(response({ok:true,source:"alpaca",positions:[{symbol:"AAA"}],summary:{totalOpenPL:42.125}})))];
    try{
      await settle();expect(held[held.length-1].kind).toBe("held");
      await vi.advanceTimersByTimeAsync(failure==="timeout"?30001:20001);
      expect(held[held.length-1].kind).toBe("held");
      expect(held[held.length-1].validity).toBe("held");expect(held[held.length-1].freshness).toBe("stale");
      expect(canDisplay(held[held.length-1].kind)).toBe(false);
      expect(held[held.length-1].payload?.rows).toEqual([{trade_id:9}]);
      expect(canDisplay(history[history.length-1].kind)).toBe(true);
      expect(broker[broker.length-1].payload?.summary).toEqual({totalOpenPL:42.125});
    }finally{stops.forEach(stop=>stop());}
  });
  it("a stale transport flag cannot precede a financial hold",async()=>{
    const {sourceKind}=await import("./sourceState");
    expect(sourceKind({ok:true,status:"stale",performance_status:"HELD_UNVALIDATED_PERFORMANCE"})).toBe("held");
    expect(sourceKind({ok:true,stale:true,performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"UNAVAILABLE"})).toBe("unavailable");
  });
  it.each(["ready","stale"])("actual Dashboard exposure is independent of held overview performance (%s)",async brokerKind=>{
    vi.resetModules();const computed:any[]=[];
    vi.doMock("react",async()=>{
      const actual=await vi.importActual<typeof import("react")>("react");
      return {...actual,useState:(initial:any)=>[typeof initial==="function"?initial():initial,()=>{}],useEffect:()=>{},useMemo:(fn:()=>unknown)=>{const value=fn();computed.push(value);return value;}};
    });
    vi.doMock("./sourceState",async()=>{
      const actual=await vi.importActual<typeof import("./sourceState")>("./sourceState");
      return {...actual,useSource:(path:string)=>path.includes("positions/monitoring")?
        {kind:brokerKind,payload:{ok:true,source:"alpaca",observed_at:"2026-10-25T16:00:00Z",positions:[{symbol:"AAA",qty:1,currentPrice:13.5,dollarPL:21.125},{symbol:"BBB",qty:2,currentPrice:12,dollarPL:21}],summary:{totalOpenPL:42.125}}}:
        path.includes("trades/overview")?{kind:"held",payload:{ok:true,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",metrics:{net_pnl:77777},rows:[{trade_id:9}],open_positions:{count:999,unrealized_pnl:9999}}}:
        {kind:"unavailable",payload:null}};
    });
    try{
      const {default:Dashboard}=await import("../../screens/DashboardHealth");
      const html=renderToStaticMarkup(<Dashboard/>);
      const cards=computed.find(value=>Array.isArray(value)&&value.some(card=>card?.title==="Open Positions"));
      const exposure=cards.find((card:any)=>card.title==="Open Positions");
      expect(exposure.value).toBe("2");expect(exposure.detail).toContain("42.13");
      expect(exposure.footnote).toContain("Alpaca");expect(exposure.footnote).toContain(brokerKind);
      expect(html).toContain("2 OPEN");
      expect(html).toContain("42.13");expect(html).toContain(`Broker positions: ${brokerKind}`);
      expect(html).not.toContain("9,999");expect(html).not.toContain("77,777");
    }finally{vi.doUnmock("react");vi.doUnmock("./sourceState");vi.resetModules();}
  });
});


describe("v3 shared-consumer invalidity preservation",()=>{
  it("policy unavailable remains unavailable after timeout and age expiry even with successful physical-history envelope",async()=>{
    vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T16:00:00Z"));const events:SourceResult[]=[];
    const payload={ok:true,performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"UNAVAILABLE",rows:[{trade_id:9}]};
    const transport=vi.fn().mockResolvedValueOnce(response(payload)).mockReturnValue(new Promise(()=>{}));
    const stop=watchSource("/overview",v=>events.push(v),20000,transport);
    try{
      await settle();expect(events[events.length-1].kind).toBe("unavailable");
      await vi.advanceTimersByTimeAsync(61000);
      const current=events[events.length-1];
      expect(current.kind).toBe("unavailable");expect(current.validity).toBe("unavailable");expect(current.freshness).toBe("stale");
      expect(canDisplay(current.kind)).toBe(false);
      const {default:Notice}=await import("./SourceNotice");const html=renderToStaticMarkup(<Notice label="Performance" {...current}/>);
      expect(html).toContain("business status remains UNAVAILABLE");expect(html).not.toContain("77777");
    }finally{stop();}
  });
});
