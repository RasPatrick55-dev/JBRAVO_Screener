// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePollingRead, refreshNotice } from "../src/components/ui/usePollingRead";
import { useSource } from "../src/components/ui/sourceState";
import LegacyExecuteTab from "./fixtures/LegacyExecuteTab";
import ExecuteTab from "../src/components/execute/ExecuteTab";
import ExecutionDataQualityCard from "../src/components/execute/ExecutionDataQualityCard";
import ScreenerPicksCard from "../src/components/screener/ScreenerPicksCard";
import MetricsResultsCard from "../src/components/screener/MetricsResultsCard";
import LogsPanel from "../src/components/screener/LogsPanel";
import AccountTab from "../src/components/account/AccountTab";
import { useMlOverview } from "../src/components/ml/useMlOverview";
import metrics from "./unifiedMetricsFixture.json";

const response = (payload: unknown) => new Response(JSON.stringify(payload));
const settle = async () => { for (let i=0;i<60;i++) await Promise.resolve(); };
let container: HTMLDivElement, root: Root;
beforeEach(() => {
  vi.useFakeTimers(); vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("EventSource", vi.fn(() => { throw new Error("SSE must not open"); }));
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => { root.unmount(); await settle(); }); container.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); });
const render = async (node: React.ReactNode) => { await act(async () => { root.render(node); await settle(); }); };
const advance = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms); await settle(); }); };
const parse = (value: unknown) => value as {name:string};
function ReadHarness({path}: {path:string}) {
  const state = usePollingRead(path, parse);
  return <><output>{state.isLoading ? "loading" : state.data?.name ?? "unavailable"}</output><p>{refreshNotice(state.error, state.receivedAt)}</p><button onClick={state.refresh}>Refresh</button></>;
}
function SourceHarness({path}: {path:string}) {
  const state = useSource(path);
  return <output>{state.kind}:{String(state.payload?.name ?? "")}</output>;
}

function MlHarness() {
  const state = useMlOverview();
  return <output>{state.isLoading ? "loading" : state.hasError ? "unavailable" : JSON.stringify(state.data)}</output>;
}

describe("one-shot ML reads",()=>{
  it("accepts the observed slow response without changing artifact age or adding polls",async()=>{
    const payload={ok:true,freshness:{status:"stale",artifact_timestamp:"2026-10-25T16:00:00Z",age_seconds:999999,source_authority:"diagnostic_only"}};
    const fetcher=vi.fn(()=>new Promise(resolve=>setTimeout(()=>resolve(response(payload)),24492)));vi.stubGlobal("fetch",fetcher);
    await render(<MlHarness/>);await advance(24000);expect(container.textContent).toBe("loading");
    await advance(492);expect(container.textContent).toBe(JSON.stringify(payload));await advance(200000);expect(fetcher).toHaveBeenCalledOnce();
  });
  it("ends loading at the finite deadline without a retry loop",async()=>{
    const fetcher=vi.fn().mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<MlHarness/>);await advance(45000);expect(container.textContent).toBe("unavailable");expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);
    await advance(200000);expect(fetcher).toHaveBeenCalledOnce();
  });
  it("aborts unmount and ignores a late ML response",async()=>{
    let finish!:(value:Response)=>void;const fetcher=vi.fn(()=>new Promise<Response>(resolve=>{finish=resolve;}));vi.stubGlobal("fetch",fetcher);
    await render(<MlHarness/>);await render(<p>new screen</p>);expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);
    await act(async()=>{finish(response({ok:true}));await settle();});expect(container.textContent).toBe("new screen");
  });
});

describe("mounted request ownership", () => {
  it("does not let an old query response or finally clear the newer loading/error state", async () => {
    const finish = new Map<string,(value:Response)=>void>();
    const fetcher = vi.fn((path:string) => new Promise<Response>(resolve=>finish.set(path,resolve))); vi.stubGlobal("fetch",fetcher);
    await render(<ReadHarness path="/old"/>); await render(<ReadHarness path="/new"/>);
    expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);
    await act(async()=>{finish.get("/old")!(response({name:"obsolete"}));await settle();});
    expect(container.textContent).toContain("loading"); expect(container.textContent).not.toContain("obsolete");
    await act(async()=>{finish.get("/new")!(response({name:"current"}));await settle();});
    expect(container.textContent).toContain("current");
  });
  it("a superseded response cannot clear the current path's error", async()=>{
    let finish!: (value:Response)=>void;
    const fetcher=vi.fn().mockImplementationOnce(()=>new Promise<Response>(resolve=>{finish=resolve;})).mockRejectedValue(new Error("current failed"));vi.stubGlobal("fetch",fetcher);
    await render(<ReadHarness path="/old"/>);await render(<ReadHarness path="/new"/>);
    expect(container.textContent).toContain("current failed");
    await act(async()=>{finish(response({name:"obsolete"}));await settle();});
    expect(container.textContent).toContain("current failed");expect(container.textContent).not.toContain("obsolete");
  });
  it("resets a prior successful useSource snapshot immediately on path change", async()=>{
    const fetcher=vi.fn().mockResolvedValueOnce(response({ok:true,name:"old snapshot"})).mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<SourceHarness path="/old"/>);expect(container.textContent).toBe("ready:old snapshot");
    await render(<SourceHarness path="/new"/>);expect(container.textContent).toBe("loading:");
    expect(fetcher.mock.calls[1][1].signal.aborted).toBe(false);
  });
  it("StrictMode disposal leaves exactly one active request and no stuck loading flag", async()=>{
    const fetcher=vi.fn().mockResolvedValue(response({name:"current"}));vi.stubGlobal("fetch",fetcher);
    await render(<StrictMode><ReadHarness path="/strict"/></StrictMode>);
    expect(fetcher).toHaveBeenCalledOnce();expect(container.textContent).toContain("current");
  });
  it("retains stale data with its receipt on failure and coalesces repeated retries", async()=>{
    const fetcher=vi.fn().mockResolvedValueOnce(response({name:"last sample"})).mockRejectedValueOnce(new Error("offline")).mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<ReadHarness path="/data"/>); await advance(20000);
    expect(container.textContent).toContain("last sample");expect(container.textContent).toContain("Showing stale data from");
    await act(async()=>{for(let i=0;i<20;i++)container.querySelector("button")!.click();await settle();});
    expect(fetcher).toHaveBeenCalledTimes(3);expect(container.textContent).not.toContain("loading");
    expect(container.textContent).toContain("current status is unverified");
  });
});

describe("Execute finite polling and failure display",()=>{
  const goodState={ok:true,summary:{ok:true,candidates:7,submitted:2},orders:{ok:true,rows:[{symbol:"SYN",status:"new",order_id:"synthetic-1"}]},trailing_stops:{ok:true,rows:[]},logs:{execute:{ok:true,rows:[]},monitor:{ok:true,rows:[]},pipeline:{ok:true,rows:[]}}};
  it("uses only one state read, tolerates 27s, then retains stale rows after a failed refresh",async()=>{
    let stateCalls=0;
    const fetcher=vi.fn((path:string)=>path.includes("/task")?Promise.resolve(response({ok:false})):++stateCalls===1?new Promise(resolve=>setTimeout(()=>resolve(response(goodState)),27000)):Promise.reject(new Error("offline")));
    vi.stubGlobal("fetch",fetcher);await render(<ExecuteTab/>);await advance(20000);
    expect(stateCalls).toBe(1);expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
    await advance(7000);expect(container.textContent).toContain("SYN");expect(container.querySelectorAll(".animate-pulse")).toHaveLength(0);
    await advance(20000);expect(stateCalls).toBe(2);expect(container.textContent).toContain("SYN");expect(container.textContent).toContain("Showing stale data from");
    expect(container.textContent).toContain("current protection is unverified");expect(EventSource).not.toHaveBeenCalled();
  });
  it("stops skeletons after initial failure, including while the next poll is pending",async()=>{
    let stateCalls=0;
    vi.stubGlobal("fetch",vi.fn((path:string)=>path.includes("/task")?Promise.resolve(response({ok:false})):++stateCalls===1?Promise.reject(new Error("offline")):new Promise(()=>{})));
    await render(<ExecuteTab/>);expect(container.querySelectorAll(".animate-pulse")).toHaveLength(0);expect(container.textContent).toContain("current order count is unknown");
    await advance(20000);expect(container.querySelectorAll(".animate-pulse")).toHaveLength(0);expect(container.textContent).toContain("unavailable");expect(EventSource).not.toHaveBeenCalled();
  });
  it("audit uses a finite request and never displays a missing count as zero",async()=>{
    const fetcher=vi.fn().mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);await render(<ExecutionDataQualityCard/>);
    expect(container.textContent).toContain("Loading");await advance(45000);
    expect(container.textContent).toContain("Unavailable");expect(container.textContent).not.toContain("Healthy");expect(container.textContent).toContain("--");
    expect(fetcher).toHaveBeenCalledOnce();expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);expect(EventSource).not.toHaveBeenCalled();
  });
});

describe("Screener and account evidence retention",()=>{
  it("slow Picks load successfully, retain exact rows as stale, and disable retries while busy",async()=>{
    const payload={ok:true,rows:[{symbol:"SYN",rank:1,price:123.45}],run_date:"2026-11-09",completion_status:"verified",computation_complete:true,status:"COMPLETE"};
    let calls=0;const fetcher=vi.fn(()=>++calls===1?new Promise(resolve=>setTimeout(()=>resolve(response(payload)),25000)):calls===2?Promise.reject(new Error("offline")):new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<ScreenerPicksCard/>);await advance(15000);expect(container.textContent).toContain("LOADING");await advance(10000);expect(container.textContent).toContain("SYN");
    await advance(20000);expect(container.textContent).toContain("SYN");expect(container.textContent).toContain("STALE");expect(container.textContent).toContain("Computation: unverified");
    await act(async()=>{container.querySelector<HTMLButtonElement>("button")!.click();await settle();});
    expect(container.querySelector<HTMLButtonElement>("button")!.disabled).toBe(true);expect(fetcher).toHaveBeenCalledTimes(3);
  });
  it("Metrics keep validated simulation rows on failed refresh but clear them for a new filter",async()=>{
    const fetcher=vi.fn().mockResolvedValueOnce(response(metrics.populated)).mockRejectedValueOnce(new Error("offline")).mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<MetricsResultsCard/>);expect(container.textContent).toContain("SYN");expect(container.textContent).toContain("$12.34");
    await advance(20000);expect(container.textContent).toContain("$12.34");expect(container.textContent).toContain("Showing stale data");
    const filter=Array.from(container.querySelectorAll("button")).find(button=>button.textContent==="Completed history")!;
    await act(async()=>{filter.click();await settle();});expect(container.textContent).toContain("Loading strategy metrics");expect(container.textContent).not.toContain("$12.34");
  });
  it("log stage changes discard prior receipts and abort obsolete filters",async()=>{
    const fetcher=vi.fn().mockResolvedValueOnce(response({ok:true,source:"postgres",rows:[{message:"old stage",level:"INFO"}],source_receipts:[{source:"old receipt"}]})).mockReturnValue(new Promise(()=>{}));vi.stubGlobal("fetch",fetcher);
    await render(<LogsPanel title="Screener logs" stage="screener"/>);expect(container.textContent).toContain("old stage");
    await render(<LogsPanel title="Backtest logs" stage="backtest"/>);expect(container.textContent).not.toContain("old stage");expect(container.textContent).not.toContain("old receipt");
    await render(<LogsPanel title="Metrics logs" stage="metrics"/>);expect(fetcher.mock.calls[1][1].signal.aborted).toBe(true);
  });
  it("Account publishes summary independently, preserves it as stale, and clears an explicit unavailable envelope",async()=>{
    let summaryCalls=0;
    vi.stubGlobal("fetch",vi.fn((path:string)=>{
      if(path==="/api/account/summary"){
        summaryCalls++;
        if(summaryCalls===1)return Promise.resolve(response({ok:true,source:"alpaca",equity:12345.67,cash:2345.67,buying_power:4000}));
        if(summaryCalls===2)return Promise.reject(new Error("offline"));
        return Promise.resolve(response({ok:false,source:"alpaca",equity:999999}));
      }
      if(path.includes("portfolio_history"))return new Promise(()=>{});
      return Promise.resolve(response({ok:true,rows:[]}));
    }));
    await render(<AccountTab/>);expect(container.textContent).toContain("2,345.67");expect(container.textContent).toContain("Account equity history: LOADING");expect(container.textContent).toContain("Account summary: READY");
    await advance(20000);expect(container.textContent).toContain("2,345.67");expect(container.textContent).toContain("Account summary: STALE");
    await advance(20000);expect(container.textContent).toContain("Account summary: UNAVAILABLE");expect(container.textContent).not.toContain("2,345.67");expect(container.textContent).not.toContain("999,999");
  });
});

// The fixture tests legacy JavaScript's error/fallback route only. It does not
// simulate browser reconnection rules or claim browser QA of HTTP 204.
describe("cached pre-repair Execute compatibility",()=>{
  it("stream error starts finite fallback immediately and leaves the legacy 20s poll usable",async()=>{
    const streams: Array<{onerror:(()=>void)|null; onmessage:unknown;close:()=>void}> = [];
    const close=vi.fn();
    vi.stubGlobal("EventSource",class {
      onerror: (()=>void)|null=null;onmessage: unknown=null;close=close;
      constructor(){streams.push(this);}
    });
    const state={ok:true,summary:{ok:true,candidates:1},orders:{ok:true,rows:[]},trailing_stops:{ok:true,rows:[]},logs:{}};
    const fetcher=vi.fn((path:string)=>Promise.resolve(response(path.includes("/execute/state")?state:{ok:false})));vi.stubGlobal("fetch",fetcher);
    await render(<LegacyExecuteTab/>);expect(streams).toHaveLength(1);
    expect(fetcher.mock.calls.filter(([path])=>String(path).includes("/execute/state"))).toHaveLength(0);
    await act(async()=>{streams[0].onerror!();await settle();});
    expect(fetcher.mock.calls.filter(([path])=>String(path).includes("/execute/state"))).toHaveLength(1);
    await advance(2500);expect(fetcher.mock.calls.filter(([path])=>String(path).includes("/execute/state"))).toHaveLength(1);
    await advance(17500);expect(fetcher.mock.calls.filter(([path])=>String(path).includes("/execute/state"))).toHaveLength(2);
    await act(async()=>{root.render(<></>);await settle();});expect(close).toHaveBeenCalledOnce();
    const count=fetcher.mock.calls.length;await advance(20000);expect(fetcher).toHaveBeenCalledTimes(count);
  });
});
