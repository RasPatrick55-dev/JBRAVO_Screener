// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TradesTab from "../src/components/trades/TradesTab";
import { STRATEGY_HISTORY_PATH } from "../src/components/trades/strategyHistory";
import { strategyHistoryFixture } from "./fixtures/strategyHistoryFixture";
import gross from "./executionGrossFixture.json";

// Separate, wholly fictional legacy row exercises the no-merge boundary.
const legacy = {
  history: { ok:true, status:"ready", source:"postgres", rows:[{
    trade_id:999001, symbol:"LEDGER", status:"CLOSED", qty:40, entry_price:10, exit_price:12,
    realized_pnl:80, entry_time:"2026-10-01T10:00:00Z", exit_time:"2026-10-05T10:00:00Z",
  }] },
  stats: { ok:false, status:"held", source:"postgres", performance_status:"HELD_UNVALIDATED_PERFORMANCE", rows:[] },
};

const response = (payload: unknown) => new Response(JSON.stringify(payload));
const settle = async () => { for (let i=0;i<80;i++) await Promise.resolve(); };
let container: HTMLDivElement, root: Root;
beforeEach(() => {
  vi.useFakeTimers(); vi.setSystemTime(new Date("2026-11-09T16:00:00Z")); vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT",true);
  container=document.createElement("div"); document.body.append(container); root=createRoot(container);
});
afterEach(async () => { await act(async()=>{root.unmount();await settle();});container.remove();vi.useRealTimers();vi.unstubAllGlobals(); });
const render=async(node:React.ReactNode)=>(await act(async()=>{root.render(node);await settle();}));
const advance=async(ms:number)=>(await act(async()=>{await vi.advanceTimersByTimeAsync(ms);await settle();}));
const panel=()=>container.querySelector('section[aria-label="Strategy trade history"]')!;
function fetcher(strategy:(init:RequestInit)=>Promise<Response>) {
  return vi.fn((path:string,init:RequestInit)=>path===STRATEGY_HISTORY_PATH?strategy(init):Promise.resolve(response(
    path.includes("/history")?legacy.history:path.includes("/leaderboard")?gross.winners:path.includes("basis=execution_gross")?gross.stats:legacy.stats)));
}

describe("strategy history independent mounted lifecycle",()=>{
  it("adds one canonical read while preserving separate ledger rows and P/L sources",async()=>{
    const fetch=fetcher(async()=>response(strategyHistoryFixture()));vi.stubGlobal("fetch",fetch);
    await render(<TradesTab/>);
    expect(fetch.mock.calls.filter(([path])=>path===STRATEGY_HISTORY_PATH)).toHaveLength(1);
    expect(panel().textContent).toContain("SYNTH");
    const legacyTable=container.querySelector('section[aria-label="Latest trades"]')!;
    expect(legacyTable.textContent).toContain("LEDGER");expect(legacyTable.textContent).not.toContain("SYNTH");
    expect(legacyTable.textContent).toContain("Recorded P/L");expect(panel().textContent).not.toContain("LEDGER");
  });
  it("keeps the four-read concurrency ceiling including the new history read",async()=>{
    const reads:{path:string,init:RequestInit,finish:(value:Response)=>void}[]=[];
    vi.stubGlobal("fetch",vi.fn((path:string,init:RequestInit)=>new Promise<Response>(finish=>reads.push({path,init,finish}))));
    await render(<TradesTab/>);expect(reads).toHaveLength(4);expect(reads.some(read=>read.path===STRATEGY_HISTORY_PATH)).toBe(false);
    await act(async()=>{reads[0].finish(response(legacy.stats));await settle();});
    expect(reads).toHaveLength(5);expect(reads[4].path).toBe(STRATEGY_HISTORY_PATH);
    await render(<p>Another screen</p>);for(const read of reads.slice(1))expect(read.init.signal?.aborted).toBe(true);
  });
  it("ends a first-read timeout at 45s without hiding independent ledger content",async()=>{
    const fetch=fetcher(()=>new Promise(()=>{}));vi.stubGlobal("fetch",fetch);await render(<TradesTab/>);
    await advance(44999);expect(panel().textContent).toContain("Loading stored");await advance(1);
    expect(panel().textContent).toContain("financial results are unknown");expect(panel().textContent).not.toContain("Loading stored");
    expect(container.querySelector('section[aria-label="Latest trades"]')?.textContent).toContain("LEDGER");
    const call=fetch.mock.calls.find(([path])=>path===STRATEGY_HISTORY_PATH)!;expect(call[1].signal?.aborted).toBe(true);
  });
  it("uses completion-based polling and keeps the original receipt when a refresh fails",async()=>{
    let calls=0;const fetch=fetcher(()=>++calls===1?new Promise(resolve=>setTimeout(()=>resolve(response(strategyHistoryFixture())),27000)):Promise.reject(new Error("offline")));
    vi.stubGlobal("fetch",fetch);await render(<TradesTab/>);await advance(20000);expect(calls).toBe(1);
    await advance(7000);expect(panel().textContent).toContain("SYNTH");expect(panel().textContent).toContain("16:00:27 UTC");
    await advance(19999);expect(calls).toBe(1);await advance(1);expect(calls).toBe(2);
    expect(panel().textContent).toContain("stale strategy read");expect(panel().textContent).toContain("16:00:27 UTC");expect(panel().textContent).toContain("SYNTH");
  });
  it("replaces old completed financials when the source now reports partial or incomplete evidence",async()=>{
    let calls=0;vi.stubGlobal("fetch",fetcher(async()=>{
      const fixture=strategyHistoryFixture();fixture.observed_at=new Date().toISOString();
      if(calls++>0) Object.assign(fixture.rows[0],{status:"partial",state:"OPEN",completion_status:"PARTIAL_EXIT_OBSERVED",gross_realized_pnl:null,exited_qty:16,remaining_qty:24});
      return response(fixture);
    }));
    await render(<TradesTab/>);expect(panel().textContent).toContain("$80.00");await advance(20000);
    expect(panel().textContent).toContain("Partial exit observed");expect(panel().textContent).toContain("OPEN");expect(panel().textContent).not.toContain("$80.00");
  });
  it("clears old rows on a business failure rather than presenting an empty success",async()=>{
    let calls=0;vi.stubGlobal("fetch",fetcher(async()=>response(calls++?{ok:false,status:"unavailable",rows:[]}:strategyHistoryFixture())));
    await render(<TradesTab/>);expect(panel().textContent).toContain("SYNTH");await advance(20000);
    expect(panel().textContent).not.toContain("SYNTH");expect(panel().textContent).toContain("financial results are unknown");expect(panel().textContent).not.toContain("Showing 0");
  });
  it("aborts navigation and ignores a late completed-history response",async()=>{
    let finish!:(value:Response)=>void;const fetch=fetcher(()=>new Promise(resolve=>{finish=resolve;}));vi.stubGlobal("fetch",fetch);
    await render(<TradesTab/>);await render(<p>Another screen</p>);
    expect(fetch.mock.calls.find(([path])=>path===STRATEGY_HISTORY_PATH)![1].signal?.aborted).toBe(true);
    await act(async()=>{finish(response(strategyHistoryFixture()));await settle();});expect(container.textContent).toBe("Another screen");
  });
});
