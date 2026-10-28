import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import fixture from "../src/components/trades/registeredApiFixture.json";
import TradesTab, { physicalHistoryRows } from "../src/components/trades/TradesTab";
import * as sources from "../src/components/ui/sourceState";
import type { SourceResult, SourcePayload } from "../src/components/ui/sourceState";

function initial() {
  const history: SourceResult={kind:"ready",validity:"ready",freshness:"current",receivedAt:fixture.received_at,
    payload:structuredClone(fixture.history) as SourcePayload};
  const stats: SourceResult={kind:"held",validity:"held",freshness:"current",receivedAt:fixture.received_at,
    payload:structuredClone(fixture.stats) as SourcePayload};
  return {history,stats};
}
function render(history: SourceResult,stats: SourceResult) {
  const leaders: SourceResult={...stats,payload:{ok:false,source:"postgres",performance_status:"HELD_UNVALIDATED_PERFORMANCE",
    policy_validation_status:"VALIDATED_POLICY",rows:[]}};
  const spy=vi.spyOn(sources,"useSource").mockImplementation(path=>path.includes("/history")?history:path.includes("/stats")?stats:leaders);
  try{return renderToStaticMarkup(<TradesTab/>);}finally{spy.mockRestore();}
}
const response=(payload: SourcePayload)=>({ok:true,json:async()=>structuredClone(payload)}) as Response;
const settle=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
afterEach(()=>{vi.restoreAllMocks();vi.useRealTimers();});

describe("full registered history/stats watcher to actual Trades tab",()=>{
  it("shows scoped net in original table while all global performance states remain held",()=>{
    const f=initial();const html=render(f.history,f.stats);
    expect(html).toContain("observed-cost net -$63.39");expect(html).toContain("Allocated fees $0.01");
    expect(html).toContain("Performance held; accounting qualification is not established.");
    expect(html).toContain("Leaderboard held; performance is not established.");
    expect(html).not.toContain("No leaderboard trades yet.");expect(html).not.toContain("Audited fill calculations");
  });
  it.each(["history_failure","stats_failure","expired","revoked"])("invalidates prior scoped net after %s, keeping physical history",async failure=>{
    vi.useFakeTimers();vi.setSystemTime(new Date(fixture.received_at));
    const f=initial();let history=f.history,stats=f.stats;
    const historyTransport=vi.fn().mockImplementation(async()=>response(f.history.payload!));
    const statsTransport=vi.fn().mockImplementation(async()=>response(f.stats.payload!));
    const stopH=sources.watchSource("/api/trades/history",value=>{history=value;},20000,historyTransport);
    const stopS=sources.watchSource("/api/trades/stats",value=>{stats=value;},20000,statsTransport);
    try {
      await settle();expect(render(history,stats)).toContain("observed-cost net -$63.39");
      if(failure==="history_failure")historyTransport.mockRejectedValue(new Error("synthetic_transport_unavailable"));
      if(failure==="stats_failure")statsTransport.mockRejectedValue(new Error("synthetic_transport_unavailable"));
      if(failure==="revoked"){
        const next=structuredClone(f.history.payload!);
        delete (next.rows as Record<string,unknown>[])[0].derived_accounting;
        next.derived_read_context={status:"NOT_ACTIVE"};
        historyTransport.mockImplementation(async()=>response(next));
      }
      await vi.advanceTimersByTimeAsync(failure==="expired"?61000:20000);await settle();
      const html=render(history,stats);
      expect(physicalHistoryRows(history,stats)).toHaveLength(1);
      expect(html).toContain("CRC");expect(html).not.toContain("observed-cost net -$63.39");
      expect(html).toContain("Performance held");expect(html).not.toContain("No recorded ledger trades in this page.");
    }finally{stopH();stopS();}
  });
});

describe("missing physical source fields",()=>{
  it.each(["entry_time","exit_time","qty","entry_price","exit_price","realized_pnl"])("keeps absent %s unknown without dropping the trade",key=>{
    const f=initial();const row=(f.history.payload!.rows as Record<string,unknown>[])[0];delete row[key];
    const mapped=physicalHistoryRows(f.history,f.stats);expect(mapped).toHaveLength(1);
    const keys: Record<string,string>={entry_time:"buyDate",exit_time:"sellDate",qty:"totalShares",entry_price:"avgEntryPrice",exit_price:"priceSold",realized_pnl:"totalPL"};
    expect((mapped[0] as unknown as Record<string,unknown>)[keys[key]]).toBeNull();
    expect(mapped[0].derivedObservedNet).toBeNull();
  });
  it.each([null,"","   ",undefined])("keeps an identifiable ledger row with missing symbol %s visible as unknown",symbol=>{
    const f=initial();(f.history.payload!.rows as Record<string,unknown>[])[0].symbol=symbol;
    const html=render(f.history,f.stats);
    expect(html).toContain("#110");expect(html).toContain("UNKNOWN");
    expect(html).toContain("#110 | Symbol unknown");
    expect(html).not.toContain("No recorded ledger trades in this page.");
    expect(html).not.toContain("observed-cost net");
  });
  it.each([null,undefined,"",0,-1,1.5,Number.MAX_SAFE_INTEGER+1,true])("does not turn malformed identifier %s into a successful empty ledger",id=>{
    const f=initial();(f.history.payload!.rows as Record<string,unknown>[])[0].trade_id=id;
    const html=render(f.history,f.stats);
    expect(html).not.toContain("No recorded ledger trades in this page.");
    expect(html).toContain("invalid identifier");expect(html).toContain("PARTIAL");
  });
  it("keeps known rows while qualifying additional undisplayable records",()=>{
    const f=initial();(f.history.payload!.rows as Record<string,unknown>[]).push({trade_id:null,symbol:"SYNTHETIC"});
    const html=render(f.history,f.stats);expect(html).toContain("CRC");expect(html).toContain("observed-cost net -$63.39");
    expect(html).toContain("1 returned history record has an invalid identifier");expect(html).toContain("PARTIAL");
  });
  it("preserves genuine empty versus unavailable messages",()=>{
    const f=initial();f.history.payload!.rows=[];
    expect(render(f.history,f.stats)).toContain("No recorded ledger trades in this page.");
    f.history.kind="unavailable";f.history.validity="unavailable";
    const html=render(f.history,f.stats);expect(html).toContain("History unavailable; ledger count is unknown.");
    expect(html).not.toContain("No recorded ledger trades in this page.");
  });
});
