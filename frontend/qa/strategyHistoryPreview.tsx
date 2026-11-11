// Standalone local visual fixture. No API, database or broker reads.
import { useState } from "react";
import { createRoot } from "react-dom/client";
import "../src/styles/tailwind.css";
import StrategyHistoryCard from "../src/components/trades/StrategyHistoryCard";
import type { SourceResult } from "../src/components/ui/sourceState";
import { strategyHistoryFixture, completeStrategySignal } from "./fixtures/strategyHistoryFixture";

function Preview() {
  const [state,setState]=useState("populated");
  const fixture=strategyHistoryFixture();
  const clone=(letter:string,symbol:string,fields:Record<string,unknown>)=>({...completeStrategySignal,signal_id:letter.repeat(64),symbol,...fields});
  const rows=[clone("a","SYNTH",{}),clone("b","PART",{state:"OPEN",status:"partial",completion_status:"PARTIAL_EXIT_OBSERVED",gross_realized_pnl:null,exited_qty:16,remaining_qty:24}),
    clone("c","OPEN",{state:"OPEN",status:"partial",completion_status:"NO_EXIT_OBSERVED",gross_realized_pnl:null,exited_qty:0,remaining_qty:40,exit_avg_price:null,exit_time:null}),
    clone("d","WAIT",{state:"SUBMITTED",status:"partial",completion_status:"NO_ENTRY_FILL_OBSERVED",reason:"entry_not_filled",entry_qty:0,exited_qty:0,remaining_qty:0,entry_avg_price:null,exit_avg_price:null,entry_time:null,exit_time:null,gross_realized_pnl:null}),
    clone("e","EXT",{state:"RECOGNIZED_EXTERNAL_ENTRY",status:"incomplete",ownership_scope:"ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS",ownership_basis:"STORED_SIGNAL_PAYLOAD_NOT_REVALIDATED",strategy_id:"a_long_strategy_identifier_".repeat(4),reason:"order_evidence_missing"})];
  const result:SourceResult={payload:{...fixture,rows:state==="empty"?[]:rows,status:state==="empty"?"empty":"partial",has_more:true},kind:state==="loading"?"loading":state==="unavailable"?"unavailable":state==="stale"?"stale":"ready",validity:"ready",freshness:state==="stale"?"stale":"current",receivedAt:"2026-11-09T16:00:01Z"};
  return <main className="min-h-screen bg-slate-50 p-3 text-slate-900 dark:bg-slate-950 dark:text-slate-100 sm:p-6">
    <div className="mx-auto max-w-[1240px] space-y-4">
      <div className="flex flex-wrap gap-3 text-sm"><strong>Synthetic QA only</strong><button onClick={()=>document.documentElement.classList.toggle("dark")}>Toggle theme</button>
        <label>Scenario <select value={state} onChange={event=>setState(event.target.value)} className="bg-transparent">{["populated","empty","unavailable","loading","stale"].map(value=><option key={value}>{value}</option>)}</select></label></div>
      <StrategyHistoryCard result={result}/>
    </div>
  </main>;
}
createRoot(document.getElementById("root")!).render(<Preview/>);
