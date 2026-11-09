// Local synthetic visual QA. No API, provider, database or broker reads.
import { useState } from "react";
import { createRoot } from "react-dom/client";
import "../src/styles/tailwind.css";
import CombinedTradesView from "../src/components/trades/CombinedTradesView";
import type { SourceResult } from "../src/components/ui/sourceState";
import { combinedGrossFixture, combinedGrossEmptyFixture, combinedGrossPaginationFixture } from "./fixtures/combinedGrossFixture";
function Preview(){
  const [scenario,setScenario]=useState("populated");
  const payload=scenario === "empty" ? combinedGrossEmptyFixture() : scenario === "paginated" ? combinedGrossPaginationFixture() : combinedGrossFixture();
  if(scenario === "malformed") payload.schema_version="unrecognized";
  const result:SourceResult={payload,kind:scenario === "loading" ? "loading" : scenario === "unavailable" ? "unavailable" : scenario === "stale" ? "stale" : payload.status,validity:payload.status,freshness:scenario === "stale" ? "stale" : "current",receivedAt:payload.observed_at,
    ...(scenario === "stale" ? {refreshStatus:"unavailable",lastAttemptAt:"2026-11-09T20:00:00Z"} : {})};
  return <main className="min-h-screen bg-slate-50 p-3 text-slate-900 dark:bg-slate-950 dark:text-slate-100 sm:p-6"><div className="mx-auto max-w-[1240px] space-y-4">
    <div className="flex flex-wrap items-center gap-3 text-sm"><strong>Synthetic QA only</strong><button type="button" onClick={()=>document.documentElement.classList.toggle("dark")}>Toggle theme</button>
      <label>Scenario <select value={scenario} onChange={event=>setScenario(event.target.value)} className="bg-transparent">{["populated","paginated","empty","unavailable","loading","stale","malformed"].map(value=><option key={value}>{value}</option>)}</select></label></div>
    {scenario === "paginated" ? <p className="text-xs">QA-only pagination extension: 25 extra synthetic excluded records; eligible gross values are unchanged.</p> : null}
    <CombinedTradesView result={result}/>
  </div></main>;
}
createRoot(document.getElementById("root")!).render(<Preview/>);
