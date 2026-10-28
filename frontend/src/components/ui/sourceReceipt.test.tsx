import { afterEach, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { watchSource, type SourceResult } from "./sourceState";
import SourceNotice from "./SourceNotice";
afterEach(()=>vi.useRealTimers());
it("tracks actual successful receipt separately from historical source time and retains it after a failed refresh",async()=>{
  vi.useFakeTimers();vi.setSystemTime(new Date("2026-10-25T20:57:55Z"));
  const events:SourceResult[]=[];
  const payload={ok:false,performance_status:"HELD_UNVALIDATED_PERFORMANCE",policy_validation_status:"VALIDATED_POLICY",source:"postgres",source_as_of:"2026-04-16T00:00:00Z"};
  const fetcher=vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(payload),{status:200})).mockRejectedValueOnce(new Error("offline"));
  const stop=watchSource("/stats",value=>events.push(value),20000,fetcher);
  for(let i=0;i<12;i++)await Promise.resolve();
  const first=events[events.length-1];expect(first.receivedAt).toBe("2026-10-25T20:57:55.000Z");
  await vi.advanceTimersByTimeAsync(20000);
  const last=events[events.length-1];expect(last.receivedAt).toBe(first.receivedAt);expect(last.freshness).toBe("stale");expect(last.kind).toBe("held");
  const html=renderToStaticMarkup(<SourceNotice label="Performance" {...last}/>);
  expect(html).toContain("Last successful read (UTC): 2026-10-25T20:57:55.000Z");expect(html).toContain("Source data as of (UTC): 2026-04-16");
  expect(html).not.toContain("Observed (UTC): unknown");stop();
});
