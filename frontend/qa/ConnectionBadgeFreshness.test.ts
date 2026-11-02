import { afterEach, describe, expect, it, vi } from "vitest";
import { parseTradingStatus, buildNavbarBadges, watchTradingStatus, type ApiHealthResponse } from "../src/components/navbar/liveStatus";

const NOW=Date.parse("2026-10-27T22:00:00Z");
function payload(age=30): ApiHealthResponse {
  return {freshness:{age_seconds:72000,freshness_level:"gray"},trading_ok:false,
    connection_check:{source:"connection_health.json",status:"fresh",trading_ok:true,
      checked_at:new Date(NOW-age*1000).toISOString(),freshness:{age_seconds:age,freshness_level:"green"}}};
}
const response=(data:unknown)=>({ok:true,json:async()=>data}) as Response;
const settle=async()=>{for(let i=0;i<16;i++)await Promise.resolve();};
afterEach(()=>{vi.useRealTimers();vi.restoreAllMocks();});

describe("connection-check badge evidence",()=>{
  it("accepts a fresh stored connection despite old metrics and independently held page data",()=>{
    expect(parseTradingStatus(payload(),NOW)).toBe(true);
    expect(buildNavbarBadges(true,"error")?.[1]).toMatchObject({label:"Stored API check",tone:"success",showDot:true});
  });
  it("does not let fresh metrics promote an old connection",()=>{
    const value=payload(72000);value.freshness={age_seconds:30,freshness_level:"green"};
    expect(parseTradingStatus(value,NOW)).toBeNull();
    // Even inconsistent server-green metadata cannot override its old timestamp.
    value.connection_check!.freshness={age_seconds:1,freshness_level:"green"};
    expect(parseTradingStatus(value,NOW)).toBeNull();
  });
  it.each(["missing","invalid","naive","invalid_date","future","unknown_source","missing_result","stale_status","legacy_only"])("keeps %s unknown",kind=>{
    const value=payload();
    if(kind==="legacy_only")delete value.connection_check;
    else {
      const check=value.connection_check!;
      if(kind==="missing")delete check.checked_at;
      if(kind==="invalid")check.checked_at="invalid";
      if(kind==="naive")check.checked_at="2026-10-27T21:59:30";
      if(kind==="invalid_date")check.checked_at="2026-02-30T21:59:30Z";
      if(kind==="future")check.checked_at=new Date(NOW+1).toISOString();
      if(kind==="unknown_source")check.source="unverified";
      if(kind==="missing_result")delete check.trading_ok;
      if(kind==="stale_status")check.status="stale";
    }
    expect(parseTradingStatus(value,NOW)).toBeNull();
  });
  it("retains a fresh actual failed check as failed rather than unknown or success",()=>{
    const value=payload();value.connection_check!.trading_ok=false;
    expect(parseTradingStatus(value,NOW)).toBe(false);
    expect(buildNavbarBadges(false)?.[1]).toMatchObject({tone:"error",showDot:false});
  });
});

describe("bounded connection polling and expiry",()=>{
  it.each(["request","body"])("bounds a hung %s and ignores late completion",async phase=>{
    vi.useFakeTimers();vi.setSystemTime(NOW);
    let finish:((value:Response|ApiHealthResponse)=>void)|undefined;
    const pending=new Promise<Response|ApiHealthResponse>(resolve=>{finish=resolve;});
    const transport=vi.fn<typeof fetch>().mockImplementation(async()=>phase==="request"?await pending as Response:{ok:true,json:()=>pending} as Response);
    const events:(boolean|null)[]=[];const stop=watchTradingStatus(value=>events.push(value),transport);
    try {
      await settle();await vi.advanceTimersByTimeAsync(5001);await settle();
      expect(events[events.length-1]).toBeNull();expect(transport.mock.calls[0][1]?.signal?.aborted).toBe(true);
      finish!(phase==="request"?response(payload()):payload());await settle();
      expect(events).not.toContain(true);
    }finally{stop();}
  });
  it("expires a successful stored check without needing the next poll",async()=>{
    vi.useFakeTimers();vi.setSystemTime(NOW);
    const events:(boolean|null)[]=[];const transport=vi.fn<typeof fetch>().mockResolvedValue(response(payload(7199)));
    const stop=watchTradingStatus(value=>events.push(value),transport);
    try {
      await settle();expect(events[events.length-1]).toBe(true);
      await vi.advanceTimersByTimeAsync(1001);expect(events[events.length-1]).toBeNull();expect(transport).toHaveBeenCalledTimes(1);
    }finally{stop();}
  });
  it("failed poll clears prior success; unmount stops publication",async()=>{
    vi.useFakeTimers();vi.setSystemTime(NOW);
    const events:(boolean|null)[]=[];
    const transport=vi.fn<typeof fetch>().mockResolvedValueOnce(response(payload())).mockRejectedValue(new Error("synthetic failure"));
    const stop=watchTradingStatus(value=>events.push(value),transport);
    await settle();expect(events[events.length-1]).toBe(true);
    await vi.advanceTimersByTimeAsync(15000);expect(events[events.length-1]).toBeNull();
    stop();const count=events.length;await vi.advanceTimersByTimeAsync(20000);expect(events).toHaveLength(count);
  });
});
