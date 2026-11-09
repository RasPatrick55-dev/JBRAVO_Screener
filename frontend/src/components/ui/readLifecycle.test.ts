import { afterEach, describe, expect, it, vi } from "vitest";
import { deadlineRead, pollRead, READ_TIMEOUT_MS, MAX_CONCURRENT_READS } from "./readLifecycle";
import { watchSource, type SourceResult } from "./sourceState";
import { watchProviderLogs } from "../dashboard/providerLogs";

const stops: Array<() => void> = [];
const settle = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
const response = (payload: unknown) => new Response(JSON.stringify(payload));
afterEach(async () => { stops.splice(0).forEach(stop => stop()); await settle(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("bounded dashboard reads", () => {
  it.each([10_310, 11_850, 24_490, 26_805, 35_001])("accepts a successful %ims source without premature unavailability", async delay => {
    vi.useFakeTimers(); const events: SourceResult[] = [];
    const transport = vi.fn().mockImplementation(() => new Promise(resolve => setTimeout(() => resolve(response({ok: true, rows: [{symbol: "SYN"}]})), delay)));
    stops.push(watchSource("/slow", value => events.push(value), 20000, transport));
    await vi.advanceTimersByTimeAsync(delay - 1);
    expect(events[events.length - 1]?.kind).toBe("loading"); expect(transport).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(events[events.length - 1]?.kind).toBe("ready"); expect(transport).toHaveBeenCalledTimes(1);
  });
  it("includes hung body decoding in the deadline and rejects a late decoder", async () => {
    vi.useFakeTimers(); const events: SourceResult[] = []; let resolve!: (value: unknown) => void;
    const transport = vi.fn().mockResolvedValue({ok: true, json: () => new Promise(done => { resolve = done; })});
    stops.push(watchSource("/hung-body", value => events.push(value), 20000, transport));
    await settle(); await vi.advanceTimersByTimeAsync(READ_TIMEOUT_MS);
    expect(events[events.length - 1]?.kind).toBe("unavailable"); expect(transport.mock.calls[0][1].signal.aborted).toBe(true);
    const count = events.length; resolve({ok:true,rows:[{symbol:"LATE"}]}); await settle();
    expect(events).toHaveLength(count); expect(transport).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(19999); expect(transport).toHaveBeenCalledTimes(1);
  });
  it("aborts transport and clears all deadline/poll timers on unmount", async () => {
    vi.useFakeTimers(); const events: SourceResult[] = [];
    const transport = vi.fn().mockReturnValue(new Promise(() => {}));
    const stop = watchSource("/unmount", value => events.push(value), 20000, transport); stops.push(stop);
    await settle(); const signal = transport.mock.calls[0][1].signal;
    stop(); await settle(); expect(signal.aborted).toBe(true); expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(200000); expect(events).toHaveLength(1); expect(transport).toHaveBeenCalledTimes(1);
  });
  it("bounds provider cleanup too, without replacing its receipt source", async () => {
    vi.useFakeTimers(); const transport = vi.fn().mockReturnValue(new Promise(() => {}));
    const stop = watchProviderLogs(() => {}, transport); stops.push(stop); await settle();
    stop(); await settle(); expect(transport.mock.calls[0][1].signal.aborted).toBe(true); expect(vi.getTimerCount()).toBe(0);
  });
  it("coalesces manual refresh and polls only a full interval after settlement", async () => {
    vi.useFakeTimers(); let resolve!: (value: string) => void;
    const read = vi.fn().mockImplementation(() => new Promise<string>(done => { resolve = done; }));
    const success = vi.fn(), failure = vi.fn(); const polling = pollRead({read, success, failure}); stops.push(polling.stop);
    await settle(); for (let i=0;i<20;i++) polling.refresh();
    await vi.advanceTimersByTimeAsync(30000); expect(read).toHaveBeenCalledTimes(1);
    resolve("done"); await settle(); expect(success).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(19999); expect(read).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1); expect(read).toHaveBeenCalledTimes(2); expect(failure).not.toHaveBeenCalled();
  });
  it("a caller abort bounds even a transport ignoring its signal", async () => {
    vi.useFakeTimers(); const controller = new AbortController(); let signal!: AbortSignal;
    const pending = deadlineRead(s => { signal=s; return new Promise(() => {}); }, READ_TIMEOUT_MS, controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({name:"AbortError"}); controller.abort(); await rejected;
    expect(signal.aborted).toBe(true); expect(vi.getTimerCount()).toBe(0);
  });
});

describe("shared FIFO dispatch", () => {
  it("caps bursts, leaves queued reads loading, and starts deadlines only on dispatch", async () => {
    vi.useFakeTimers(); const activeStops: Array<() => void> = [];
    const blockers = vi.fn().mockReturnValue(new Promise(() => {}));
    for (let i=0;i<MAX_CONCURRENT_READS;i++) activeStops.push(watchSource(`/busy/${i}`, () => {}, 20000, blockers));
    stops.push(...activeStops);
    const events: SourceResult[] = []; let resolve!: (value: Response) => void;
    const transport = vi.fn().mockImplementation(() => new Promise<Response>(done => { resolve=done; }));
    stops.push(watchSource("/queued", value => events.push(value), 20000, transport));
    await settle(); expect(blockers).toHaveBeenCalledTimes(MAX_CONCURRENT_READS); expect(transport).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(44000); expect(events[events.length - 1]?.kind).toBe("loading");
    activeStops.forEach(stop=>stop()); await settle(); expect(transport).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(44000); expect(events[events.length - 1]?.kind).toBe("loading");
    resolve(response({ok:true,rows:[]})); await settle(); expect(events[events.length - 1]?.kind).toBe("empty");
  });
  it("cancels queued navigation without ever sending obsolete requests", async () => {
    vi.useFakeTimers(); const activeStops: Array<() => void> = [];
    for (let i=0;i<MAX_CONCURRENT_READS;i++) activeStops.push(watchSource(`/busy/${i}`, () => {}, 20000, vi.fn().mockReturnValue(new Promise(()=>{}))));
    stops.push(...activeStops); const old = vi.fn(), current = vi.fn().mockResolvedValue(response({ok:true,rows:[]}));
    const stopOld = watchSource("/old", () => {}, 20000, old); stops.push(stopOld);
    stopOld(); stops.push(watchSource("/current", () => {}, 20000, current));
    activeStops.forEach(stop => stop()); await settle();
    expect(old).not.toHaveBeenCalled(); expect(current).toHaveBeenCalledOnce();
  });
});
