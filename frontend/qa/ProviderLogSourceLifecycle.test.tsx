import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { watchProviderLogs, providerLogNotice, type ProviderLogs } from "../src/components/dashboard/providerLogs";
import DashboardHealth, { buildLogEntries } from "../src/screens/DashboardHealth";
import LatestLogsTable from "../src/components/dashboard/LatestLogsTable";
import PipelineStatusCard from "../src/components/dashboard/PipelineStatusCard";
import { watchSource, type SourceResult } from "../src/components/ui/sourceState";

const response = (payload: unknown, status = 200) => new Response(JSON.stringify(payload), { status });
const source = { source: "strategy_monitor_boot.py", text: 'Nov 8 19:43:42 {"finished_at":"2026-11-08T19:43:42.729306+00:00"}',
  entries: [{ raw: 'Nov 8 19:43:42 {"finished_at":"2026-11-08T19:43:42.729306+00:00"}', ts_utc: "2026-11-08T19:43:42.729306+00:00" }],
  observed_at: "2026-11-08T19:44:00Z", latest_event_at: "2026-11-08T19:43:42.729306+00:00" };
const good = () => response({ ok: true, source: "pythonanywhere", sources: [source] });
const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("provider-only log evidence lifecycle", () => {
  it("starts loading, tolerates a slow provider, and never requests local files", async () => {
    vi.useFakeTimers(); const events: ProviderLogs[] = [];
    const transport = vi.fn().mockImplementation(() => new Promise(resolve => setTimeout(() => resolve(good()), 17000)));
    const stop = watchProviderLogs(value => events.push(value), transport);
    expect(events[0].kind).toBe("loading");
    await vi.advanceTimersByTimeAsync(10001); expect(events[events.length - 1]?.kind).toBe("loading");
    await vi.advanceTimersByTimeAsync(7000); await settle(); expect(events[events.length - 1]?.kind).toBe("ready");
    expect(transport).toHaveBeenCalledTimes(1);
    expect(transport.mock.calls[0][0]).toBe("/api/pythonanywhere/logs"); stop();
  });
  it("retains the exact prior provider sample and its timestamps after a failed refresh, then recovers", async () => {
    vi.useFakeTimers(); const events: ProviderLogs[] = [];
    const transport = vi.fn().mockResolvedValueOnce(good()).mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(good());
    const stop = watchProviderLogs(value => events.push(value), transport);
    await settle(); const previous = events[events.length - 1]!;
    await vi.advanceTimersByTimeAsync(20000); await settle(); const stale = events[events.length - 1]!;
    expect(stale.kind).toBe("stale"); expect(stale.sources).toBe(previous.sources);
    expect(stale.receivedAt).toBe(previous.receivedAt); expect(stale.sources[0].observed_at).toBe(source.observed_at);
    expect(providerLogNotice(stale)).toContain("refresh failed"); expect(providerLogNotice(stale)).toContain("not current activity");
    await vi.advanceTimersByTimeAsync(20000); await settle(); expect(events[events.length - 1]?.kind).toBe("ready"); stop();
  });
  it.each([null, {}, { ok: false, source: "pythonanywhere", sources: [] }, { ok: true, source: "local", sources: [source] },
    { ok: true, source: "pythonanywhere", sources: [null] }, { ok: true, source: "pythonanywhere", sources: [{ source: "bad", text: null }] }])("does not turn invalid or failed evidence into no-activity claims: %j", async payload => {
    const events: ProviderLogs[] = [];
    const transport = vi.fn().mockResolvedValue(response(payload));
    const stop = watchProviderLogs(value => events.push(value), transport); await settle();
    expect(events[events.length - 1]?.kind).toBe("unavailable"); expect(events[events.length - 1]?.sources).toEqual([]);
    expect(providerLogNotice(events[events.length - 1]!)).toContain("current activity is unknown"); expect(transport).toHaveBeenCalledTimes(1); stop();
  });
  it("bounds a hung read and ignores a late response without overlapping requests", async () => {
    vi.useFakeTimers(); const events: ProviderLogs[] = []; let resolve!: (value: Response) => void;
    const transport = vi.fn().mockImplementation(() => new Promise<Response>(done => { resolve = done; }));
    const stop = watchProviderLogs(value => events.push(value), transport);
    await vi.advanceTimersByTimeAsync(40000); expect(transport).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5001); expect(events[events.length - 1]?.kind).toBe("unavailable");
    const count = events.length; resolve(good()); await settle(); expect(events).toHaveLength(count); stop();
  });
  it("does not publish a prior visit after unmount/navigation", async () => {
    const events: ProviderLogs[] = []; let resolve!: (value: Response) => void;
    const stop = watchProviderLogs(value => events.push(value), vi.fn().mockImplementation(() => new Promise<Response>(done => { resolve = done; })));
    stop(); resolve(good()); await settle(); expect(events).toHaveLength(1);
  });
  it("distinguishes a verified empty returned sample from absent activity", async () => {
    const events: ProviderLogs[] = [];
    const stop = watchProviderLogs(value => events.push(value), vi.fn().mockResolvedValue(response({ ok: true, source: "pythonanywhere", sources: [] })));
    await settle(); expect(events[events.length - 1]?.kind).toBe("empty");
    expect(providerLogNotice(events[events.length - 1]!)).toContain("does not establish that no activity occurred"); stop();
  });
  it("uses canonical monitor timestamps to sort newest first, above older scheduler rows", () => {
    const older = { ...source, text: "old", entries: [{ raw: "old", ts_utc: "2026-11-07T20:00:00Z" }] };
    const rows = buildLogEntries([older, source]);
    expect(rows[0].eventTimestamp).toBe(source.entries[0].ts_utc); expect(rows[0].time).not.toBe("--");
    expect(rows[1].eventTimestamp).toBe("2026-11-07T20:00:00Z");
  });
});

describe("honest initial dashboard state", () => {
  it("does not report overdue, pending stages, or no activity before reads complete", () => {
    const html = renderToStaticMarkup(<DashboardHealth />);
    expect(html).toContain("LOADING"); expect(html).toContain("Loading PythonAnywhere task logs");
    expect(html).toContain("Loading scheduled tasks"); expect(html).toContain("Loading PythonAnywhere resource evidence");
    expect(html).not.toContain("OVERDUE"); expect(html).not.toContain(">PENDING<");
    expect(html).not.toContain("No recent activity"); expect(html).not.toContain("No recent logs");
  });
  it("retained pipeline dates cannot assert fresh completion or stage results", () => {
    const html = renderToStaticMarkup(<PipelineStatusCard lastRun={{ date: "Nov 8", start: "03:45", end: "03:47", duration: "2m" }}
      subprocess={{ screener: "OK", backTester: "OK", metrics: "OK" }} isLive status="COMPLETED"
      taskSource={{ kind: "stale", freshness: "stale", receivedAt: "2026-11-08T19:00:00Z" }} />);
    expect(html).toContain("STALE"); expect(html).not.toContain(">COMPLETED<"); expect(html).not.toContain(">COMPLETE<");
    expect(html).toContain("Retained dates do not establish current readiness");
  });
  it("exposes stale/failure notice even with rows and filters", () => {
    const notice = providerLogNotice({ kind: "stale", sources: [source], receivedAt: "2026-11-08T19:44:00Z" });
    const html = renderToStaticMarkup(<LatestLogsTable notice={notice} receipts={[source]} rows={[{ dateTime: "Nov 8", script: source.source, text: source.text }]} />);
    expect(html).toContain("refresh failed"); expect(html).toContain("2026-11-08T19:44:00Z"); expect(html).toContain("Fetched:");
  });
  it("lets task receipts publish independently of a slow provider request", async () => {
    vi.useFakeTimers(); const tasks: SourceResult[] = [], logs: ProviderLogs[] = [];
    const stopTask = watchSource("/api/pipeline/task", value => tasks.push(value), 20000,
      vi.fn().mockResolvedValue(response({ ok: true, source: "primary_receipt", status: "COMPLETED" })));
    const stopLogs = watchProviderLogs(value => logs.push(value), vi.fn().mockImplementation(() => new Promise(() => {})));
    await settle(); expect(tasks[tasks.length - 1]?.kind).toBe("ready"); expect(logs[logs.length - 1]?.kind).toBe("loading"); stopTask(); stopLogs();
    const code = readFileSync(new URL("../src/screens/DashboardHealth.tsx", import.meta.url), "utf8");
    expect(code).toContain('useSource("/api/pipeline/task")'); expect(code).not.toContain("Promise.all");
    expect(code).not.toContain("/api/logs/"); expect(code).not.toContain("parsePipelineLogRun");
  });
});
