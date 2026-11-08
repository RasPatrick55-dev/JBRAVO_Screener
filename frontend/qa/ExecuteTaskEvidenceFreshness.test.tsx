import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { watchSource, type SourceResult } from "../src/components/ui/sourceState";
import ExecutionSummaryCard from "../src/components/execute/ExecutionSummaryCard";
import ExecuteTradesStatusCard from "../src/components/dashboard/ExecuteTradesStatusCard";
import ScheduledAttemptNotice from "../src/components/dashboard/ScheduledAttemptNotice";
import type { ProcessReceipt } from "../src/components/dashboard/processReceipt";

const payload = {
  ok:true, source:"executor_receipt", status:"COMPLETED",
  last_execution_receipt:{ok:true,source:"executor_receipt",status:"COMPLETED",finished_utc:"2026-02-04T11:00:00Z"},
  last_scheduled_attempt:{ok:true,source:"pythonanywhere_schedule_log",status:"RUNNING",started_utc:"2026-02-04T12:00:00Z"},
};
const response = (value: unknown) => new Response(JSON.stringify(value), {status:200});
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const renderSummary = (source: SourceResult) => {
  const task = source.payload as ProcessReceipt | null;
  return renderToStaticMarkup(<ExecutionSummaryCard summary={{ok:true,candidates:9,source:"independent_state_snapshot"}}
    isLoading={false} hasError={false} taskSource={source}
    lastExecutionReceipt={task?.last_execution_receipt} lastScheduledAttempt={task?.last_scheduled_attempt} />);
};

afterEach(() => vi.useRealTimers());

describe("execution task evidence freshness", () => {
  it("qualifies retained RUNNING and receipt evidence after successful load then failed refresh", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date("2026-02-04T12:00:00Z"));
    const events: SourceResult[] = [];
    const transport = vi.fn().mockResolvedValueOnce(response(payload)).mockRejectedValue(new Error("offline"));
    const stop = watchSource("/api/execute/task", result => events.push(result), 20000, transport);
    try {
      await settle();
      const before = events[events.length - 1];
      expect(before.kind).toBe("ready");
      expect(renderSummary(before)).toContain("Last scheduled attempt: RUNNING");
      expect(renderSummary(before)).toContain("Last execution receipt");
      await vi.advanceTimersByTimeAsync(20000);
      const after = events[events.length - 1];
      expect(after.payload).toEqual(before.payload); expect(after.freshness).toBe("stale");
      const html = renderSummary(after);
      expect(html).toContain("Last scheduled attempt: STALE EVIDENCE");
      expect(html).toContain("Cached attempt status: RUNNING");
      expect(html).toContain("current attempt and receipt availability are unknown");
      expect(html).toContain("Cached execution receipt");
      expect(html).toContain("Last known receipt status: COMPLETED");
      expect(html).not.toContain("Last scheduled attempt: RUNNING");
      expect(html).not.toContain("Actual receipt status: COMPLETED");
      // Independently refreshed state is still shown without refreshing task evidence.
      expect(html).toContain("independent_state_snapshot"); expect(html).toContain(">9<");
    } finally { stop(); }
  });

  it("marks a retained attempt stale when the next request hangs past expiry", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date("2026-02-04T12:00:00Z"));
    const events: SourceResult[] = [];
    const transport = vi.fn().mockResolvedValueOnce(response(payload)).mockReturnValue(new Promise(() => {}));
    const stop = watchSource("/api/execute/task", result => events.push(result), 20000, transport);
    try {
      await settle(); await vi.advanceTimersByTimeAsync(61000);
      const latest = events[events.length - 1];
      expect(latest.freshness).toBe("stale");
      expect(renderSummary(latest)).toContain("Cached attempt status: RUNNING");
      expect(renderSummary(latest)).not.toContain("Last scheduled attempt: RUNNING");
    } finally { stop(); }
  });

  it("shows unavailable evidence after initial transport failure", async () => {
    const events: SourceResult[] = [];
    const stop = watchSource("/api/execute/task", result => events.push(result), 20000, vi.fn().mockRejectedValue(new Error("offline")));
    try {
      await settle();
      expect(events[events.length - 1].kind).toBe("unavailable");
      const html = renderSummary(events[events.length - 1]);
      expect(html).toContain("Last scheduled attempt: UNAVAILABLE");
      expect(html).toContain("current attempt and receipt availability are unknown");
    } finally { stop(); }
  });

  it("does not discard a fresh independent scheduled attempt because the actual receipt is unavailable", () => {
    const html = renderSummary({payload:{...payload,ok:false,last_execution_receipt:{ok:false,status:"UNAVAILABLE"}},
      kind:"unavailable",validity:"unavailable",freshness:"current",receivedAt:"2026-02-04T12:00:00Z"});
    expect(html).toContain("Last scheduled attempt: RUNNING");
    expect(html).toContain("Actual receipt status: UNAVAILABLE");
  });

  it("does not promote retained values when task-feed freshness is unknown", () => {
    const html = renderSummary({payload,kind:"unavailable",validity:"unavailable",freshness:"unknown"});
    expect(html).toContain("Last scheduled attempt: UNAVAILABLE");
    expect(html).toContain("Unverified retained attempt status: RUNNING");
    expect(html).toContain("Unverified execution receipt");
    expect(html).not.toContain("Last scheduled attempt: RUNNING");
  });

  it("also qualifies cached receipt/attempt status in the Dashboard card", () => {
    const html = renderToStaticMarkup(<ExecuteTradesStatusCard lastRun={{date:"Feb 4",start:"11:00",end:"11:01",duration:"1m"}}
      ordersPlaced={0} totalValue={0} successRate={NaN} isCycleComplete status="COMPLETED"
      lastScheduledAttempt={payload.last_scheduled_attempt}
      taskSource={{kind:"stale",freshness:"stale",receivedAt:"2026-02-04T12:00:00Z"}} />);
    expect(html).toContain(">STALE<"); expect(html).not.toContain(">COMPLETED<");
    expect(html).toContain("Last Known Execution Receipt");
    expect(html).toContain("Cached attempt status: RUNNING");
    expect(html).not.toContain("Last scheduled attempt: RUNNING");
  });

  it("fails closed when a notice receives a record without source evidence", () => {
    const html = renderToStaticMarkup(<ScheduledAttemptNotice attempt={payload.last_scheduled_attempt} />);
    expect(html).toContain("Last scheduled attempt: UNAVAILABLE");
    expect(html).toContain("Unverified retained attempt status: RUNNING");
  });

  it("wires tracked source freshness through both consumers", () => {
    const execute = readFileSync(new URL("../src/components/execute/ExecuteTab.tsx", import.meta.url), "utf8");
    const dashboard = readFileSync(new URL("../src/screens/DashboardHealth.tsx", import.meta.url), "utf8");
    expect(execute).toContain("taskSource={executionTask}");
    expect(dashboard).toContain('const executeTaskSource = useSource("/api/execute/task")');
    expect(dashboard).toContain("taskSource: executeTaskSource");
    expect(dashboard).not.toContain('fetchJson<ExecuteTaskRunResponse>("/api/execute/task")');
  });
});
