// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TradesTab from "./fixtures/LegacyTradesTab";
import gross from "./executionGrossFixture.json";

const response = (value: unknown) => new Response(JSON.stringify(value));
const settle = async () => {for (let i = 0; i < 80; i++) await Promise.resolve();};
let container: HTMLDivElement, root: Root;
beforeEach(() => {
  vi.useFakeTimers(); vi.setSystemTime(new Date("2026-11-09T17:00:00Z")); vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
});
afterEach(async () => {await act(async () => {root.unmount(); await settle();}); container.remove(); vi.useRealTimers(); vi.unstubAllGlobals();});
describe("gross filter request isolation", () => {
  it("does not show the previous range or accept its late response after navigation", async () => {
    let finishDaily!: (value: Response) => void;
    const calls: Array<{path: string; signal: AbortSignal | null | undefined}> = [];
    vi.stubGlobal("fetch", vi.fn((path: string, init: RequestInit) => {
      calls.push({path, signal: init.signal});
      if (path.includes("/leaderboard") && path.includes("range=d")) return new Promise<Response>(resolve => {finishDaily = resolve;});
      if (path.includes("/leaderboard") && path.includes("range=w")) return Promise.resolve(response({ok: false, status: "unavailable", rows: []}));
      if (path.includes("/leaderboard")) return Promise.resolve(response({...gross.winners, observed_at: new Date().toISOString()}));
      return Promise.resolve(response({ok: false, status: "unavailable", rows: []}));
    }));
    await act(async () => {root.render(<TradesTab/>); await settle();});
    const board = () => container.querySelector('section[aria-label="Leaderboard"]')!;
    const firstSymbol = gross.winners.rows[0].symbol;
    expect(board().textContent).toContain(firstSymbol);
    const click = async (label: string) => {await act(async () => {
      [...board().querySelectorAll("button")].find(button => button.textContent === label)!.click(); await settle();
    });};
    await click("D"); expect(board().textContent).not.toContain(firstSymbol);
    await click("W"); expect(board().textContent).toContain("Leaderboard unavailable");
    expect(calls.find(call => call.path.includes("/leaderboard") && call.path.includes("range=d"))?.signal?.aborted).toBe(true);
    await act(async () => {finishDaily(response({...gross.winners, observed_at: new Date().toISOString()})); await settle();});
    expect(board().textContent).not.toContain(firstSymbol); expect(board().textContent).toContain("Last successful read UTC: unknown");
    expect(calls.filter(call => call.path.includes("/leaderboard"))).toHaveLength(3);
    expect(calls.every(call => !call.path.includes("/leaderboard") || call.path.includes("basis=execution_gross"))).toBe(true);
  });
});
