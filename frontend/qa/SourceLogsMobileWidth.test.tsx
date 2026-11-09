import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ScreenerLogsResponse, LogsStage } from "../src/components/screener/types";
import fixture from "./sourceLogsLongReceiptFixture.json";

afterEach(() => {
  vi.doUnmock("react");
  vi.resetModules();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// Exercise the real request -> state -> component flow without a browser or provider.
async function renderLoaded(stage: LogsStage, payload: ScreenerLogsResponse) {
  vi.resetModules();
  let cursor = 0;
  const states: unknown[] = [];
  const effects: Array<() => void | (() => void)> = [];
  vi.stubGlobal("window", globalThis);
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 })));
  vi.doMock("react", async () => {
    const actual = await vi.importActual<typeof import("react")>("react");
    return {
      ...actual,
      useState: (initial: unknown) => {
        const index = cursor++;
        if (!(index in states)) states[index] = initial;
        return [states[index], (next: unknown) => { states[index] = next; }];
      },
      useEffect: (effect: () => void | (() => void)) => { effects.push(effect); },
      useMemo: (calculate: () => unknown) => calculate(),
      useRef: (value: unknown) => ({ current: value }),
    };
  });
  const { default: LogsPanel } = await import("../src/components/screener/LogsPanel");
  renderToStaticMarkup(<LogsPanel title={`${stage} logs`} stage={stage} />);
  const stop = effects[0]();
  try {
    for (let i = 0; i < 40; i++) await Promise.resolve();
    cursor = 0;
    return renderToStaticMarkup(<LogsPanel title={`${stage} logs`} stage={stage} />);
  } finally {
    if (typeof stop === "function") stop();
  }
}

describe("Source Logs Board mobile intrinsic sizing", () => {
  it.each(["screener", "backtest", "metrics"] as const)(
    "%s lets the grid item shrink while retaining the actual long source receipts",
    async stage => {
      const payload = fixture.responses[stage];
      const html = await renderLoaded(stage, payload);
      const classes = html.match(/^<section class="([^"]+)"/)![1].split(/\s+/);
      // break-word does not lower min-content width. The grid item must opt out
      // of its automatic minimum instead of clipping or globally hiding overflow.
      expect(classes).toContain("min-w-0");
      expect(classes).not.toContain("overflow-hidden");
      expect(classes).toContain("jbravo-panel");
      expect(classes).toContain({ screener: "jbravo-panel-cyan", backtest: "jbravo-panel-violet", metrics: "jbravo-panel-emerald" }[stage]);
      expect(html).toContain(payload.source_detail);
      for (const receipt of payload.source_receipts) {
        expect(html).toContain(receipt.source);
        expect(html).toContain(receipt.timestamp_basis);
        expect(html).toContain(receipt.observed_at);
        expect(html).toContain(receipt.latest_event_at);
      }
      expect(html).toContain("Fetch success does not imply a current computation");
      expect(html).toContain("Log text is not broker fill evidence");
      expect(html).toContain("No results available.");
      expect(html).not.toContain('role="alert"');
      expect(fetch).toHaveBeenCalledTimes(1);
    },
  );
});
