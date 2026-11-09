import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import TradesTab from "../src/components/trades/TradesTab";
import TradesRangePills from "../src/components/trades/TradesRangePills";
import * as sources from "../src/components/ui/sourceState";

afterEach(() => vi.restoreAllMocks());
describe("bounded narrow Trades grid", () => {
  it("allows every legacy grid child to shrink without hiding page overflow", () => {
    vi.spyOn(sources, "useSource").mockReturnValue({payload: null, kind: "loading", validity: "loading", freshness: "unknown"});
    const html = renderToStaticMarkup(<TradesTab/>);
    expect(html).toContain("grid min-w-0 grid-cols-1 gap-md xl:grid-cols-3");
    for (const order of ["order-1 min-w-0", "order-3 min-w-0", "order-2 xl:order-none xl:col-span-1 xl:h-full xl:min-h-0 min-w-0"]) expect(html).toContain(order);
    expect(html).toContain("min-w-0 max-w-full rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-cyan");
    expect(html).toContain("min-w-0 max-w-full rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-emerald");
    expect(html).toContain("min-w-0 max-w-full flex-col overflow-hidden rounded-2xl p-md shadow-card jbravo-panel jbravo-panel-violet");
    // Existing local table/card clipping is unchanged; no page-wide concealment.
    expect(html).not.toContain("overflow-x-hidden");
  });
  it("uses five shrinkable controls on narrow screens and original desktop widths", () => {
    const html = renderToStaticMarkup(<TradesRangePills value="all" onChange={() => {}}/>);
    expect(html).toContain("grid min-w-0 grid-cols-5 gap-2 sm:inline-flex sm:items-center");
    expect(html.match(/sm:min-w-11/g)).toHaveLength(5);
    expect(html.match(/min-w-0 items-center justify-center rounded-lg px-1/g)).toHaveLength(5);
    for (const label of ["D", "W", "M", "Y", "ALL"]) expect(html).toContain(">"+label+"</button>");
    expect(html).toContain('aria-pressed="true"'); expect(html).toContain("dark:bg-amber-500/20");
  });
});
