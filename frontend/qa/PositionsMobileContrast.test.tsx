import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { PositionRowMobile } from "../src/components/positions/PositionRow";
const props = {symbol:"TEST",logoUrl:"",shares:10,entryPrice:20,currentPrice:22,openPL:20,daysHeld:3,trailingStop:19,capturedPL:-5};
describe("mobile position field theme contrast", () => {
  it("gives all six secondary labels a light surface without changing the dark surface", () => {
    const html = renderToStaticMarkup(<PositionRowMobile {...props} />);
    const fields = [...html.matchAll(/<div class="([^"]+)"><div class="[^"]*text-secondary">([^<]+)<\/div>/g)];
    expect(fields.map(match => match[2])).toEqual(["Shares","Days","Entry Price","Current Price","Trailing Stop","Captured P/L"]);
    for (const [, classes] of fields) {
      expect(classes.split(/\s+/)).toContain("bg-slate-50/75");
      expect(classes.split(/\s+/)).toContain("dark:bg-slate-950/45");
      expect(classes.split(/\s+/)).not.toContain("bg-slate-950/45");
    }
  });
  it.each([[-5,"text-accent-red"],[5,"text-accent-green"],[0,"text-financial"],[null,"text-financial"]] as const)("preserves captured P/L semantics for %s", (capturedPL, tone) => {
    const html = renderToStaticMarkup(<PositionRowMobile {...props} capturedPL={capturedPL} />);
    expect(html).toMatch(new RegExp(`Captured P/L</div><div class="[^"]*${tone}`));
    expect(html).toContain("text-secondary");
  });
});
