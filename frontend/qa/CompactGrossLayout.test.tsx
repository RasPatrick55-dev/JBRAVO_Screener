import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import TradesPerformanceBoard from "../src/components/trades/TradesPerformanceBoard";
import TradesOverview from "../src/screens/TradesOverview";
import ScreenerOverview from "../src/screens/ScreenerOverview";
import { normalizeRecordedGrossStats } from "../src/components/trades/recordedGrossData";
import fixture from "./recordedGrossFixture.json";

describe("compact original-style gross table",()=>{
  it("uses desktop column headers once instead of repeating metric captions in every range",()=>{
    const html=renderToStaticMarkup(<TradesPerformanceBoard gross rows={normalizeRecordedGrossStats(fixture.stats)}/>);
    const header=html.match(/<thead[^>]*>([\s\S]*?)<\/thead>/)![1];
    const body=html.match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
    for(const label of ["Gross Win Rate","Known Gross P/L","Best Gross Trade","Worst Gross Trade","Trades"]){
      expect(header).toContain(label);expect(body).not.toContain(label);
    }
    expect(body.match(/scope="row"/g)).toHaveLength(5);
    expect(body).toContain("40.00%");expect(body).toContain("-$51.38");
    expect(body).toContain("AAA");expect(body).toContain("CRC");expect(body).toContain("-$63.38");
    expect(body).toContain("1 excluded / 0 unknown");
    expect(body).not.toMatch(/font-cousine mt-1/);
    // Mobile cards retain their necessary per-value labels and the responsive layout.
    const mobile=html.slice(html.indexOf('class="sm:hidden"'),html.indexOf('class="hidden overflow-auto sm:block"'));
    expect(mobile).toContain("Gross Win Rate (known)");expect(mobile).toContain("Best Gross Trade");
  });
  it("retains nullable/partial coverage qualifications and the global hold note",()=>{
    const html=renderToStaticMarkup(<TradesPerformanceBoard gross rows={normalizeRecordedGrossStats(fixture.unknown)}
      sourceNote="Before fees | Qualified net held | 0 included / 0 excluded / 1 unknown"/>);
    expect(html).toContain("Before fees");expect(html).toContain("Qualified net held");
    expect(html).toContain("0 excluded / 1 unknown");expect(html).toContain("--");
    expect(html).not.toContain("$0.00");
  });
});

describe("offline-font fallback in the two reviewed screens",()=>{
  it.each([["Trades",TradesOverview],["Screener",ScreenerOverview]] as const)("%s inherits the existing safe body font stack",(_name,Screen)=>{
    const html=renderToStaticMarkup(<Screen/>),rootClass=html.match(/^<div class="([^"]*)"/)![1];
    expect(rootClass).not.toContain("font-[");
    const css=readFileSync("src/styles/tailwind.css","utf8");
    expect(css).toMatch(/body\s*\{\s*font-family:\s*"Manrope", ui-sans-serif, system-ui, sans-serif;/);
    expect(css).toContain('font-family: "Arimo", ui-sans-serif, system-ui, sans-serif');
    expect(css).toContain('font-family: "Cousine", ui-monospace');
  });
});
