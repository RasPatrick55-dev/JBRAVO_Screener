import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import MonitoringPositionsCard from "../src/components/dashboard/MonitoringPositionsCard";
import AccountOverview from "../src/screens/AccountOverview";
import DashboardHealth from "../src/screens/DashboardHealth";
import ExecuteTrades from "../src/screens/ExecuteTrades";
import MLPipeline from "../src/screens/MLPipeline";
import PositionsMonitoring from "../src/screens/PositionsMonitoring";
import ScreenerOverview from "../src/screens/ScreenerOverview";
import TradesOverview from "../src/screens/TradesOverview";

const css = readFileSync("src/styles/tailwind.css", "utf8");
describe("dashboard theme and responsive presentation", () => {
  it.each([DashboardHealth, AccountOverview, TradesOverview, PositionsMonitoring, ExecuteTrades, ScreenerOverview, MLPipeline])(
    "%s inherits the document theme without a nested dark wrapper", (Screen) => {
      const html = renderToStaticMarkup(<Screen />);
      const classes = [...html.matchAll(/class="([^"]*)"/g)].flatMap(match => match[1].split(/\s+/));
      expect(classes).not.toContain("dark");
      expect(html).toContain("var(--app-nav-height,208px)");
      expect(html).toContain("dark:text-slate-100");
    });
  it("switches monitoring columns by card width, not desktop viewport width", () => {
    const html = renderToStaticMarkup(<MonitoringPositionsCard positions={[{symbol:"ARDT", currentPrice:17.3, percentPL:22.7, dollarPL:1008, costBasis:4440, sparklineData:[]}]} />);
    expect(html).toContain("monitoring-positions");
    expect(html).toContain("monitoring-position-row");
    expect(html).toContain("monitoring-position-compact");
    expect(html).not.toContain("sm:grid-cols-[");
    expect(html).not.toContain("min-w-[44px]");
    expect(html).toContain("ARDT"); expect(html).toContain("$17.30"); expect(html).toContain("+$1,008.00");
    expect(css).toContain("container-type: inline-size");
    expect(css).toContain("@container (min-width: 38rem)");
  });
  it("keeps unknown exposure distinct from zero in the compact presentation", () => {
    const html = renderToStaticMarkup(<MonitoringPositionsCard available={false} positions={[]} />);
    expect(html).toContain("EXPOSURE UNKNOWN"); expect(html).toContain("exposure are unknown");
    expect(html).not.toContain("$0.00");
  });
  it("uses readable secondary text in each surface palette", () => {
    const luminance = (hex: string) => {
      const rgb = hex.match(/[a-f0-9]{2}/gi)!.map(value => parseInt(value,16)/255)
        .map(value => value <= 0.04045 ? value/12.92 : ((value+0.055)/1.055)**2.4);
      return rgb[0]*0.2126+rgb[1]*0.7152+rgb[2]*0.0722;
    };
    for (const [selector, background] of [[":root", "#e2e8f0"], [".dark", "#0f172a"], [".jbravo-inverse", "#0f172a"]]) {
      const start=css.indexOf(`${selector} {`), block=css.slice(start,css.indexOf("}",start));
      const color=block.match(/--ds-text-secondary: (#[a-f0-9]+)/)![1];
      const levels=[luminance(color),luminance(background)].sort((a,b)=>b-a);
      expect((levels[0]+0.05)/(levels[1]+0.05)).toBeGreaterThanOrEqual(4.5);
    }
  });
});
