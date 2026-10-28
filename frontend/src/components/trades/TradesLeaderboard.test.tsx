import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import TradesLeaderboard from "./TradesLeaderboard";

const base = {selectedRange: "all" as const, onRangeChange: () => {}, mode: "winners" as const, onModeChange: () => {}};
describe("leaderboard business state", () => {
  it.each(["Leaderboard held; performance is not established.", "Leaderboard unavailable; performance is not established."])("does not misrepresent %s as no trades", errorMessage => {
    const html = renderToStaticMarkup(<TradesLeaderboard {...base} rows={[]} errorMessage={errorMessage}/>);
    expect(html).toContain(errorMessage); expect(html).not.toContain("No leaderboard trades yet.");
  });
  it("hides previously ready rows when a hold arrives", () => {
    const rows = [{rank: 1, symbol: "STALE", pl: 999}];
    const ready = renderToStaticMarkup(<TradesLeaderboard {...base} rows={rows}/>);
    expect(ready).toContain("STALE");
    const held = renderToStaticMarkup(<TradesLeaderboard {...base} rows={rows} errorMessage="Leaderboard held; performance is not established."/>);
    expect(held).not.toContain("STALE"); expect(held).not.toContain("999"); expect(held).not.toContain("No leaderboard trades yet.");
  });
  it("retains a genuine successful empty state and does not show it while loading", () => {
    expect(renderToStaticMarkup(<TradesLeaderboard {...base} rows={[]}/>)).toContain("No leaderboard trades yet.");
    expect(renderToStaticMarkup(<TradesLeaderboard {...base} rows={[]} isLoading/>)).not.toContain("No leaderboard trades yet.");
  });
});
