import { describe, expect, it } from "vitest";
import { buildLeaderboardRequestPath, normalizeLeaderboardPayload } from "./leaderboardData";

describe("leaderboardData", () => {
  it("builds leaderboard URL with winners/losers mode and limit", () => {
    const winnersUrl = new URL(buildLeaderboardRequestPath("m", "winners", 12), "http://localhost");
    const losersUrl = new URL(buildLeaderboardRequestPath("m", "losers", 12), "http://localhost");

    expect(winnersUrl.pathname).toBe("/api/trades/leaderboard");
    expect(winnersUrl.searchParams.get("range")).toBe("m");
    expect(winnersUrl.searchParams.get("mode")).toBe("winners");
    expect(winnersUrl.searchParams.get("limit")).toBe("12");

    expect(losersUrl.pathname).toBe("/api/trades/leaderboard");
    expect(losersUrl.searchParams.get("range")).toBe("m");
    expect(losersUrl.searchParams.get("mode")).toBe("losers");
    expect(losersUrl.searchParams.get("limit")).toBe("12");
  });

  it("uses API leaderboard rows as-is after normalization", () => {
    const payload = {
      rows: [
        { rank: 1, symbol: "AAPL", pl: 123.45 },
        { rank: 2, symbol: "TSLA", pl: -21.5 },
      ],
    };

    expect(normalizeLeaderboardPayload(payload)).toEqual([
      { rank: 1, symbol: "AAPL", pl: 123.45 },
      { rank: 2, symbol: "TSLA", pl: -21.5 },
    ]);
  });
});
