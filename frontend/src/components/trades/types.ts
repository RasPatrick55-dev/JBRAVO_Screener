export type RangeKey = "d" | "w" | "m" | "y" | "all";

export interface RangeRowMetrics {
  key: RangeKey;
  label: "DAILY" | "WEEKLY" | "MONTHLY" | "YEARLY" | "ALL";
  winRatePct: number;
  totalPL: number;
  topTrade: { symbol: string; pl: number };
  worstLoss: { symbol: string; pl: number };
  tradesCount: number;
}

export interface LeaderRow {
  rank: number;
  symbol: string;
  pl: number;
}

export interface LatestTradeRow {
  symbol: string;
  buyDate: string;
  sellDate: string;
  totalDays: number;
  totalShares: number;
  avgEntryPrice: number;
  priceSold: number;
  totalPL: number;
}

export type LeaderboardMode = "winners" | "losers";
