export type RangeKey = "d" | "w" | "m" | "y" | "all";

export interface RangeRowMetrics {
  key: RangeKey;
  label: "DAILY" | "WEEKLY" | "MONTHLY" | "YEARLY" | "ALL";
  winRatePct: number | null;
  totalPL: number | null;
  topTrade: { symbol: string; pl: number | null };
  worstLoss: { symbol: string; pl: number | null };
  tradesCount: number;
  coverage?: { included: number; excluded: number; unknown: number };
}

export interface LeaderRow {
  rank: number;
  symbol: string;
  pl: number;
}

export interface LatestTradeRow {
  symbol: string;
  buyDate: string | null;
  sellDate: string | null;
  totalDays: number | null;
  totalShares: number | null;
  avgEntryPrice: number | null;
  priceSold: number | null;
  totalPL: number | null;
  tradeId?: number;
  status?: string;
  grossPL?: string | null;
  grossStale?: boolean;
  disposition?: string;
  derivedObservedNet?: { net: string; fees: string; asOf: string } | null;
}

export type LeaderboardMode = "winners" | "losers";
