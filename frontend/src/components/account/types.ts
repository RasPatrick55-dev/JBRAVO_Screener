export type AccountRangeKey = "d" | "w" | "m" | "y" | "all";

export interface AccountSummary {
  equity: number | null;
  cash: number | null;
  buyingPower: number | null;
  openPositionsValue: number | null;
  cashToPositionsRatio: number | null;
  takenAtUtc: string;
}

export interface AccountPerformanceRow {
  period: string;
  netChangePct: number | null;
  netChangeUsd: number | null;
}

export interface AccountTotal {
  equity: number | null;
  netChangePct: number | null;
  netChangeUsd: number | null;
  equityBasis?: "live" | "last_close";
  asOfUtc?: string;
  performanceBasis?: "live_vs_close_baselines" | "close_to_close";
}

export interface EquityCurvePoint {
  t: string;
  equity: number;
}

export interface OpenOrderRow {
  symbol: string;
  type: string;
  side: string;
  qty: number | null;
  priceOrStop: number | null;
  submittedAt: string;
}

export type OrderLogLevel = "success" | "info" | "warning";

export interface OrderLogRow {
  ts: string;
  level: OrderLogLevel;
  message: string;
}
