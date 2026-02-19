export type AccountRangeKey = "d" | "w" | "m" | "y" | "all";

export interface AccountSummary {
  equity: number;
  cash: number;
  buyingPower: number;
  openPositionsValue: number;
  cashToPositionsRatio: number | null;
  takenAtUtc: string;
}

export interface AccountPerformanceRow {
  period: string;
  netChangePct: number;
  netChangeUsd: number;
}

export interface AccountTotal {
  equity: number;
  netChangePct: number;
  netChangeUsd: number;
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
  qty: number;
  priceOrStop: number | null;
  submittedAt: string;
}

export type OrderLogLevel = "success" | "info" | "warning";

export interface OrderLogRow {
  ts: string;
  level: OrderLogLevel;
  message: string;
}
