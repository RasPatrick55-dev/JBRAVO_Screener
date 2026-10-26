export type ScreenerPicksFilter = "all" | "top10" | "passed" | "errors";
export type BacktestWindow = "3M" | "6M" | "1Y" | "ALL";
export type MetricsFilter = "all" | "with_history" | "no_history";
export type LogsStage = "screener" | "backtest" | "metrics";
export type LogsChipFilter = "all" | "errors" | "warnings" | "today";

export interface ScreenerPickRow {
  rank: number | null;
  symbol: string;
  exchange: string;
  screened_at_utc: string | null;
  final_score: number | null;
  volume: number | null;
  dollar_volume: number | null;
  price: number | null;
  sma_ema_pct: number | null;
  entry_price: number | null;
  adv20: number | null;
  atrp: number | null;
}

export interface ScreenerPicksResponse {
  ok?: boolean;
  run_ts_utc?: string | null;
  status?: string | null;
  source?: string | null;
  source_detail?: string | null;
  rows?: ScreenerPickRow[];
}

export interface BacktestRow {
  symbol: string;
  window: BacktestWindow | string;
  trades: number | null;
  win_rate_pct: number | null;
  avg_return_pct: number | null;
  pl_ratio: number | null;
  max_dd_pct: number | null;
  avg_hold_days: number | null;
  total_pl_usd: number | null;
}

export interface BacktestResponse {
  ok?: boolean;
  run_ts_utc?: string | null;
  window?: BacktestWindow | string;
  source?: string | null;
  rows?: BacktestRow[];
}

export type GateValue = "PASS" | "FAIL";
export type BarsCompleteValue = "YES" | "NO";
export type ConfidenceValue = "Low" | "Medium" | "High";

export interface MetricsRow {
  symbol: string;
  signal_id: string;
  strategy_id: string;
  strategy_version: string;
  definition_sha256: string;
  backtest_result_id: string;
  trades: number | null;
  wins: number | null;
  losses: number | null;
  win_rate_pct: number | null;
  net_pnl: number | null;
  expectancy: number | null;
  profit_factor: number | null;
  max_drawdown_cash: number | null;
  sample_status: string;
}

export interface MetricsResponse {
  ok?: boolean;
  status?: string;
  run_ts_utc?: string | null;
  source?: string | null;
  strategy_id?: string;
  strategy_version?: string;
  definition_sha256?: string;
  backtest_result_id?: string;
  metrics_result_id?: string;
  metrics_status?: string;
  rows?: MetricsRow[];
}

export type LogLevel = "ERROR" | "WARN" | "INFO" | "SUCCESS";

export interface ScreenerLogRow {
  ts_utc: string | null;
  level: LogLevel | string;
  message: string;
}

export interface ScreenerLogsResponse {
  ok?: boolean;
  stage?: LogsStage | string;
  source?: string | null;
  source_detail?: string | null;
  rows?: ScreenerLogRow[];
}
