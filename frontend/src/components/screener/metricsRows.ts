import type { MetricsResponse, MetricsRow } from "./types";
import { normalizeSymbol, parseNumber } from "./utils";

export const formatMetricWinRate = (value: number | null): string => value === null
  ? "--" : `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value)}%`;

export const strategyLabel = (id: string, version: string): string =>
  id === "trend_pullback_reclaim_qualifying_close_v2" ? `Pullback/Reclaim v${version}` : `${id.replace(/_/g, " ")} v${version}`;

export function metricRows(payload: MetricsResponse): MetricsRow[] {
  if (payload.ok !== true || !payload.strategy_id || !payload.strategy_version ||
      !payload.definition_sha256 || !payload.backtest_result_id || !payload.metrics_result_id) {
    throw new Error(payload.status === "METRICS_PENDING" ? "Metrics are waiting for the strategy backtest." : "Strategy metrics are unavailable.");
  }
  return (payload.rows ?? []).map(row => {
    if (row.strategy_id !== payload.strategy_id || row.strategy_version !== payload.strategy_version ||
        row.definition_sha256 !== payload.definition_sha256 || row.backtest_result_id !== payload.backtest_result_id ||
        !/^[0-9a-f]{64}$/.test(row.signal_id ?? "") || normalizeSymbol(row.symbol) === "--") {
      throw new Error("Metrics do not match the strategy backtest.");
    }
    return { ...row, symbol: normalizeSymbol(row.symbol), trades: parseNumber(row.trades),
      wins: parseNumber(row.wins), losses: parseNumber(row.losses), win_rate_pct: parseNumber(row.win_rate_pct),
      net_pnl: parseNumber(row.net_pnl), expectancy: parseNumber(row.expectancy),
      profit_factor: parseNumber(row.profit_factor), max_drawdown_cash: parseNumber(row.max_drawdown_cash) };
  });
}
