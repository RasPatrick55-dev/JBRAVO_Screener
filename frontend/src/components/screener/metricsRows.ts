import type { MetricsResponse, MetricsRow } from "./types";
import { normalizeSymbol, parseNumber } from "./utils";

export const formatMetricWinRate = (value: number | null): string => value === null
  ? "--" : `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value)}%`;

export const strategyLabel = (id: string, version: string): string =>
  id === "ALL_STRATEGIES" ? "All strategies (per-setup results)" :
  id === "trend_pullback_reclaim_qualifying_close_v2" ? `Pullback/Reclaim v${version}` : `${id.replace(/_/g, " ")} v${version}`;

const digest = (value: unknown): value is string => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
const strategyId = (value: unknown): value is string => typeof value === "string" && /^[a-z][a-z0-9_]{0,127}$/.test(value);
const version = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);

export function metricRows(payload: MetricsResponse): MetricsRow[] {
  const unified = payload.strategy_id === "ALL_STRATEGIES";
  if (payload.ok !== true || !version(payload.strategy_version) || !digest(payload.definition_sha256) ||
      !Array.isArray(payload.rows) || (unified ?
        payload.strategy_version !== "1" || payload.metric_basis !== "STRATEGY_RESEARCH_SIMULATION" ||
        payload.metrics_status !== "COMPLETED_UNIFIED_SETUP_METRICS" || payload.strategy_priority !== false ||
        payload.backtest_result_id !== undefined || payload.metrics_result_id !== undefined :
        !strategyId(payload.strategy_id) || !digest(payload.backtest_result_id) || !digest(payload.metrics_result_id))) {
    throw new Error(payload.status === "METRICS_PENDING" ? "Metrics are waiting for the strategy backtest." : "Strategy metrics are unavailable.");
  }
  const bindings = new Map<string, string>();
  const metricsReports = new Map<string, string>();
  const backtestReports = new Map<string, string>();
  const signals = new Set<string>();
  const bindReport = (reports: Map<string, string>, id: string, binding: string) => {
    if (reports.has(id) && reports.get(id) !== binding) {
      throw new Error("Metrics do not match the strategy backtest.");
    }
    reports.set(id,binding);
  };
  return payload.rows.map(row => {
    // The unified definition binds selection, not every row's strategy definition.
    // Its rows retain separate report identities; never manufacture pooled IDs/P&L.
    if (!row || typeof row !== "object" || !strategyId(row.strategy_id) || !version(row.strategy_version) ||
        !digest(row.definition_sha256) || !digest(row.backtest_result_id) || !digest(row.signal_id) ||
        normalizeSymbol(row.symbol) === "--" || (unified ? !digest(row.metrics_result_id) :
          row.strategy_id !== payload.strategy_id || row.strategy_version !== payload.strategy_version ||
          row.definition_sha256 !== payload.definition_sha256 || row.backtest_result_id !== payload.backtest_result_id ||
          row.metrics_result_id !== undefined && row.metrics_result_id !== payload.metrics_result_id)) {
      throw new Error("Metrics do not match the strategy backtest.");
    }
    const binding = JSON.stringify([row.strategy_version,row.definition_sha256,row.backtest_result_id,
      unified ? row.metrics_result_id : payload.metrics_result_id]);
    if (bindings.has(row.strategy_id) && bindings.get(row.strategy_id) !== binding) {
      throw new Error("Metrics do not match the strategy backtest.");
    }
    bindings.set(row.strategy_id,binding);
    // Report hashes bind their contracts in both directions. Distinct strategy
    // keys cannot make a reused report ID valid; many signals may share one report.
    const contract = [row.strategy_id,row.strategy_version,row.definition_sha256];
    const metricsId = row.metrics_result_id ?? payload.metrics_result_id!;
    bindReport(metricsReports,metricsId,JSON.stringify([...contract,row.backtest_result_id]));
    bindReport(backtestReports,row.backtest_result_id,JSON.stringify(contract));
    // The source emits one row per selected current signal (also the React key).
    if (signals.has(row.signal_id)) throw new Error("Metrics do not match the strategy backtest.");
    signals.add(row.signal_id);
    return { ...row, symbol: normalizeSymbol(row.symbol), trades: parseNumber(row.trades),
      wins: parseNumber(row.wins), losses: parseNumber(row.losses), win_rate_pct: parseNumber(row.win_rate_pct),
      net_pnl: parseNumber(row.net_pnl), expectancy: parseNumber(row.expectancy),
      profit_factor: parseNumber(row.profit_factor), max_drawdown_cash: parseNumber(row.max_drawdown_cash) };
  });
}
