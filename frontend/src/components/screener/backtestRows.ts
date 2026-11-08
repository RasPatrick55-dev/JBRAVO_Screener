import type { BacktestRow } from "./types";
import { evidenceCount, evidenceNumber, evidenceText, isDigest, isRecord, isStrategyId, isVersion } from "./strategyEvidence";

export function backtestRows(payload: Record<string, unknown>): BacktestRow[] {
  const unified = payload.strategy_id === "ALL_STRATEGIES";
  if (payload.ok !== true || !["COMPLETE", "EMPTY"].includes(evidenceText(payload.status)) || payload.metric_basis !== "STRATEGY_RESEARCH_SIMULATION" ||
      !isVersion(payload.strategy_version) || !isDigest(payload.definition_sha256) || !Array.isArray(payload.rows) ||
      (unified ? payload.strategy_version !== "1" || payload.strategy_priority !== false ||
        payload.backtest_status !== "COMPLETED_UNIFIED_SETUP_BACKTEST" :
        !isStrategyId(payload.strategy_id) || !isDigest(payload.backtest_result_id))) {
    throw new Error("Strategy backtest evidence is unavailable.");
  }
  const signals = new Set<string>();
  const reports = new Map<string, string>();
  return payload.rows.map(raw => {
    if (!isRecord(raw)) throw new Error("Strategy backtest identity is unavailable.");
    const reportId = raw.backtest_result_id ?? (unified ? null : payload.backtest_result_id);
    const signalId = raw.current_signal_id;
    if (!isStrategyId(raw.strategy_id) || !isVersion(raw.strategy_version) || !isDigest(reportId) ||
        !isDigest(signalId) || raw.signal_id !== signalId || signals.has(signalId) ||
        !evidenceText(raw.symbol) || (!unified && (raw.strategy_id !== payload.strategy_id || raw.strategy_version !== payload.strategy_version))) {
      throw new Error("Backtest rows do not match their strategy or signal identities.");
    }
    const binding = JSON.stringify([raw.strategy_id, raw.strategy_version]);
    if (reports.has(reportId) && reports.get(reportId) !== binding) throw new Error("Backtest report identity is shared by different strategies.");
    reports.set(reportId, binding);
    signals.add(signalId);
    const count = evidenceCount(raw.closed_signals);
    const numeric = (key: string) => evidenceNumber(raw[key]);
    return {
      symbol: evidenceText(raw.symbol).toUpperCase(), strategy_id: raw.strategy_id, strategy_version: raw.strategy_version,
      signal_id: signalId, current_signal_session: evidenceText(raw.current_signal_session), backtest_result_id: reportId,
      definition_sha256: evidenceText(raw.definition_sha256) || (unified ? "" : evidenceText(payload.definition_sha256)),
      closed_signal_ledger_sha256: evidenceText(raw.closed_signal_ledger_sha256),
      closed_signals: count, wins: evidenceCount(raw.winning_signals), losses: evidenceCount(raw.losing_signals),
      breakeven: evidenceCount(raw.breakeven_signals), win_rate_pct: count === null || count === 0 ? null : numeric("win_rate_pct"),
      net_pnl: numeric("closed_trade_net_pnl"), expectancy: count === null || count === 0 ? null : numeric("expectancy_net_per_closed_signal"),
      profit_factor: count === null || count === 0 || raw.profit_factor_status !== "DEFINED" ? null : numeric("profit_factor"),
      profit_factor_status: evidenceText(raw.profit_factor_status),
      max_drawdown_cash: raw.drawdown_units === "USD_CLOSED_SIGNAL_PNL" ? numeric("max_closed_signal_drawdown_cash") : null,
      fees: numeric("fees_closed_signals"), slippage: numeric("slippage_closed_signals"),
      average_hold_calendar_days: count === 0 ? null : numeric("average_hold_calendar_days"),
      sample_status: evidenceText(raw.sample_status), current_proposal_state: evidenceText(raw.current_proposal_state),
    };
  });
}
