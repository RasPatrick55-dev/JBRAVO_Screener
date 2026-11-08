import { useEffect, useState } from "react";
import { useSource, type SourceKind, type SourcePayload, type SourceResult } from "../ui/sourceState";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import type { BacktestRow } from "./types";
import { ComputationScope, computationComplete, type ComputationContract } from "./completion";
import { backtestRows } from "./backtestRows";
import { evidenceLabel, evidenceText, evidenceTone, formatRatio } from "./strategyEvidence";
import { strategyLabel, formatMetricWinRate } from "./metricsRows";
import { formatCurrency, formatNumber, formatRunBadge, formatSignedCurrency } from "./utils";

export const BACKTEST_PATH = "/api/screener/backtest?limit=200";
type BacktestResultsCardProps = { onSyncStateChange?: (state: LiveDataSyncState) => void };

export function BacktestCompletionBadge({payload, kind}: {payload: SourcePayload | null; kind: SourceKind}) {
  const status = String(payload?.status ?? "").toUpperCase();
  const complete = computationComplete(payload as ComputationContract | null);
  const label = kind === "loading" ? "LOADING" : kind === "stale" ? "STALE" : kind === "held" ? "HELD"
    : kind === "unavailable" ? "UNAVAILABLE" : status === "ERROR" ? "ERROR" : complete ? "COMPUTE COMPLETE" : "UNVERIFIED";
  const tone = label === "COMPUTE COMPLETE" ? "jbravo-status-success" : ["ERROR", "UNAVAILABLE"].includes(label) ? "jbravo-status-error" : "text-secondary";
  return <span className={`inline-flex rounded-full px-2 py-1 text-[10px] font-semibold uppercase ring-1 ring-inset ${tone}`}>{label}</span>;
}

export function backtestPresentation(result: Pick<SourceResult, "kind" | "payload">) {
  let rows: BacktestRow[] = [], error: string | null = null;
  if (result.kind !== "loading") {
    if (!["ready", "empty", "partial"].includes(result.kind)) error = `Backtest ${result.kind}; current evidence is not established.`;
    else {
      try { rows = backtestRows(result.payload ?? {}); }
      catch (cause) { error = cause instanceof Error ? cause.message : "Strategy backtest is unavailable."; }
    }
  }
  return { rows, error, sync: result.kind === "loading" ? "loading" : error ? "error" : "ready" } as const;
}

export function BacktestResultsContent({ result, onRetry }: { result: Pick<SourceResult, "kind" | "payload">; onRetry?: () => void }) {
  const { rows, error } = backtestPresentation(result);
  const isLoading = result.kind === "loading";
  const payload = error || isLoading ? null : result.payload;
  const evaluationEnd = evidenceText(payload?.evaluation_end);
  const window = evidenceText(payload?.window);
  return (
    <section className="min-w-0 rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-violet sm:p-md" aria-label="Backtest results">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">Backtest Results for Screener Picks</h2>
        <div className="flex flex-wrap items-center gap-2">
          <span className="max-w-full truncate rounded-full px-2 py-1 font-cousine text-[10px] font-semibold uppercase tracking-[0.06em] text-secondary outline outline-1 outline-slate-300/80 dark:outline-slate-600/80 sm:px-2.5 sm:text-[11px]">
            {formatRunBadge(evidenceText(payload?.run_ts_utc))}
          </span>
          <BacktestCompletionBadge payload={payload} kind={error && result.kind !== "stale" && result.kind !== "held" ? "unavailable" : result.kind}/>
        </div>
      </header>
      <ComputationScope payload={payload as ComputationContract | null} />
      <p className="mt-2 text-xs text-secondary">Candidate-symbol history only. Simulated completed signals, not actual paper-account returns. Histories are not pooled.</p>
      <p className="mt-1 text-xs text-secondary">Closed-signal P&amp;L includes modeled costs and excludes cash dividends and open-position marks. Drawdown is USD, not a percentage. Current proposals have no future result yet.</p>
      {!isLoading && !error ? <p className="mt-1 text-[11px] text-secondary">
        {window ? `Window: ${evidenceLabel(window)}.` : "History window: not supplied by this source."}
        {evaluationEnd ? ` Evaluation end: ${evaluationEnd}.` : ""} A zero-sample total is not evidence of profitability or safety.
      </p> : null}
      {error ? <div role="alert" className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-rose-300/60 bg-rose-50/80 px-3 py-2 text-xs text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/10 dark:text-rose-200">
        <span>{error}</span><button type="button" onClick={onRetry} className="rounded-md px-2 py-1 font-semibold uppercase tracking-wide outline outline-1 outline-rose-300/70 transition hover:bg-rose-100/70 dark:outline-rose-400/45 dark:hover:bg-rose-500/20">Retry</button>
      </div> : null}
      <div className="mt-2 overflow-x-auto sm:mt-3" role="region" aria-label="Scrollable Backtest results" tabIndex={0}>
        <div className="max-h-[320px] min-w-[980px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-violet sm:max-h-[360px] sm:min-w-[1180px]">
          <table className="w-full table-fixed">
            <caption className="sr-only">Candidate-symbol strategy simulations. Closed-signal cash results, not account returns.</caption>
            <colgroup><col className="w-[25%]"/>{Array.from({length: 6}, (_, i) => <col key={i} className="w-[10%]"/>)}<col className="w-[15%]"/></colgroup>
            <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
              <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
                {["Symbol / strategy", "Closed signals", "Win rate", "Net P&L (USD)", "Expectancy (USD)", "Profit factor", "Drawdown (USD)", "History"].map((label, i) => <th key={label} scope="col" className={`px-2 py-2 ${i === 0 || i === 7 ? "text-left" : "text-right"}`}>{label}</th>)}
              </tr>
            </thead>
            <tbody>
              {isLoading ? <tr><td colSpan={8} className="px-3 py-8 text-center text-sm text-secondary">Loading strategy backtest...</td></tr> : null}
              {!isLoading && !error && rows.length === 0 ? <tr><td colSpan={8} className="px-3 py-8 text-center text-sm text-secondary">No Screener picks in this batch.</td></tr> : null}
              {rows.map(row => <tr key={row.signal_id} className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45 sm:text-sm">
                <th scope="row" className="px-2 py-2.5 text-left font-semibold text-primary">
                  {row.symbol}<span className="mt-1 block break-words text-[11px] font-normal text-secondary">{strategyLabel(row.strategy_id, row.strategy_version)}</span>
                  <details className="mt-1 text-[11px] font-normal text-secondary"><summary className="cursor-pointer">Evidence</summary>
                    <div className="mt-1 space-y-1 break-all">
                      <p>Strategy: {row.strategy_id}; version {row.strategy_version}</p>
                      <p>Signal session: {row.current_signal_session || "unknown"}</p><p>Current signal: {row.signal_id}</p>
                      <p>Backtest: {row.backtest_result_id}</p><p>Definition: {row.definition_sha256 || "not supplied per row"}</p>
                      <p>Closed ledger: {row.closed_signal_ledger_sha256 || "unknown"}</p>
                      <p>Wins / losses / breakeven: {formatNumber(row.wins)} / {formatNumber(row.losses)} / {formatNumber(row.breakeven)}</p>
                      <p>Modeled fees: {formatCurrency(row.fees)}; slippage: {formatCurrency(row.slippage)}</p>
                      <p>Average hold: {formatRatio(row.average_hold_calendar_days)} calendar days</p>
                      <p>Current proposal: {evidenceLabel(row.current_proposal_state)}</p>
                    </div>
                  </details>
                </th>
                <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatNumber(row.closed_signals)}</td>
                <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatMetricWinRate(row.win_rate_pct)}</td>
                <td className={`font-cousine px-2 py-2.5 text-right tabular-nums ${evidenceTone(row.net_pnl)}`}>{formatSignedCurrency(row.net_pnl)}</td>
                <td className={`font-cousine px-2 py-2.5 text-right tabular-nums ${evidenceTone(row.expectancy)}`}>{formatSignedCurrency(row.expectancy)}</td>
                <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatRatio(row.profit_factor)}{row.profit_factor_status === "UNDEFINED_NO_LOSSES" ? <span className="block text-[10px] text-secondary">No losses; undefined</span> : null}</td>
                <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatCurrency(row.max_drawdown_cash)}</td>
                <td className="px-2 py-2.5 text-left text-secondary">{row.closed_signals === 0 ? "Insufficient evidence: 0 completed signals" : evidenceLabel(row.sample_status)}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default function BacktestResultsCard({ onSyncStateChange }: BacktestResultsCardProps) {
  const [reloadToken, setReloadToken] = useState(0);
  const result = useSource(`${BACKTEST_PATH}&view=${reloadToken}`);
  const { sync } = backtestPresentation(result);
  useEffect(() => { onSyncStateChange?.(sync); }, [sync, onSyncStateChange]);
  return <BacktestResultsContent result={result} onRetry={() => setReloadToken(value => value + 1)}/>;
}
