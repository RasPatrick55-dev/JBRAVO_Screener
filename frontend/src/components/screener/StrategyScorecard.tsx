import { useEffect, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import { useSource } from "../ui/sourceState";
import { strategyLabel, formatMetricWinRate } from "./metricsRows";
import { evidenceLabel, evidenceTone, formatRatio } from "./strategyEvidence";
import { strategyScorecardPresentation, STRATEGY_SCORECARD_PATH, type StrategyScorecardRow, type ScorecardSource } from "./strategyScorecardRows";
import { formatCurrency, formatNumber, formatSignedCurrency } from "./utils";

function ScorecardEvidence({ row }: { row: StrategyScorecardRow }) {
  return <details className="mt-1 text-[11px] font-normal text-secondary">
    <summary className="cursor-pointer">Evidence / window</summary>
    <div className="mt-1 space-y-1 break-all">
      <p>Strategy: {row.strategy_id}; version {row.strategy_version || "unknown"}</p>
      <p>Definition: {row.definition_sha256 || "unknown"}</p><p>Episode: {row.episode_id || "unknown"}</p>
      <p>Window: {row.episode.test_start || "unknown"} to {row.episode.as_of || row.as_of || "unknown"}</p>
      <p>Universe: {evidenceLabel(row.episode.universe_kind)}; cost case: {evidenceLabel(row.episode.cost_case)}</p>
      <p>Declared symbols: {formatNumber(row.episode.declared_symbols)}</p>
      <p>Capital: {formatCurrency(row.episode.initial_cash)}; allocation: {formatMetricWinRate(row.episode.allocation_pct)}; max positions: {formatNumber(row.episode.max_positions)}; top N: {formatNumber(row.episode.top_n)}</p>
      <p>Modeled fee / fill: {formatCurrency(row.episode.fee_per_fill)}; slippage: {formatRatio(row.episode.slippage_bps)} bps</p>
      <p>Coverage: {evidenceLabel(row.episode.coverage_status)}; evaluated: {formatNumber(row.episode.evaluated_count)}; excluded: {formatNumber(row.episode.excluded_count)}; discovered: {formatNumber(row.episode.discovered_count)}</p>
      <p>Scope: {evidenceLabel(row.scope)}</p>
      <p>Archives: {formatNumber(row.evidence.archive_count)}; distinct signals: {formatNumber(row.evidence.distinct_signals)}; duplicate observations: {formatNumber(row.evidence.duplicate_observations)}</p>
      <p>Archive ID: {row.evidence.archive_id || "unknown"}</p><p>Scan: {row.evidence.scan_run_id || "unknown"}</p>
      <p>Result hash: {row.evidence.result_sha256 || "unknown"}</p><p>Manifest hash: {row.evidence.manifest_sha256 || "unknown"}</p>
      <p>Open: {formatNumber(row.open_signals)}; unfilled: {formatNumber(row.unfilled_signals)}. These are excluded from the completed sample.</p>
      <p>Sample: {evidenceLabel(row.sample_status)}</p>
      {row.reason ? <p>Evidence limitation: {evidenceLabel(row.reason)}</p> : null}
      {row.limitations.map((text, index) => <p key={index}>{text}</p>)}
    </div>
  </details>;
}

export function StrategyScorecardContent({ result, onRetry }: { result: ScorecardSource; onRetry?: () => void }) {
  const { rows, error, status, limitations, actualPaperReason, coverage } = strategyScorecardPresentation(result);
  const loading = result.kind === "loading";
  return <section className="min-w-0 rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-emerald sm:p-md" aria-label="Strategy scorecard">
    <header className="flex flex-wrap items-start justify-between gap-3">
      <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">Strategy Scorecard</h2>
      <span className={`inline-flex rounded-full px-2 py-1 text-[10px] font-semibold uppercase ring-1 ring-inset ${error ? "jbravo-status-error" : "text-secondary"}`}>
        {loading ? "LOADING" : error ? "UNAVAILABLE" : status === "PARTIAL" ? "PARTIAL EVIDENCE" : "SIMULATED EVIDENCE"}
      </span>
    </header>
    <p className="mt-2 text-xs text-secondary">One row per strategy version and compatible episode. Completed signals are deduplicated within each episode. Separate rows may overlap and must not be summed or averaged.</p>
    <p className="mt-1 text-xs text-secondary">Declared simulated portfolio signals only. This is not every qualifying opportunity, proof of a target win rate, or permission to trade.</p>
    <p className="mt-1 text-[11px] text-secondary">Closed-signal P&amp;L excludes cash dividends and open-position marks. Intervals describe USD expectancy, not win-rate certainty.</p>
    <div className="mt-3 rounded-xl p-3 jbravo-panel-inner jbravo-panel-inner-emerald">
      <p className="text-xs font-semibold text-primary">Actual paper performance: unavailable</p>
      <p className="mt-1 text-[11px] text-secondary">{actualPaperReason || "Complete actual paper cost accounting is not supplied by this research source."} Modeled costs below are separate from actual costs.</p>
    </div>
    {error ? <div role="alert" className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-rose-300/60 bg-rose-50/80 px-3 py-2 text-xs text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/10 dark:text-rose-200">
      <span>{error}</span><button type="button" onClick={onRetry} className="rounded-md px-2 py-1 font-semibold uppercase tracking-wide outline outline-1 outline-rose-300/70 transition hover:bg-rose-100/70 dark:outline-rose-400/45 dark:hover:bg-rose-500/20">Retry</button>
    </div> : null}
    {!loading && coverage.selection ? <p className="mt-2 text-[11px] text-secondary">Coverage: {evidenceLabel(coverage.selection)}. Verified groups: {formatNumber(coverage.verified_groups)}; unavailable groups: {formatNumber(coverage.unavailable_groups)}.
      {coverage.truncated ? ` Limited to ${formatNumber(coverage.max_groups)} groups; more history is not shown.` : ""}
      {coverage.rejected_archives !== null && coverage.rejected_archives > 0 ? ` Rejected archives: ${formatNumber(coverage.rejected_archives)}. ${coverage.reasons.map(evidenceLabel).join("; ")}.` : ""}
    </p> : null}
    {!loading && limitations.length ? <details className="mt-2 text-[11px] text-secondary"><summary className="cursor-pointer">Research limitations</summary><ul className="mt-1 space-y-1">{limitations.map((text, index) => <li key={index}>{text}</li>)}</ul></details> : null}
    <div className="mt-2 overflow-x-auto sm:mt-3" role="region" aria-label="Scrollable strategy scorecard" tabIndex={0}>
      <div className="max-h-[420px] min-w-[1200px] overflow-auto rounded-xl jbravo-panel-inner jbravo-panel-inner-emerald sm:max-h-[480px]">
        <table className="w-full table-fixed">
          <caption className="sr-only">Strategy-level simulated evidence, grouped by compatible episode. Unknown metrics are not zero.</caption>
          <colgroup><col className="w-[22%]"/><col className="w-[10%]"/><col className="w-[7%]"/><col className="w-[10%]"/><col className="w-[11%]"/><col className="w-[9%]"/><col className="w-[8%]"/><col className="w-[11%]"/><col className="w-[12%]"/></colgroup>
          <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95">
            <tr className="font-arimo border-b border-slate-300/80 text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary dark:border-slate-600/80">
              {["Strategy / evidence", "Completed sample", "Win rate", "Net expectancy", "Average win / loss", "Payoff ratio", "Profit factor", "Costs (USD)", "Expectancy uncertainty"].map((label, i) => <th key={label} scope="col" className={`px-2 py-2 ${i === 0 ? "text-left" : "text-right"}`}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {loading ? <tr><td colSpan={9} className="px-3 py-8 text-center text-sm text-secondary">Loading strategy-wide evidence...</td></tr> : null}
            {!loading && !error && !rows.length ? <tr><td colSpan={9} className="px-3 py-8 text-center text-sm text-secondary">No strategy episodes are available. This is not a zero-performance result.</td></tr> : null}
            {rows.map(row => <tr key={row.key} className="border-b border-slate-200/70 text-xs transition-colors hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45 sm:text-sm">
              <th scope="row" className="px-2 py-2.5 text-left font-semibold text-primary">
                <span className="break-words">{row.strategy_version ? row.strategy_name ? `${row.strategy_name} v${row.strategy_version}` : strategyLabel(row.strategy_id, row.strategy_version) : `${row.strategy_name || row.strategy_id.replace(/_/g, " ")} · version unknown`}</span>
                <span className="mt-1 block text-[11px] font-normal text-secondary">Simulated · {row.evidence_status === "VERIFIED" ? "verified ledger" : evidenceLabel(row.evidence_status)}</span>
                <span className="mt-1 block text-[11px] font-normal text-secondary">Archive as of: {row.freshness.evidence_as_of || row.as_of || "unknown"}{row.freshness.source_age_calendar_days !== null ? ` (${row.freshness.source_age_calendar_days} calendar days old)` : ""}</span>
                <span className="block text-[10px] font-normal text-secondary">Latest attempt: {evidenceLabel(row.freshness.latest_attempt_status)}</span>
                {row.evidence_status !== "VERIFIED" ? <span className="mt-1 block text-[11px] font-normal text-secondary">Strategy-wide performance unavailable</span> : null}
                
                <ScorecardEvidence row={row}/>
              </th>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                {formatNumber(row.closed_signals)}<span className="mt-1 block text-[10px] text-secondary">W / L / BE</span>
                <span className="text-xs">{formatNumber(row.wins)} / {formatNumber(row.losses)} / {formatNumber(row.breakeven)}</span>
                {row.closed_signals === 0 ? <span className="mt-1 block text-[10px] text-secondary">Insufficient evidence: 0 completed signals</span> : null}
              </td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatMetricWinRate(row.win_rate_pct)}</td>
              <td className={`font-cousine px-2 py-2.5 text-right tabular-nums ${evidenceTone(row.expectancy)}`}>
                {formatSignedCurrency(row.expectancy)}<span className="mt-1 block text-[10px] text-secondary">USD / closed signal</span>
                <span className="mt-1 block text-[10px] text-secondary">Net P&amp;L {formatSignedCurrency(row.net_pnl)}</span>
              </td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatCurrency(row.average_win)} / {formatCurrency(row.average_loss)}<span className="mt-1 block text-[10px] text-secondary">USD</span></td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatRatio(row.payoff_ratio)}<span className="mt-1 block text-[10px] text-secondary">Avg win / |avg loss|</span></td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatRatio(row.profit_factor)}<span className="mt-1 block text-[10px] text-secondary">{row.profit_factor === null ? evidenceLabel(row.profit_factor_status) : "Sum net wins / |net losses|"}</span></td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">{formatCurrency(row.costs.total)}<span className="mt-1 block text-[10px] text-secondary">{evidenceLabel(row.costs.basis)}</span>
                <span className="mt-1 block text-[10px] text-secondary">Fees {formatCurrency(row.costs.fees)}</span><span className="block text-[10px] text-secondary">Slippage {formatCurrency(row.costs.slippage)}</span><span className="block text-[10px] text-secondary">Actual: unknown</span>
              </td>
              <td className="font-cousine px-2 py-2.5 text-right tabular-nums text-primary">
                {row.uncertainty.low === null ? "--" : `${formatCurrency(row.uncertainty.low)} to ${formatCurrency(row.uncertainty.high)}`}
                <span className="mt-1 block text-[10px] text-secondary">{row.uncertainty.level === null ? evidenceLabel(row.uncertainty.status) : `${formatMetricWinRate(row.uncertainty.level * 100)} descriptive interval`}</span>
                <span className="mt-1 block text-[10px] text-secondary">{evidenceLabel(row.uncertainty.method)}</span>
                <span className="mt-1 block text-[10px] text-secondary">Signal weeks: {formatNumber(row.uncertainty.signal_weeks)}</span>
                {row.uncertainty.warning ? <span className="mt-1 block text-[10px] font-semibold text-secondary">Warning: {evidenceLabel(row.uncertainty.warning)}. Uncertainty may be unstable.</span> : null}
              </td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </div>
    <p className="mt-2 text-[11px] text-secondary">A verified ledger confirms the evidence source, not research qualification. Small samples and overlapping market periods limit certainty. No combined win-rate claim is calculated.</p>
  </section>;
}

export default function StrategyScorecard({ onSyncStateChange }: { onSyncStateChange?: (state: LiveDataSyncState) => void }) {
  const [reloadToken, setReloadToken] = useState(0);
  // Allow a complete bounded refresh after the one-minute poll before transport
  // evidence expires. Archive age remains separately displayed on every row.
  const result = useSource(`${STRATEGY_SCORECARD_PATH}?view=${reloadToken}`, 60_000, 35_000, 120_000);
  const { sync } = strategyScorecardPresentation(result);
  useEffect(() => { onSyncStateChange?.(sync); }, [sync, onSyncStateChange]);
  return <StrategyScorecardContent result={result} onRetry={() => setReloadToken(value => value + 1)}/>;
}
