import type { ReactNode } from "react";
import type { SourceResult } from "../ui/sourceState";
import { STRATEGY_HISTORY_LIMIT, strategyCompletion, strategyCurrency, strategyEvidenceLabel, strategyHistoryView, strategyNumber, strategyOwnership, strategyTime, type StrategyHistoryRow } from "./strategyHistory";

function Evidence({ row }: { row: StrategyHistoryRow }) {
  return <details className="mt-2 text-[11px] font-normal text-secondary">
    <summary className="cursor-pointer">Signal / order evidence</summary>
    <div className="mt-1 space-y-1 break-all">
      <p>Signal: {row.signalId}</p>
      <p>Strategy: {row.strategyId ?? "Unknown"}</p>
      <p>Entry orders: {row.entryOrders.length ? row.entryOrders.join(", ") : "Unknown"}</p>
      <p>Exit orders: {row.exitOrders.length ? row.exitOrders.join(", ") : "None recorded; not proof of no exit"}</p>
      <p>Record updated: {strategyTime(row.updatedAt)}</p>
      <p>Evidence: stored broker order observations in PostgreSQL, not a live broker fill or position read.</p>
      <p>{strategyOwnership(row.ownership)}. Historical label only; current management authority has not been revalidated.</p>
    </div>
  </details>;
}

function State({ row }: { row: StrategyHistoryRow }) {
  return <div className="space-y-1 break-words">
    <p className="font-semibold text-primary">{row.state}</p>
    <p className="text-secondary">{strategyCompletion(row.completion)}</p>
    <p className="text-[11px] text-secondary">{strategyOwnership(row.ownership)}</p>
    {row.reason ? <p className="text-[11px] text-secondary">{strategyEvidenceLabel(row.reason)}</p> : null}
  </div>;
}

const tone = (value: number | null) => value === null || value === 0 ? "text-primary" : value > 0 ? "jbravo-text-success" : "jbravo-text-error";
function MobileField({ label, children }: { label: string; children: ReactNode }) {
  return <div className="min-w-0 rounded-lg p-2.5 jbravo-panel-inner jbravo-panel-inner-emerald">
    <p className="text-[10px] font-semibold uppercase tracking-wide text-secondary">{label}</p>
    <div className="mt-1 break-words text-xs text-primary">{children}</div>
  </div>;
}

export default function StrategyHistoryCard({ result }: { result: SourceResult }) {
  const view = strategyHistoryView(result);
  const loading = view.kind === "loading", unavailable = view.kind === "unavailable";
  const stale = result.freshness === "stale";
  const empty = stale ? "No rows in the retained strategy sample. Current history is unverified." :
    view.kind === "empty" ? "No strategy signals were returned by this successful stored-history read." : "No displayable strategy records. The strategy count is unknown.";
  return <section className="min-w-0 rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-emerald sm:p-md" aria-label="Strategy trade history">
    <header className="flex flex-wrap items-start justify-between gap-2">
      <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Strategy Trade History</h2>
      <span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase ring-1 ring-inset ${unavailable ? "jbravo-status-error" : "text-secondary"}`}>{loading ? "Loading" : unavailable ? "Unavailable" : stale ? "Stale read" : view.kind === "partial" ? "Partial evidence" : "Stored paper observations"}</span>
    </header>
    <p className="mt-2 text-xs text-secondary">Recent strategy signals and their stored paper fills. Native state and observed exits are shown separately; a full exit does not establish complete cost accounting.</p>
    <p className="mt-1 text-xs font-semibold text-secondary">Separate from the recorded ledger below. These rows are not added to its performance totals or leaderboard. Fees and net P/L are unknown.</p>
    {stale ? <p role="status" className="mt-2 text-xs font-semibold text-secondary">Showing a stale strategy read. Current lifecycle and fill state are unverified; timestamps below belong to the retained sample.</p> : null}
    <p className="mt-2 break-words text-[11px] text-secondary">Source: {result.payload?.source === "postgres_strategy_execution" ? "PostgreSQL strategy execution" : "Unknown"} | Last successful read: {strategyTime(result.receivedAt ?? null)}</p>
    <details className="mt-1 text-[11px] text-secondary">
      <summary className="cursor-pointer">Source and accounting limits</summary>
      <div className="mt-1 space-y-1 break-words">
        <p>Source read: {strategyTime(view.observedAt)} | Latest stored record: {strategyTime(view.sourceAsOf)}</p>
        <p>Freshness: stored observations, not live broker state. An updated read does not make historical fills current.</p>
        <p>Scope: configured paper broker | Stored broker account binding: {result.payload?.broker_account_match === "verified" ? "Verified" : "Unavailable"}</p>
        <p>Gross realized P/L is supplied only for a fully exited signal with complete fill evidence and native CLOSED state. Partial exits and unreconciled states have unknown gross here. Open-position marks are not included.</p>
        <p>Fees have not been allocated. Net performance is not qualified. Strategy and legacy records may overlap and must not be summed.</p>
        <p>Signals excluded because their order IDs also appear in the legacy ledger: {strategyNumber(view.legacyOverlap)}.</p>
      </div>
    </details>
    {view.error ? <p role="alert" className="mt-3 rounded-lg border border-rose-300/60 bg-rose-50/80 p-3 text-xs text-rose-800 dark:border-rose-400/40 dark:bg-rose-500/10 dark:text-rose-200">{view.error}</p> : null}
    {loading ? <p role="status" className="mt-3 rounded-xl p-5 text-center text-sm text-secondary jbravo-panel-inner jbravo-panel-inner-emerald">Loading stored strategy history...</p> : null}
    {!loading && !unavailable && !view.rows.length ? <p className="mt-3 rounded-xl border border-dashed border-slate-300 p-5 text-center text-sm text-secondary dark:border-slate-700">{empty}</p> : null}
    {view.rows.length > 0 ? <>
      <div className="mt-3 space-y-3 sm:hidden" data-strategy-layout="mobile">
        {view.rows.map(row => <article key={row.signalId} className="min-w-0 rounded-xl border border-slate-200/70 p-3 dark:border-slate-700/70" aria-label={`${row.symbol} strategy signal`}>
          <div className="flex flex-wrap justify-between gap-2">
            <div className="min-w-0"><h3 className="break-words text-sm font-semibold text-primary">{row.symbol}</h3><p className="mt-1 break-all text-[11px] text-secondary">{row.strategyId ?? "Strategy unknown"}</p></div>
            <div className="min-w-0 text-xs"><State row={row}/></div>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <MobileField label="Shares"><p>Entered: {strategyNumber(row.entryQty)}</p><p>Exited: {strategyNumber(row.exitedQty)}</p><p>Remaining: {strategyNumber(row.remainingQty)}</p></MobileField>
            <MobileField label="Average fill prices"><p>Entry: {strategyCurrency(row.entryPrice)}</p><p>Exit: {strategyCurrency(row.exitPrice)}</p></MobileField>
            <MobileField label="Entry fill time"><time>{strategyTime(row.entryTime)}</time></MobileField>
            <MobileField label="Last exit fill time"><time>{strategyTime(row.exitTime)}</time></MobileField>
            <MobileField label="Gross realized P/L"><p className={`font-cousine font-bold tabular-nums ${tone(row.gross)}`}>{strategyCurrency(row.gross)}</p><p className="mt-1 text-[10px] text-secondary">Before fees</p></MobileField>
            <MobileField label="Cost accounting"><p>Fees: Unknown</p><p>Net P/L: Unknown</p></MobileField>
          </div>
          <Evidence row={row}/>
        </article>)}
      </div>
      <div className="mt-3 hidden max-h-[440px] max-w-full overflow-auto sm:block" role="region" aria-label="Scrollable strategy trade history" tabIndex={0} data-strategy-layout="desktop">
        <table className="w-full min-w-[920px] table-fixed">
          <caption className="sr-only">Stored paper strategy signals, separate from recorded ledger accounting. All dates are UTC. Unknown amounts are not zero.</caption>
          <colgroup><col className="w-[17%]"/><col className="w-[18%]"/><col className="w-[12%]"/><col className="w-[13%]"/><col className="w-[18%]"/><col className="w-[12%]"/><col className="w-[10%]"/></colgroup>
          <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95"><tr className="border-b border-slate-200/70 text-left text-[10px] font-semibold uppercase tracking-wide text-secondary dark:border-slate-700/70">
            {["Symbol / strategy", "Native state / fills", "Shares", "Average fill prices", "Fill times (UTC)", "Gross realized P/L", "Cost accounting"].map(label => <th scope="col" key={label} className="px-2 py-2">{label}</th>)}
          </tr></thead>
          <tbody>{view.rows.map(row => <tr key={row.signalId} className="border-b border-slate-200/70 text-xs hover:bg-sky-100/35 dark:border-slate-700/70 dark:hover:bg-slate-800/45">
            <th scope="row" className="px-2 py-3 text-left align-top"><span className="break-words font-semibold text-primary">{row.symbol}</span><span className="mt-1 block break-all text-[11px] font-normal text-secondary">{row.strategyId ?? "Strategy unknown"}</span><Evidence row={row}/></th>
            <td className="px-2 py-3 align-top"><State row={row}/></td>
            <td className="break-words px-2 py-3 align-top text-primary"><p>In: {strategyNumber(row.entryQty)}</p><p>Out: {strategyNumber(row.exitedQty)}</p><p>Left: {strategyNumber(row.remainingQty)}</p></td>
            <td className="break-words px-2 py-3 align-top text-primary"><p>In: {strategyCurrency(row.entryPrice)}</p><p>Out: {strategyCurrency(row.exitPrice)}</p></td>
            <td className="px-2 py-3 align-top text-secondary"><p>Entry: {strategyTime(row.entryTime)}</p><p className="mt-1">Last exit: {strategyTime(row.exitTime)}</p></td>
            <td className="break-words px-2 py-3 align-top"><p className={`font-cousine font-bold tabular-nums ${tone(row.gross)}`}>{strategyCurrency(row.gross)}</p><p className="mt-1 text-[10px] text-secondary">Before fees</p></td>
            <td className="px-2 py-3 align-top text-secondary"><p>Fees: Unknown</p><p>Net: Unknown</p></td>
          </tr>)}</tbody>
        </table>
      </div>
    </> : null}
    {!loading && !unavailable ? <p className="mt-3 text-[11px] text-secondary">Latest {STRATEGY_HISTORY_LIMIT} strategy signals requested. Showing {view.rows.length} after exclusions and identity checks.{view.hasMore === true ? " More signals exist; older history is not shown." : view.hasMore === null ? " Additional-history coverage is unknown." : ""} This is not an account-wide trade count.</p> : null}
  </section>;
}
