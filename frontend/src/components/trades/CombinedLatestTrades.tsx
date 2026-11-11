import { Fragment } from "react";
import { formatCurrency, formatNumber, formatSignedCurrency } from "../dashboard/formatters";
import { strategyEvidenceLabel, strategyTime } from "./strategyHistory";
import type { CombinedTradeRow } from "./combinedOverview";
const money = (value:number|null) => value === null ? "Unknown" : formatSignedCurrency(value);
const price = (value:number|null) => value === null ? "Unknown" : formatCurrency(value);
const number = (value:number|null) => value === null ? "Unknown" : formatNumber(value);
const date = (value:string|null) => value === null ? "Unknown" : new Date(value).toLocaleDateString("en-US",{year:"numeric",month:"numeric",day:"numeric",timeZone:"UTC"});
const source = (row:CombinedTradeRow) => row.sourceKind === "LEGACY_EXECUTION" ? "Legacy executions" : "Strategy fills";
const tone = (value:number|null) => value === null || value === 0 ? "text-primary" : value > 0 ? "jbravo-text-success" : "jbravo-text-error";
function Evidence({row}:{row:CombinedTradeRow}) {
  return <details className="mt-2 text-[11px] text-secondary"><summary className="cursor-pointer">Source / original record</summary>
    <div className="mt-1 space-y-1 break-words">
      <p className="break-all">Record: {row.rowId}</p>
      <p>Source: {source(row)} | Account binding: {row.accountBinding}</p>
      <p>Gross inclusion: {strategyEvidenceLabel(row.inclusion)}{row.reason ? ` | ${strategyEvidenceLabel(row.reason)}` : ""}</p>
      <p>Price basis: {strategyEvidenceLabel(row.priceBasis)}</p>
      <p>Stored ownership: {strategyEvidenceLabel(row.ownershipScope)}. Historical evidence, not current position-management authority.</p>
      <p>Fill entry: {strategyTime(row.buyDate)} | Last exit: {strategyTime(row.sellDate)}</p>
      {row.sourceKind === "LEGACY_EXECUTION" ? <><p>Original Recorded P/L: {money(row.recordedPL)} (unqualified; not substituted for execution gross).</p>
        <p>Recorded entry: {strategyTime(row.recordedEntryTime)} | Recorded exit: {strategyTime(row.recordedExitTime)}</p></> : null}
      <p>Fees: Unknown | Net P/L: Unknown. Gross is before fees and does not qualify account or portfolio performance.</p>
    </div>
  </details>;
}
export default function CombinedLatestTrades({rows,isLoading,available,stale,page,totalRows,onPageChange}:{
  rows:CombinedTradeRow[];isLoading:boolean;available:boolean;stale:boolean;page:number;totalRows:number;onPageChange:(page:number)=>void;
}) {
  const empty = !available ? "History unavailable; combined count and amounts are unknown." : stale ? "No realized candidates in the retained sample. Current history is unverified." : "No realized trade candidates in this stored cohort.";
  return <section className="min-w-0 max-w-full rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-emerald sm:p-md" aria-label="Latest trades">
    <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">Latest Trades</h2>
    <p className="mt-1 text-xs text-secondary">Combined legacy executions and strategy fills. Known gross before fees; original ledger P/L stays in record details. Unknown amounts are not zero.</p>
    <p className="mt-1 text-[11px] text-secondary">Rows with excluded or unknown evidence remain visible but do not contribute to gross totals. Open and unfilled signals appear in Strategy Trade History.</p>
    {stale ? <p className="mt-2 text-xs font-semibold text-secondary" role="status">Stale cohort. Current source and bindings are unverified.</p> : null}
    {isLoading ? <p className="mt-3 p-5 text-center text-sm text-secondary" role="status">Loading combined trade history...</p> : null}
    {!isLoading && rows.length === 0 ? <p className="mt-3 rounded-xl border border-dashed border-slate-300 p-5 text-center text-sm text-secondary dark:border-slate-700">{empty}</p> : null}
    <div className="mt-3 space-y-3 sm:hidden" data-combined-layout="mobile">{rows.map(row=><article key={row.rowId} className="min-w-0 rounded-xl border border-slate-200/70 p-3 dark:border-slate-700/70">
      <div className="flex flex-wrap justify-between gap-2"><div className="min-w-0"><h3 className="text-sm font-semibold text-primary">{row.symbol}</h3><p className="text-[11px] text-secondary">{source(row)}</p></div>
        <p className={`min-w-0 break-all font-cousine text-sm font-bold tabular-nums ${tone(row.totalPL)}`}>{money(row.totalPL)}<small className="block text-[10px] font-normal text-secondary">Known gross P/L</small></p></div>
      <p className="mt-1 break-words text-xs text-secondary">{row.status} | {strategyEvidenceLabel(row.inclusion)}{row.reason ? `: ${strategyEvidenceLabel(row.reason)}` : ""}</p>
      <div className="mt-2 grid grid-cols-2 gap-2">{[
        ["Buy date",date(row.buyDate)],["Sell date",date(row.sellDate)],["Total days",number(row.totalDays)],["Shares",number(row.totalShares)],
        ["Entry price",price(row.avgEntryPrice)],["Exit price",price(row.priceSold)],
      ].map(([label,value])=><div key={label} className="min-w-0 rounded-md p-2 jbravo-panel-inner jbravo-panel-inner-emerald"><p className="text-[10px] font-semibold uppercase text-secondary">{label}</p><p className="mt-1 break-words font-cousine text-xs text-primary">{value}</p></div>)}</div>
      <Evidence row={row}/>
    </article>)}</div>
    {rows.length ? <div className="mt-3 hidden max-h-[440px] max-w-full overflow-auto sm:block" role="region" aria-label="Scrollable combined trade history" tabIndex={0} data-combined-layout="desktop">
      <table className="w-full min-w-[800px] table-fixed"><caption className="sr-only">One stored cohort, gross before fees. UTC dates. Original Recorded P/L is unqualified.</caption>
        <colgroup><col className="w-[17%]"/><col className="w-[13%]"/><col className="w-[13%]"/><col className="w-[9%]"/><col className="w-[10%]"/><col className="w-[12%]"/><col className="w-[12%]"/><col className="w-[14%]"/></colgroup>
        <thead className="sticky top-0 z-10 bg-slate-100/95 dark:bg-slate-900/95"><tr>{["Symbol / source","Buy date","Sell date","Days","Shares","Entry price","Exit price","Known gross P/L"].map(label=><th key={label} className="px-2 py-2 text-left text-[10px] font-semibold uppercase text-secondary">{label}</th>)}</tr></thead>
        <tbody>{rows.map(row=><Fragment key={row.rowId}><tr className="text-xs text-primary"><th scope="row" className="px-2 pt-3 text-left align-top">{row.symbol}<small className="block font-normal text-secondary">{source(row)}</small></th>
          {[date(row.buyDate),date(row.sellDate),number(row.totalDays),number(row.totalShares),price(row.avgEntryPrice),price(row.priceSold)].map((value,index)=><td key={index} className="break-words px-2 pt-3 align-top font-cousine">{value}</td>)}
          <td className={`break-words px-2 pt-3 align-top font-cousine font-bold ${tone(row.totalPL)}`}>{money(row.totalPL)}</td></tr>
          <tr className="border-b border-slate-200/70 dark:border-slate-700/70"><td colSpan={8} className="px-2 pb-3"><p className="mt-1 text-[11px] text-secondary">{row.status} | {strategyEvidenceLabel(row.inclusion)}{row.reason ? `: ${strategyEvidenceLabel(row.reason)}` : ""}</p><Evidence row={row}/></td></tr>
        </Fragment>)}</tbody></table>
    </div> : null}
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-secondary">
      <button type="button" disabled={!available || page === 0} onClick={()=>onPageChange(page-1)} className="rounded px-1 py-1 disabled:opacity-40">Previous history page</button>
      <button type="button" disabled={!available || (page+1)*25 >= totalRows} onClick={()=>onPageChange(page+1)} className="rounded px-1 py-1 disabled:opacity-40">Next history page</button>
      {available ? <span>Page {page+1} · {totalRows} stored candidates in this cohort</span> : null}
    </div>
  </section>;
}
