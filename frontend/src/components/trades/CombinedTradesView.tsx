import { useEffect, useMemo, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import type { SourceResult } from "../ui/sourceState";
import { currentCombinedOverview } from "./combinedOverview";
import StrategyHistoryCard from "./StrategyHistoryCard";
import CombinedLatestTrades from "./CombinedLatestTrades";
import TradesPerformanceBoard from "./TradesPerformanceBoard";
import TradesLeaderboard from "./TradesLeaderboard";
import { strategyTime } from "./strategyHistory";
import type { LeaderboardMode, RangeKey } from "./types";

export default function CombinedTradesView({result,onSyncStateChange}:{result:SourceResult;onSyncStateChange?:(state:LiveDataSyncState)=>void}) {
  const [range,setRange]=useState<RangeKey>("all"), [mode,setMode]=useState<LeaderboardMode>("winners");
  const [pagination,setPagination]=useState({cohort:"",page:0});
  const view=useMemo(()=>currentCombinedOverview(result),[result]);
  const loading=result.kind === "loading", stale=result.freshness === "stale", available=view !== null;
  const page=available && pagination.cohort === view.cohortId ? pagination.page : 0;
  const rows=view?.rows.slice(page*25,(page+1)*25) ?? [];
  const selected=view?.stats.find(row=>row.key === range);
  const partial=result.validity === "partial" || !!view?.coverage.unknown || !!view?.stats.some(row=>row.periodUnavailable);
  useEffect(()=>onSyncStateChange?.(loading ? "loading" : !available || stale || partial ? "error" : "ready"),[loading,available,stale,partial,onSyncStateChange]);
  const lifecycle=view ? {...view.strategy,kind:stale ? "stale" as const : view.strategy.kind} : {
    rows:[],kind:loading ? "loading" as const : "unavailable" as const,error:loading ? null : "Strategy history unavailable with the combined observation. Count and financial results are unknown.",
    omitted:0,sourceAsOf:null,observedAt:null,hasMore:null,legacyOverlap:null,
  };
  const failureReason = !available && !loading ? ({
    population_overflow: "Source population exceeds the read limit. Totals and rankings are withheld rather than based on truncated history.",
    cohort_changed: "Source records changed during the read. A coherent combined observation is unavailable.",
    source_busy: "The stored source is busy; no current combined observation was established.",
    source_timeout: "The stored source read timed out; no current combined observation was established.",
  } as Record<string,string>)[String(result.payload?.reason)] ?? "The combined source contract could not be verified. This is not an empty trading history." : null;
  const state=loading ? "Loading" : !available ? "Unavailable" : stale ? "Stale read" : partial ? "Partial evidence" : "Stored gross observation";
  const sourceCounts=view?.coverage.bySource as Record<string,{included:number;excluded:number;unknown:number}> | undefined;
  const note=loading ? "Loading the combined stored observation. Counts, totals and ranking remain unknown until the read completes." : available ? `${stale ? "Stale" : partial ? "Partial" : "Stored"} combined cohort | ${view.coverage.included} included / ${view.coverage.excluded} excluded / ${view.coverage.unknown} unknown. Gross before fees; no qualified net or portfolio return.` :
    "Combined gross unavailable. Counts, totals and ranking are unknown; no empty or zero result is implied.";
  return <section className="space-y-4" aria-label="Trades tab">
    <section className="min-w-0 rounded-2xl p-3 shadow-card jbravo-panel jbravo-panel-cyan sm:p-md" aria-label="Combined trade source">
      <div className="flex flex-wrap items-start justify-between gap-2"><h2 className="text-sm font-semibold uppercase tracking-wide text-primary">Trades · Combined stored gross</h2><span className="text-xs font-semibold text-secondary" role="status">{state}</span></div>
      <p className="mt-2 text-xs text-secondary">{note}</p>
      {failureReason ? <p className="mt-1 text-xs text-secondary" role="status">{failureReason}</p> : null}
      <p className="mt-1 text-xs text-secondary">Qualified net performance remains held; accounting qualification is not established. Fees and net P/L are unknown.</p>
      {stale && available ? <p className="mt-2 text-xs font-semibold text-secondary" role="status">Showing one retained cohort across all sections. Current source, lifecycle and policy bindings are unverified. A new read has not been established.</p> : null}
      <p className="mt-2 break-words text-[11px] text-secondary">Last successful read: {strategyTime(result.receivedAt ?? null)} | Source observation: {strategyTime(view?.observedAt ?? null)}</p>
      {result.refreshStatus === "unavailable" ? <p className="mt-1 text-[11px] text-secondary">Current source unavailable. Last attempt: {strategyTime(result.lastAttemptAt ?? null)}.</p> : null}
      <details className="mt-2 text-[11px] text-secondary"><summary className="cursor-pointer">Coverage, account scope and accounting limits</summary>
        <div className="mt-1 space-y-1 break-words">
          <p>Legacy account binding: {available ? "unverified" : "unknown"}. Strategy account binding: {available ? "verified for the configured paper account at the source read" : "unknown"}. This is not verified account-wide performance.</p>
          <p>Source: PostgreSQL. Stored execution evidence is not proof of live broker completeness. Latest included source time: {strategyTime(view?.sourceAsOf ?? null)}.</p>
          {sourceCounts ? <>{["legacy","strategy"].map(source=><p key={source}>{source === "legacy" ? "Legacy" : "Strategy"}: {sourceCounts[source].included} included / {sourceCounts[source].excluded} excluded / {sourceCounts[source].unknown} unknown.</p>)}</> : null}
          {view ? <p>{view.coverage.deduplicated} duplicate representations excluded; {view.coverage.strategyNotClosed} strategy signals are not closed realized candidates.</p> : null}
          <p>One bounded complete source population supplies all realized candidates, the five summaries and every symbol ranking. Unknown or excluded candidates are disclosed, not assumed to be zero. If population bounds or source validation fail, the combined result is unavailable.</p>
          <p>Weekly covers open US equity trading dates in the current Monday-Friday week; monthly covers the first through last open trading date of the current calendar month, in America/New_York. Both include only exits through this observation (period-to-date), including pre/postmarket on open dates. Daily and yearly remain rolling UTC 1 and 365 days; ALL has no date cutoff. The list retains all realized candidates. Leaderboard controls select those precomputed windows without another data read. Rankings aggregate all eligible symbol contributions before the top 12 limit.</p>
          <p>Legacy gross uses matched execution evidence; strategy gross uses validated stored cumulative order averages. Original recorded ledger P/L is separately labeled in row details. Gross win rate is a closed-contribution statistic, not a portfolio return or qualified strategy performance.</p>
          <p>Open and unfilled strategy signals remain lifecycle observations. Historical ownership labels do not establish current management authority. Account equity charts and trading permissions are unchanged.</p>
          {view ? <p className="break-all">Cohort: {view.cohortId}</p> : null}
        </div>
      </details>
    </section>
    <StrategyHistoryCard result={result} combinedView={lifecycle}/>
    <div className="grid min-w-0 grid-cols-1 gap-md xl:grid-cols-3 xl:items-stretch">
      <div className="contents xl:col-span-2 xl:block xl:space-y-4">
        <div className="order-1 min-w-0"><TradesPerformanceBoard gross unknownLabel="Unknown" rows={view?.stats ?? []} isLoading={loading} sourceNote={note} statusMessage="Combined performance unavailable; amounts and coverage are unknown."/></div>
        <div className="order-3 min-w-0"><CombinedLatestTrades rows={rows} isLoading={loading} available={available} stale={stale && available} page={page} totalRows={view?.rows.length ?? 0} onPageChange={next=>setPagination({cohort:view?.cohortId ?? "",page:next})}/></div>
      </div>
      <div className="order-2 xl:order-none xl:col-span-1 xl:h-full xl:min-h-0 min-w-0"><TradesLeaderboard gross rows={view?.rankings[range][mode] ?? []} selectedRange={range} onRangeChange={setRange} mode={mode} onModeChange={setMode}
        isLoading={loading} errorMessage={selected?.periodUnavailable ? "Trading calendar unavailable; selected period ranking is unknown." : !available && !loading ? "Leaderboard unavailable; ranking is unknown." : null}
        sourceNote={`${stale && available ? "Stale cohort. " : ""}Combined known gross before fees. ${selected?.periodLabel ?? ""}${selected?.coverage ? ` ${selected.coverage.included} included / ${selected.coverage.excluded} excluded / ${selected.coverage.unknown} unknown in this range.` : " Coverage unknown."} Unresolved records may change ranks.`}
        emptyMessage={selected?.coverage?.unknown ? "No known gross results in this range; unresolved records may change the ranking." : `No ${mode === "winners" ? "positive" : "negative"} known gross symbol totals in this stored range.`}/></div>
    </div>
  </section>;
}
