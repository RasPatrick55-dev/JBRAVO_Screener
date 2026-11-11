// Frozen legacy multi-endpoint composition from main 8138985 for its existing
// contract regressions. Not imported by the app. Current combined endpoint and
// composition are exercised by CombinedTradesOverview tests.
import { useEffect, useState } from "react";
import type { LiveDataSyncState } from "../../src/components/navbar/liveStatus";
import { canDisplay, finiteNumber, useSource, type SourceResult } from "../../src/components/ui/sourceState";
import TradesPerformanceBoard from "../../src/components/trades/TradesPerformanceBoard";
import TradesLeaderboard from "../../src/components/trades/TradesLeaderboard";
import LatestTradesTable from "../../src/components/trades/LatestTradesTable";
import StrategyHistoryCard from "../../src/components/trades/StrategyHistoryCard";
import { STRATEGY_HISTORY_PATH, strategyHistoryView, strategyHistoryReadPolicy } from "../../src/components/trades/strategyHistory";
import { allocationGrossRows } from "../../src/components/trades/AllocationGrossPanel";
import { buildLeaderboardRequestPath } from "../../src/components/trades/leaderboardData";
import { recordedGrossContract, normalizeRecordedGrossStats, normalizeRecordedGrossLeaders, coverageText, executionGrossReadPolicy } from "../../src/components/trades/recordedGrossData";
import type { RangeKey, LeaderboardMode, RangeRowMetrics, LatestTradeRow } from "../../src/components/trades/types";

import { physicalHistoryRows, normalizeStatsPayload } from "../../src/components/trades/TradesTab";
export { physicalHistoryRows, normalizeStatsPayload };
const object = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
const note = (label: string, result: SourceResult) => `${label} ${result.kind.toUpperCase()}${result.freshness === "stale" ? " (stale read)" : ""} | ${String(result.payload?.source ?? "source unknown")} | Last successful read UTC: ${result.receivedAt ?? "unknown"}${result.refreshStatus === "unavailable" ? " | Current source unavailable; any displayed values are a previous observation, not current performance. Current account/policy binding is unverified." : ""}`;

export default function TradesTab({ onSyncStateChange }: { onSyncStateChange?: (state: LiveDataSyncState) => void }) {
  const [range, setRange] = useState<RangeKey>("all");
  const [mode, setMode] = useState<LeaderboardMode>("winners");
  const [cursors, setCursors] = useState<number[]>([]);
  const cursor = cursors[cursors.length - 1];
  const stats = useSource("/api/trades/stats?range=all");
  const grossStats = useSource("/api/trades/stats?range=all&basis=execution_gross", undefined, undefined, undefined, executionGrossReadPolicy);
  const leaders = useSource(buildLeaderboardRequestPath(range, mode, 12, "execution_gross"), undefined, undefined, undefined, executionGrossReadPolicy);
  const history = useSource(`/api/trades/history?limit=25${cursor === undefined ? "" : `&before=${cursor}`}`);
  const strategyHistory = useSource(STRATEGY_HISTORY_PATH, undefined, undefined, undefined, strategyHistoryReadPolicy);
  const strategyKind = strategyHistoryView(strategyHistory).kind;
  const grossKind:SourceResult["kind"] = canDisplay(grossStats.kind) && !recordedGrossContract(grossStats.payload,"execution_gross") ? "unavailable" : grossStats.kind;
  const readyStats = canDisplay(grossKind) ? normalizeRecordedGrossStats(grossStats.payload,"execution_gross") : [];
  const leaderData = normalizeRecordedGrossLeaders(leaders.payload,"execution_gross");
  const leaderKind:SourceResult["kind"] = !canDisplay(leaders.kind) ? leaders.kind : !recordedGrossContract(leaders.payload,"execution_gross") ? "unavailable" :
    leaderData.rejected ? leaderData.rows.length ? "partial" : "unavailable" : leaders.kind;
  const readyLeaders = canDisplay(leaderKind) ? leaderData.rows : [];
  // Transport freshness must not erase retained business-validity/coverage limits.
  const unknownLeaders = object(leaders.payload?.coverage)?.unknown;
  const unresolvedLeaders = leaders.validity === "partial" || leaderData.rejected > 0 ||
    typeof unknownLeaders === "number" && unknownLeaders > 0;
  const rows = physicalHistoryRows(history, stats);
  const returnedCount = canDisplay(history.kind) && Array.isArray(history.payload?.rows) ? history.payload.rows.length : 0;
  const undisplayedCount = Math.max(0, returnedCount - rows.length);
  const incompleteHistory = undisplayedCount ? `${undisplayedCount} returned history ${undisplayedCount === 1 ? "record has an invalid identifier; its details are" : "records have invalid identifiers; their details are"} unavailable.` : null;
  useEffect(() => {
    const kinds = [grossKind, leaderKind, history.kind, strategyKind];
    onSyncStateChange?.(kinds.includes("loading") ? "loading" : kinds.every(kind => ["ready", "empty"].includes(kind)) ? "ready" : "error");
  }, [grossKind, leaderKind, history.kind, strategyKind, onSyncStateChange]);
  const netMessage = stats.kind === "held" ? "Performance held; accounting qualification is not established." : `Qualified net performance ${stats.kind}; recorded gross does not qualify net.`;
  const metricMessage = !canDisplay(grossKind) ? "Execution gross inputs unavailable; amounts and coverage are unknown." :
    !readyStats.length ? "Execution gross fields incomplete; values unknown." : null;
  return <section className="space-y-4" aria-label="Trades tab">
    <StrategyHistoryCard result={strategyHistory}/>
    <div className="grid min-w-0 grid-cols-1 gap-md xl:grid-cols-3 xl:items-stretch">
      <div className="contents xl:col-span-2 xl:block xl:space-y-4">
        <div className="order-1 min-w-0"><TradesPerformanceBoard gross rows={readyStats} isLoading={grossStats.kind === "loading"}
          sourceNote={`${note("Execution gross performance",{...grossStats,kind:grossKind})} | Before fees; known matched executions | ${coverageText(grossStats.payload)} | Last included exit UTC: ${String(grossStats.payload?.source_as_of ?? "unknown")} | UTC exit-time rolling windows | Qualified net: ${netMessage}`} statusMessage={metricMessage}/></div>
        <div className="order-3 min-w-0"><LatestTradesTable rows={rows} isLoading={history.kind === "loading"} plLabel="Recorded P/L"
          sourceNote={`Recorded history ${(undisplayedCount ? "partial" : history.kind).toUpperCase()}${history.freshness === "stale" ? " (stale read)" : ""} | PostgreSQL | Unqualified${incompleteHistory ? ` | ${incompleteHistory}` : ""}`}
          sourceDetails={<p>{note("History",history)} | Data as of UTC: {String(history.payload?.source_as_of ?? "unknown")} | Freshness: {String(history.payload?.freshness ?? "unknown")} | Broker association: {String(history.payload?.broker_account_match ?? "unverified")} | Allocation read UTC: {stats.receivedAt ?? "unknown"}; {stats.freshness}. History retains original Recorded P/L; summary and ranking use separately reconstructed execution gross. Derived values use a separate approved allocation of posted fees observed through October 25, 2026 20:54 UTC; they do not certify whole-history performance.</p>}
          emptyMessage={canDisplay(history.kind) ? incompleteHistory ?? "No recorded ledger trades in this page." : "History unavailable; ledger count is unknown."}
          footer={<div className="mt-2 flex gap-4 text-xs text-secondary">
            <button type="button" disabled={!cursors.length || history.kind === "loading"} onClick={() => setCursors(values => values.slice(0, -1))}>Previous history page</button>
            <button type="button" disabled={!history.payload?.has_more || history.kind === "loading" || !canDisplay(history.kind)} onClick={() => {
              const next = finiteNumber(history.payload?.next_before); if (next !== null) setCursors(values => [...values, next]);
            }}>Next history page</button>
          </div>}/></div>
      </div>
      <div className="order-2 xl:order-none xl:col-span-1 xl:h-full xl:min-h-0 min-w-0"><TradesLeaderboard gross rows={readyLeaders} selectedRange={range} onRangeChange={setRange} mode={mode} onModeChange={setMode}
        sourceNote={`${note("Execution gross leaderboard",{...leaders,kind:leaderKind})} | Before fees; known contributions, unresolved records may change ranks | ${coverageText(leaders.payload)}${leaderData.rejected ? ` | ${leaderData.rejected} malformed result rows omitted` : ""}`}
        emptyMessage={unresolvedLeaders ? "No known gross results in this ranking; unresolved records may change it." : `No ${mode==="winners"?"positive":"negative"} gross results in the selected period.`}
        isLoading={leaders.kind === "loading"} errorMessage={canDisplay(leaderKind) ? null : `Leaderboard ${leaderKind}; performance is not established.`}/></div>
    </div>
  </section>;
}
