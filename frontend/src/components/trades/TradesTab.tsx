import { useEffect, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import { canDisplay, finiteNumber, useSource, type SourceResult } from "../ui/sourceState";
import TradesPerformanceBoard from "./TradesPerformanceBoard";
import TradesLeaderboard from "./TradesLeaderboard";
import LatestTradesTable from "./LatestTradesTable";
import { allocationGrossRows } from "./AllocationGrossPanel";
import { buildLeaderboardRequestPath } from "./leaderboardData";
import { recordedGrossContract, normalizeRecordedGrossStats, normalizeRecordedGrossLeaders, coverageText } from "./recordedGrossData";
import type { RangeKey, LeaderboardMode, RangeRowMetrics, LatestTradeRow } from "./types";

// API percentages already have percent units: 1 means 1%, never 100%.
export function normalizeStatsPayload(payload: Record<string, unknown> | null): RangeRowMetrics[] {
  if (!payload || !Array.isArray(payload.rows)) return [];
  return payload.rows.filter((value): value is RangeRowMetrics => {
    if (!value || typeof value !== "object") return false;
    const row = value as RangeRowMetrics;
    return ["d", "w", "m", "y", "all"].includes(row.key) &&
      finiteNumber(row.winRatePct) !== null && finiteNumber(row.totalPL) !== null &&
      finiteNumber(row.tradesCount) !== null && finiteNumber(row.topTrade?.pl) !== null &&
      finiteNumber(row.worstLoss?.pl) !== null;
  });
}

const utc = (value: unknown): string | null => {
  if (typeof value !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/.test(value) || !Number.isFinite(Date.parse(value))) return null;
  const calendar = /^(\d{4})-(\d{2})-(\d{2})[T ]/.exec(value);
  if (!calendar) return null;
  const year = Number(calendar[1]), month = Number(calendar[2]), day = Number(calendar[3]);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (month < 1 || month > 12 || day < 1 || day > [31,leap ? 29 : 28,31,30,31,30,31,31,30,31,30,31][month-1]) return null;
  return new Date(value).toISOString();
};
const object = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
const digest = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const revision = (value: unknown, zero = false): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= (zero ? 0 : 1);
const decimal = (value: unknown): string | null => {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const text = String(value); if (!/^-?\d+(?:\.\d+)?$/.test(text)) return null;
  const [integer, fraction=""] = text.split(".");
  return integer.replace(/^(-?)0+(?=\d)/,"$1") + (fraction.replace(/0+$/,"") ? "."+fraction.replace(/0+$/,"") : "");
};
const scopedDerived = (row: Record<string, unknown>, id: number, history: SourceResult, stats: SourceResult): LatestTradeRow["derivedObservedNet"] => {
  const value = row.derived_accounting;
  if (id !== 110 || !value || typeof value !== "object") return null;
  const data = value as Record<string, unknown>;
  if (history.freshness !== "current" || stats.freshness !== "current" || stats.payload?.policy_validation_status !== "VALIDATED_POLICY") return null;
  const pinned = data.status === "APPROVED_SCOPED_OBSERVED_DERIVATION" && data.source === "postgres_allocation_pinned_observed_costs";
  const registered = data.status === "APPROVED_SCOPED_DERIVED_RECORD" && data.source === "postgres_derived_accounting";
  if ((!pinned && !registered) || data.trade_id !== id ||
      data.basis !== "CRC110_ALLOCATED_OBSERVED_POSTED_COST_NET_V1" || data.rule !== "posted_daily_taf_sell_share_largest_remainder_v1" ||
      data.execution_binding !== "EXACT_COMMON_FIELDS_AND_THREE_ORDER_HASHES" || data.legacy_validated_performance_eligible !== false ||
      !digest(data.record_sha256)) return null;
  const binding = object(data.binding), bound = object(data.bound_row), context = object(history.payload?.derived_read_context);
  const current = object(context?.binding);
  if (!binding || !bound || !current || !context || context.schema_version !== 1 ||
      (pinned ? context.source !== "postgres_current_observed_cost_read" || context.evidence_status !== "PINNED_OWNER_APPROVED" ||
        data.canonical_account_uuid_proved !== false || data.global_accounting_complete !== false || data.entry_ready !== false :
        context.source !== "postgres_current_derived_read" || context.registration_status !== "ACTIVE") ||
      context.status !== "CURRENT_BINDINGS_VERIFIED" || context.allocation_state !== "ACTIVE" ||
      !utc(context.checked_at) || !utc(history.receivedAt)) return null;
  const age = Date.parse(history.receivedAt!) - Date.parse(String(context.checked_at));
  if (age < -5000 || age > 60000) return null;
  const bindingKeys = ["trade_id","policy_revision","allocation_id","allocation_revision","manifest_sha256","fills_sha256","row_fingerprint","row_revision",
    "evidence_file_sha256","rule_sha256","cutoff","record_sha256",...(pinned ? ["source_binding_sha256"] : ["registration_revision","correction_revision"])];
  if (bindingKeys.some(key => binding[key] === undefined || binding[key] !== current[key]) || binding.trade_id !== id ||
      !revision(binding.policy_revision) || !revision(binding.allocation_revision) ||
      (pinned ? binding.source_binding_sha256 !== "ea561075fff5dc22fd245df979d95ef48be1de3568923499de1ecb258cc3b0d5" :
        !revision(binding.registration_revision) || !revision(binding.correction_revision,true)) ||
      typeof binding.allocation_id !== "string" || !binding.allocation_id || binding.record_sha256 !== data.record_sha256 ||
      binding.cutoff !== data.as_of || utc(binding.cutoff) !== "2026-10-25T20:54:00.000Z" ||
      binding.evidence_file_sha256 !== "6856c384cd0d76d185a2b312f020f4f9e665969ad0182f2a18925b91e3f5e01b" ||
      binding.rule_sha256 !== "51f51f1b83c3e1e60d6f5df3ecf6e580e35a49db5c559735c55281c2ffd56f53" || binding.row_revision !== row.updated_at || binding.row_fingerprint !== row.row_fingerprint ||
      [binding.manifest_sha256,binding.fills_sha256,binding.row_fingerprint,binding.evidence_file_sha256,binding.rule_sha256].some(value=>!digest(value))) return null;
  if (row.symbol !== "CRC" || row.status !== "CLOSED" || bound.symbol !== row.symbol || bound.status !== row.status || bound.trade_id !== id ||
      bound.row_fingerprint !== row.row_fingerprint || bound.updated_at !== row.updated_at) return null;
  for (const key of ["entry_time","exit_time"]) if (!utc(row[key]) || bound[key] !== row[key]) return null;
  for (const key of ["qty","entry_price","exit_price","realized_pnl"]) {
    if (!(key in row) || !(key in bound) || (row[key] === null ? bound[key] !== null : decimal(row[key]) === null || decimal(row[key]) !== decimal(bound[key]))) return null;
  }
  const policy = object(object(stats.payload?.trade_policy)?.policy), overlay = object(policy?.allocation_overlay);
  if (policy?.scope !== "VALIDATED_PERFORMANCE" || policy.policy_revision !== binding.policy_revision || overlay?.status !== "VALIDATED" ||
      overlay.policy_revision !== binding.policy_revision || !Array.isArray(overlay.allocations)) return null;
  const allocations = overlay.allocations.map(object).filter(allocation=>allocation?.trade_id === id);
  const allocation = allocations[0];
  if (allocations.length !== 1 || !allocation || allocation.state !== "ACTIVE" || allocation.allocation_id !== binding.allocation_id ||
      allocation.allocation_revision !== binding.allocation_revision || decimal(allocation.entry_quantity) !== decimal(row.qty) ||
      decimal(allocation.exit_quantity) !== decimal(row.qty)) return null;
  const net = data.net_pnl, fees = data.allocated_fee_cost, asOf = utc(data.as_of);
  if (typeof net !== "string" || typeof fees !== "string" || net.length > 96 || fees.length > 96 ||
      !/^-?\d+(?:\.\d+)?$/.test(net) || !/^\d+(?:\.\d+)?$/.test(fees) || !asOf) return null;
  return {net,fees,asOf};
};
export function physicalHistoryRows(result: SourceResult, stats: SourceResult): LatestTradeRow[] {
  if (!canDisplay(result.kind) || !Array.isArray(result.payload?.rows)) return [];
  const gross = allocationGrossRows(stats)?.rows ?? [];
  return result.payload.rows.flatMap(value => {
    if (!value || typeof value !== "object") return [];
    const row = value as Record<string, unknown>;
    const id = finiteNumber(row.trade_id), symbol = typeof row.symbol === "string" ? row.symbol.trim() : "";
    if (id === null || !Number.isSafeInteger(id) || id <= 0) return [];
    const entry = utc(row.entry_time), exit = utc(row.exit_time);
    const days = entry && exit && Date.parse(exit) >= Date.parse(entry) ?
      (Date.parse(exit.slice(0,10)+"T00:00:00Z")-Date.parse(entry.slice(0,10)+"T00:00:00Z"))/86400000 : null;
    const allocation = gross.find(item => item.tradeId === id && finiteNumber(row.qty) === finiteNumber(item.entryQuantity) && row.status === "CLOSED");
    return [{tradeId:id, symbol:symbol || "Unknown", status:String(row.status ?? "Unknown"), buyDate:entry, sellDate:exit,
      totalDays:days, totalShares:finiteNumber(row.qty), avgEntryPrice:finiteNumber(row.entry_price),
      priceSold:finiteNumber(row.exit_price), totalPL:finiteNumber(row.realized_pnl),
      grossPL:allocation?.gross ?? null, grossStale:stats.freshness === "stale", disposition:String(row.disposition ?? "UNKNOWN"),
      derivedObservedNet:scopedDerived(row,id,result,stats)}];
  });
}
const note = (label: string, result: SourceResult) => `${label} ${result.kind.toUpperCase()}${result.freshness === "stale" ? " (stale read)" : ""} | ${String(result.payload?.source ?? "source unknown")} | Last successful read UTC: ${result.receivedAt ?? "unknown"}`;

export default function TradesTab({ onSyncStateChange }: { onSyncStateChange?: (state: LiveDataSyncState) => void }) {
  const [range, setRange] = useState<RangeKey>("all");
  const [mode, setMode] = useState<LeaderboardMode>("winners");
  const [cursors, setCursors] = useState<number[]>([]);
  const cursor = cursors[cursors.length - 1];
  const stats = useSource("/api/trades/stats?range=all");
  const grossStats = useSource("/api/trades/stats?range=all&basis=recorded_gross");
  const leaders = useSource(buildLeaderboardRequestPath(range, mode, 12, "recorded_gross"));
  const history = useSource(`/api/trades/history?limit=25${cursor === undefined ? "" : `&before=${cursor}`}`);
  const grossKind:SourceResult["kind"] = canDisplay(grossStats.kind) && !recordedGrossContract(grossStats.payload) ? "unavailable" : grossStats.kind;
  const readyStats = canDisplay(grossKind) ? normalizeRecordedGrossStats(grossStats.payload) : [];
  const leaderData = normalizeRecordedGrossLeaders(leaders.payload);
  const leaderKind:SourceResult["kind"] = !canDisplay(leaders.kind) ? leaders.kind : !recordedGrossContract(leaders.payload) ? "unavailable" :
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
    const kinds = [grossKind, leaderKind, history.kind];
    onSyncStateChange?.(kinds.includes("loading") ? "loading" : kinds.every(kind => ["ready", "empty"].includes(kind)) ? "ready" : "error");
  }, [grossKind, leaderKind, history.kind, onSyncStateChange]);
  const netMessage = stats.kind === "held" ? "Performance held; accounting qualification is not established." : `Qualified net performance ${stats.kind}; recorded gross does not qualify net.`;
  const metricMessage = !canDisplay(grossKind) ? "Recorded gross inputs unavailable; amounts and coverage are unknown." :
    !readyStats.length ? "Recorded gross fields incomplete; values unknown." : null;
  return <section className="space-y-4" aria-label="Trades tab">
    <div className="grid gap-md xl:grid-cols-3 xl:items-stretch">
      <div className="contents xl:col-span-2 xl:block xl:space-y-4">
        <div className="order-1"><TradesPerformanceBoard gross rows={readyStats} isLoading={grossStats.kind === "loading"}
          sourceNote={`${note("Recorded gross performance",{...grossStats,kind:grossKind})} | Before fees | ${coverageText(grossStats.payload)} | UTC exit-time rolling windows | Qualified net: ${netMessage}`} statusMessage={metricMessage}/></div>
        <div className="order-3"><LatestTradesTable rows={rows} isLoading={history.kind === "loading"} plLabel="Recorded P/L"
          sourceNote={`Recorded history ${(undisplayedCount ? "partial" : history.kind).toUpperCase()}${history.freshness === "stale" ? " (stale read)" : ""} | PostgreSQL | Unqualified${incompleteHistory ? ` | ${incompleteHistory}` : ""}`}
          sourceDetails={<p>{note("History",history)} | Data as of UTC: {String(history.payload?.source_as_of ?? "unknown")} | Freshness: {String(history.payload?.freshness ?? "unknown")} | Broker association: {String(history.payload?.broker_account_match ?? "unverified")} | Allocation read UTC: {stats.receivedAt ?? "unknown"}; {stats.freshness}. Derived values use a separate approved allocation of posted fees observed through October 25, 2026 20:54 UTC; they do not certify whole-history performance.</p>}
          emptyMessage={canDisplay(history.kind) ? incompleteHistory ?? "No recorded ledger trades in this page." : "History unavailable; ledger count is unknown."}
          footer={<div className="mt-2 flex gap-4 text-xs text-secondary">
            <button type="button" disabled={!cursors.length || history.kind === "loading"} onClick={() => setCursors(values => values.slice(0, -1))}>Previous history page</button>
            <button type="button" disabled={!history.payload?.has_more || history.kind === "loading" || !canDisplay(history.kind)} onClick={() => {
              const next = finiteNumber(history.payload?.next_before); if (next !== null) setCursors(values => [...values, next]);
            }}>Next history page</button>
          </div>}/></div>
      </div>
      <div className="order-2 xl:order-none xl:col-span-1 xl:h-full xl:min-h-0"><TradesLeaderboard gross rows={readyLeaders} selectedRange={range} onRangeChange={setRange} mode={mode} onModeChange={setMode}
        sourceNote={`${note("Recorded gross leaderboard",{...leaders,kind:leaderKind})} | Before fees | ${coverageText(leaders.payload)}${leaderData.rejected ? ` | ${leaderData.rejected} malformed result rows omitted` : ""}`}
        emptyMessage={unresolvedLeaders ? "No known gross results in this ranking; unresolved records may change it." : `No ${mode==="winners"?"positive":"negative"} gross results in the selected period.`}
        isLoading={leaders.kind === "loading"} errorMessage={canDisplay(leaderKind) ? null : `Leaderboard ${leaderKind}; performance is not established.`}/></div>
    </div>
  </section>;
}
