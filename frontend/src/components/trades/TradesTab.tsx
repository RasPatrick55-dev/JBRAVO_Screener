import type { LiveDataSyncState } from "../navbar/liveStatus";
import { canDisplay, finiteNumber, useSource, type SourceResult } from "../ui/sourceState";
import { allocationGrossRows } from "./AllocationGrossPanel";
import type { RangeRowMetrics, LatestTradeRow } from "./types";
import { COMBINED_GROSS_PATH, combinedGrossReadPolicy } from "./combinedOverview";
import CombinedTradesView from "./CombinedTradesView";

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
export default function TradesTab({ onSyncStateChange }: { onSyncStateChange?: (state: LiveDataSyncState) => void }) {
  const result = useSource(COMBINED_GROSS_PATH, undefined, undefined, undefined, combinedGrossReadPolicy);
  return <CombinedTradesView result={result} onSyncStateChange={onSyncStateChange}/>;
}
