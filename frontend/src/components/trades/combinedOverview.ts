import { canDisplay, type SourceResult, type SourceReadPolicy } from "../ui/sourceState";
import { strategyHistoryView, type StrategyHistoryView } from "./strategyHistory";
import type { LatestTradeRow, LeaderRow, LeaderboardMode, RangeKey, RangeRowMetrics } from "./types";

export const COMBINED_GROSS_PATH = "/api/trades/combined-gross";
export const COMBINED_SCHEMA = "jbravo.trades-overview.v1";
export const RANGES: RangeKey[] = ["d", "w", "m", "y", "all"];
type ObjectValue = Record<string, unknown>;
const object = (value: unknown): ObjectValue | null => value !== null && typeof value === "object" && !Array.isArray(value) ? value as ObjectValue : null;
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const symbol = (value: unknown): value is string => typeof value === "string" && /^[A-Z0-9][A-Z0-9./^\-]{0,31}$/.test(value);
const strategySymbol = (value: unknown): value is string => typeof value === "string" && /^[A-Z][A-Z0-9.\-]{0,15}$/.test(value);
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
export const combinedTimestamp = (value: unknown): value is string => {
  if (typeof value !== "string") return false;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,9})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.exec(value);
  if (!match || !Number.isFinite(Date.parse(value))) return false;
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  return year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= [31,leap ? 29 : 28,31,30,31,30,31,31,30,31,30,31][month-1];
};
const nullableTime = (value: unknown) => value === null || combinedTimestamp(value);
// Compare already validated wire clocks without dropping sub-millisecond digits.
// This is a chronology admission guard, never range membership or P/L arithmetic.
const preciseTime = (value: string): bigint => {
  const parts = /^(.*:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$/.exec(value)!;
  return BigInt(Date.parse(parts[1] + parts[3])) * 1000000n + BigInt((parts[2] ?? "").padEnd(9, "0"));
};
// Formatting conversion only. Python owns exact-decimal economics, lifecycle,
// order ownership, UTC range membership and every aggregate/rank.
const amount = (value: unknown): number | null => {
  if (typeof value !== "string" || value.length > 96 || !/^-?\d+(?:\.\d+)?$/.test(value)) return null;
  const number = Number(value);
  return Number.isFinite(number) && Math.abs(number) <= Number.MAX_SAFE_INTEGER ? number : null;
};
const nullableAmount = (value: unknown) => value === null || amount(value) !== null;
const nullableText = (value: unknown) => value === null || typeof value === "string" && value.length > 0 && value.length <= 256;
export type CombinedCoverage = {included:number; excluded:number; unknown:number; deduplicated:number; strategyNotClosed:number; bySource: ObjectValue};
export type CombinedTradeRow = LatestTradeRow & {
  rowId:string; sourceKind:"LEGACY_EXECUTION" | "STRATEGY_EXECUTION";
  inclusion:"INCLUDED" | "POLICY_EXCLUDED" | "UNKNOWN"; reason:string|null;
  accountBinding:"verified" | "unverified"; recordedPL:number|null;
  priceBasis:string; ownershipScope:string|null; recordedEntryTime:string|null; recordedExitTime:string|null;
};
export type CombinedOverview = {
  cohortId:string; rows:CombinedTradeRow[]; stats:RangeRowMetrics[];
  rankings:Record<RangeKey, Record<LeaderboardMode, LeaderRow[]>>;
  strategy:StrategyHistoryView; coverage:CombinedCoverage; observedAt:string; sourceAsOf:string|null;
};
const coverage = (value: unknown): {included:number;excluded:number;unknown:number} | null => {
  const data = object(value);
  return data && count(data.included) && count(data.excluded) && count(data.unknown) ?
    {included:data.included,excluded:data.excluded,unknown:data.unknown} : null;
};

// Reject the whole observation on malformed/duplicate identities or aggregates.
// Omitting one bad list row while displaying its total would recreate divergence.
export function combinedOverview(payload: ObjectValue | null): CombinedOverview | null {
  const counts = coverage(payload?.coverage), bySource = object(object(payload?.coverage)?.by_source);
  if (!payload || payload.ok !== true || payload.schema_version !== COMBINED_SCHEMA || payload.source !== "postgres" ||
      payload.view_scope !== "COMBINED_STORED_EXECUTION_GROSS" || payload.performance_status !== "COMBINED_GROSS_ONLY" ||
      payload.recovery_complete !== false || payload.entry_ready !== false || payload.qualified_net_status !== "NOT_ESTABLISHED" || payload.net_eligible !== false || payload.accounting_complete !== false ||
      payload.fees !== null || payload.net_pnl !== null || payload.fees_status !== "UNKNOWN" || payload.net_status !== "UNKNOWN" ||
      payload.strategy_account_binding !== "verified" || payload.legacy_account_binding !== "unverified" ||
      payload.population_complete !== true || payload.policy_validation_status !== "VALIDATED_POLICY" || payload.source_freshness !== "STORED_EVIDENCE_NOT_LIVE_BROKER_COMPLETENESS" ||
      !["ready","partial","empty"].includes(String(payload.status)) || !hash(payload.cohort_id) ||
      !combinedTimestamp(payload.observed_at) || !nullableTime(payload.source_as_of) ||
      payload.source_as_of !== null && Date.parse(payload.source_as_of as string) > Date.parse(payload.observed_at as string) || !counts || !bySource ||
      !coverage(bySource.legacy) || !coverage(bySource.strategy) || !count(object(payload.coverage)?.deduplicated) || !count(object(payload.coverage)?.strategy_not_closed) ||
      !Array.isArray(payload.rows) || payload.rows.length > 10000 || !Array.isArray(payload.stats) || payload.stats.length !== 5) return null;
  if (counts.included > 0 && payload.source_as_of === null) return null;
  const legacyCoverage = coverage(bySource.legacy)!, strategyCoverage = coverage(bySource.strategy)!;
  if (["included","excluded","unknown"].some(key=>counts[key as keyof typeof counts] !== legacyCoverage[key as keyof typeof legacyCoverage] + strategyCoverage[key as keyof typeof strategyCoverage])) return null;
  const rows:CombinedTradeRow[] = [], seen = new Set<string>();
  for (const value of payload.rows) {
    const row = object(value);
    if (!row || typeof row.row_id !== "string" || seen.has(row.row_id) ||
        !(symbol(row.symbol) || row.source_kind === "LEGACY_EXECUTION" && row.inclusion !== "INCLUDED" && row.symbol === null) ||
        !["LEGACY_EXECUTION","STRATEGY_EXECUTION"].includes(String(row.source_kind)) ||
        !["INCLUDED","POLICY_EXCLUDED","UNKNOWN"].includes(String(row.inclusion)) ||
        typeof row.state !== "string" || !row.state || !nullableText(row.reason) || !nullableText(row.ownership_scope) ||
        !["RECORDED_LEDGER","STORED_ORDER_FILL_AVERAGES"].includes(String(row.price_basis)) ||
        ![row.entry_time,row.exit_time,row.recorded_entry_time,row.recorded_exit_time].every(nullableTime) ||
        ![row.qty,row.entry_price,row.exit_price,row.gross_pnl,row.recorded_pnl].every(nullableAmount) ||
        row.fees !== null || row.net_pnl !== null || !(row.holding_days === null || count(row.holding_days))) return null;
    const legacy = row.source_kind === "LEGACY_EXECUTION";
    if (legacy ? !count(row.trade_id) || row.trade_id === 0 || row.signal_id !== null || row.row_id !== `legacy:${row.trade_id}` || row.account_binding !== "unverified" :
        !hash(row.signal_id) || !strategySymbol(row.symbol) || row.trade_id !== null || row.row_id !== `strategy:${row.signal_id}` || row.account_binding !== "verified") return null;
    if (row.price_basis !== (legacy ? "RECORDED_LEDGER" : "STORED_ORDER_FILL_AVERAGES")) return null;
    if (row.inclusion === "INCLUDED" && (row.state !== "CLOSED" || row.gross_pnl === null || row.entry_time === null || row.exit_time === null ||
        preciseTime(row.entry_time as string) > preciseTime(row.exit_time as string)) ||
        row.inclusion !== "INCLUDED" && row.gross_pnl !== null) return null;
    seen.add(row.row_id);
    rows.push({rowId:row.row_id,sourceKind:row.source_kind as CombinedTradeRow["sourceKind"],symbol:row.symbol as string ?? "Unknown",
      inclusion:row.inclusion as CombinedTradeRow["inclusion"],reason:row.reason as string|null,accountBinding:row.account_binding as CombinedTradeRow["accountBinding"],
      status:row.state,tradeId:legacy ? row.trade_id as number : undefined,
      buyDate:row.entry_time as string|null,sellDate:row.exit_time as string|null,totalDays:row.holding_days as number|null,
      totalShares:amount(row.qty),avgEntryPrice:amount(row.entry_price),priceSold:amount(row.exit_price),totalPL:amount(row.gross_pnl),
      recordedPL:amount(row.recorded_pnl),priceBasis:row.price_basis as string,ownershipScope:row.ownership_scope as string|null,
      recordedEntryTime:row.recorded_entry_time as string|null,recordedExitTime:row.recorded_exit_time as string|null});
  }
  if (rows.length !== counts.included + counts.excluded + counts.unknown ||
      rows.filter(row=>row.inclusion === "INCLUDED").length !== counts.included ||
      rows.filter(row=>row.inclusion === "POLICY_EXCLUDED").length !== counts.excluded) return null;
  const coverageFields = ["included", "excluded", "unknown"] as const;
  const inclusionFields = ["INCLUDED", "POLICY_EXCLUDED", "UNKNOWN"] as const;
  for (const [kind, expected] of [["LEGACY_EXECUTION", legacyCoverage], ["STRATEGY_EXECUTION", strategyCoverage]] as const) {
    const group = rows.filter(row=>row.sourceKind === kind);
    if (coverageFields.some((key,index)=>expected[key] !== group.filter(row=>row.inclusion === inclusionFields[index]).length)) return null;
  }
  const sameCoverage = (value:unknown, expected:typeof counts) => {
    const actual=coverage(value);
    return actual !== null && coverageFields.every(key=>actual[key] === expected[key]);
  };
  const includedSymbols = new Set(rows.filter(row=>row.inclusion === "INCLUDED").map(row=>row.symbol));
  const stats:RangeRowMetrics[] = [], seenRanges = new Set<string>();
  for (const value of payload.stats) {
    const row = object(value), cov = coverage(row?.coverage), top = object(row?.topTrade), worst = object(row?.worstLoss);
    if (!row || !RANGES.includes(row.key as RangeKey) || seenRanges.has(String(row.key)) || !cov || !top || !worst ||
        !count(row.tradesCount) || row.tradesCount !== cov.included || !nullableAmount(row.totalPL) ||
        !(row.winRatePct === null || typeof row.winRatePct === "number" && Number.isFinite(row.winRatePct) && row.winRatePct >= 0 && row.winRatePct <= 100) ||
        ![top,worst].every(item => (item.symbol === null || item.symbol === "--" || symbol(item.symbol)) && nullableAmount(item.pl))) return null;
    const key = row.key as RangeKey;
    const label = {d:"DAILY",w:"WEEKLY",m:"MONTHLY",y:"YEARLY",all:"ALL"}[key] as RangeRowMetrics["label"];
    if (row.label !== label || cov.included > 0 && (row.totalPL === null || row.winRatePct === null || top.pl === null || worst.pl === null || !symbol(top.symbol) || !symbol(worst.symbol)) ||
        cov.included === 0 && (row.winRatePct !== null || top.pl !== null || worst.pl !== null || (cov.unknown > 0 ? row.totalPL !== null : amount(row.totalPL) !== 0))) return null;
    if (key === "all") {
      const details=object(row.coverage)!, sources=object(details.by_source);
      if (!sameCoverage(cov,counts) || !sources || !sameCoverage(sources.legacy,legacyCoverage) || !sameCoverage(sources.strategy,strategyCoverage) ||
          details.deduplicated !== object(payload.coverage)!.deduplicated || details.strategy_not_closed !== object(payload.coverage)!.strategy_not_closed) return null;
    }
    if (cov.included > 0 && (!includedSymbols.has(String(top.symbol)) || !includedSymbols.has(String(worst.symbol)))) return null;
    seenRanges.add(key);
    stats.push({key,label,coverage:cov,tradesCount:row.tradesCount,totalPL:amount(row.totalPL),winRatePct:row.winRatePct as number|null,
      topTrade:{symbol:top.symbol as string ?? "--",pl:amount(top.pl)},worstLoss:{symbol:worst.symbol as string ?? "--",pl:amount(worst.pl)}});
  }
  const rankings = {} as CombinedOverview["rankings"], ranks = object(payload.rankings);
  if (!ranks) return null;
  for (const range of RANGES) {
    const group = object(ranks[range]); if (!group) return null;
    rankings[range] = {winners:[],losers:[]};
    for (const mode of ["winners","losers"] as const) {
      const list = group[mode]; if (!Array.isArray(list) || list.length > 12) return null;
      const symbols = new Set<string>();
      for (const value of list) {
        const row = object(value), pl = amount(row?.pl);
        if (!row || !symbol(row.symbol) || !includedSymbols.has(row.symbol) || symbols.has(row.symbol) || row.rank !== rankings[range][mode].length + 1 ||
            pl === null || (mode === "winners" ? pl <= 0 : pl >= 0)) return null;
        symbols.add(row.symbol); rankings[range][mode].push({rank:row.rank as number,symbol:row.symbol,pl});
      }
    }
  }
  const nested = object(payload.strategy_history);
  if (!nested || nested.observed_at !== payload.observed_at || !Array.isArray(nested.rows) ||
      !nested.rows.every(value=>{
        const row=object(value);
        return row && ["INCLUDED","EXCLUDED","UNKNOWN","NOT_CLOSED"].includes(String(row.combined_inclusion)) && nullableText(row.combined_reason);
      })) return null;
  const strategy = strategyHistoryView({payload:nested,kind:"ready",validity:"ready",freshness:"current"});
  if (!canDisplay(strategy.kind) || strategy.omitted > 0 || strategy.rows.some(row=>row.combinedInclusion === "INCLUDED" && (row.status !== "complete" || row.gross === null))) return null;
  return {cohortId:payload.cohort_id,rows,stats,rankings,strategy,coverage:{...counts,deduplicated:object(payload.coverage)!.deduplicated as number,strategyNotClosed:object(payload.coverage)!.strategy_not_closed as number,bySource},observedAt:payload.observed_at,sourceAsOf:payload.source_as_of as string|null};
}
const unavailableKeys = new Set(["schema_version","ok","status","reason","cohort_id","source","view_scope","performance_status",
  "strategy_account_binding","legacy_account_binding","policy_validation_status","population_complete","observed_at","source_as_of","source_freshness",
  "qualified_net_status","net_eligible","accounting_complete","recovery_complete","entry_ready","fees","net_pnl","fees_status","net_status",
  "coverage","rows","stats","rankings","strategy_history","failure_stage"]);
export const combinedGrossReadPolicy:SourceReadPolicy = {
  accepts: payload => combinedOverview(payload) !== null,
  // Only two explicitly transient failures may retain one unchanged observation.
  // Identity/policy/overflow/changed-cohort/malformed envelopes always hide it.
  retainUnavailable: p => !!p && Object.keys(p).length === unavailableKeys.size && Object.keys(p).every(key=>unavailableKeys.has(key)) &&
    p.schema_version === COMBINED_SCHEMA && p.ok === false && p.status === "unavailable" &&
    ["source_busy","source_timeout"].includes(String(p.reason)) && ["broker_account_read","database_cohort_read"].includes(String(p.failure_stage)) &&
    p.source === "postgres" && p.view_scope === "COMBINED_STORED_EXECUTION_GROSS" && p.performance_status === "COMBINED_GROSS_ONLY" &&
    p.strategy_account_binding === "unavailable" && p.legacy_account_binding === "unverified" && p.policy_validation_status === "UNAVAILABLE" &&
    p.population_complete === false && combinedTimestamp(p.observed_at) && p.source_as_of === null &&
    p.source_freshness === "STORED_EVIDENCE_NOT_LIVE_BROKER_COMPLETENESS" && p.qualified_net_status === "NOT_ESTABLISHED" &&
    p.net_eligible === false && p.accounting_complete === false && p.recovery_complete === false && p.entry_ready === false &&
    p.fees === null && p.net_pnl === null && p.fees_status === "UNKNOWN" && p.net_status === "UNKNOWN" &&
    p.coverage === null && p.cohort_id === null && p.strategy_history === null &&
    Array.isArray(p.rows) && p.rows.length === 0 && Array.isArray(p.stats) && p.stats.length === 0 &&
    object(p.rankings) !== null && Object.keys(p.rankings as object).length === 0,
};
export function currentCombinedOverview(result:SourceResult) {
  return canDisplay(result.kind) && result.freshness !== "unknown" ? combinedOverview(result.payload) : null;
}
