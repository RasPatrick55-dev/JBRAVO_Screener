import { finiteNumber, type SourcePayload, type SourceReadPolicy } from "../ui/sourceState";
import type { LeaderRow, RangeRowMetrics } from "./types";
import type { GrossBasis } from "./leaderboardData";
const stamp=(value:unknown):number|null=>{
  if(typeof value!=="string")return null;
  const parts=/^(\d{4})-(\d{2})-(\d{2})T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,6})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.exec(value);
  if(!parts)return null;
  const year=Number(parts[1]),month=Number(parts[2]),day=Number(parts[3]),leap=year%4===0&&(year%100!==0||year%400===0);
  if(year<1||month<1||month>12||day<1||day>[31,leap?29:28,31,30,31,30,31,31,30,31,30,31][month-1])return null;
  const parsed=Date.parse(value);return Number.isFinite(parsed)?parsed:null;
};

export const recordedGrossContract=(payload:SourcePayload|null,basis:GrossBasis="recorded_gross")=> {
  const coverage=object(payload?.coverage);
  const counts=coverage ? [coverage.included,coverage.excluded,coverage.unknown] : [];
  const observed=stamp(payload?.observed_at),asof=stamp(payload?.source_as_of);
  const expected=basis==="execution_gross" ? payload?.view_scope==="EXECUTION_RECONSTRUCTED_GROSS" &&
    payload.pnl_basis==="POSTGRES_MATCHED_EXECUTION_GROSS_V1" && payload.performance_status==="EXECUTION_GROSS_ONLY" &&
    payload.fees===null && payload.net_pnl===null && payload.accounting_complete===false && payload.entry_ready===false &&
    typeof payload.execution_fingerprint==="string" && /^[a-f0-9]{64}$/.test(payload.execution_fingerprint) &&
    typeof payload.policy_fingerprint==="string" && /^[a-f0-9]{64}$/.test(payload.policy_fingerprint) &&
    typeof payload.input_fingerprint==="string" && /^[a-f0-9]{64}$/.test(payload.input_fingerprint) &&
    observed!==null && (payload.source_as_of===null ? coverage?.included===0 : asof!==null && asof<=observed) &&
    payload.source_freshness==="STORED_EXECUTIONS_NOT_BROKER_COMPLETENESS_PROOF" :
    payload?.view_scope==="RECORDED_GROSS" && payload.pnl_basis==="PAPER_GROSS_RECORDED_PRICES_AND_AUDITED_FILLS_V1";
  return expected && payload?.source==="postgres" && payload.net_eligible===false &&
  payload.fees_status==="UNKNOWN" && payload.net_status==="UNKNOWN" && payload.policy_validation_status==="VALIDATED_POLICY" &&
  Array.isArray(payload.rows) && counts.length===3 && counts.every(value=>typeof value==="number" && Number.isSafeInteger(value) && value>=0) &&
  typeof payload.closed_population_count==="number" && Number.isSafeInteger(payload.closed_population_count) &&
  payload.closed_population_count >= (counts as number[]).reduce((a,b)=>a+b,0);
};
const object=(value:unknown):Record<string,unknown>|null=>value!==null && typeof value==="object" && !Array.isArray(value)?value as Record<string,unknown>:null;
const amount=(value:unknown):number|null=>(typeof value==="number" || typeof value==="string" && /^-?\d+(?:\.\d+)?$/.test(value))?finiteNumber(value):null;
export function coverageText(payload:SourcePayload|null):string {
  const coverage=object(payload?.coverage);
  if(!coverage)return "Coverage unknown";
  const counts=[coverage.included,coverage.excluded,coverage.unknown];
  if(counts.some(value=>typeof value!=="number" || !Number.isSafeInteger(value) || value<0))return "Coverage unknown";
  const labels:Record<string,string>={SYMBOL_UNKNOWN:"missing symbol",TIME_UNKNOWN_OR_INVALID:"missing/invalid trade dates",
    QTY_UNKNOWN_OR_INVALID:"missing/invalid quantity",ENTRY_PRICE_UNKNOWN_OR_INVALID:"missing/invalid entry price",EXIT_PRICE_UNKNOWN_OR_INVALID:"missing/invalid exit price",
    ORDER_OWNER_AMBIGUOUS:"ambiguous order ownership",QUANTITY_MISMATCH:"execution quantity mismatch",ORDER_FILL_MISSING:"missing linked fills",
    SYMBOL_MISMATCH:"execution symbol mismatch",FILL_ECONOMICS_UNRESOLVED:"unusable fill evidence",
    SIDE_OR_LIFECYCLE_UNRESOLVED:"unresolved execution lifecycle",EXECUTION_CHRONOLOGY_UNRESOLVED:"unresolved execution dates",
    ORDER_OR_EXECUTION_INPUT_UNAVAILABLE:"missing order/execution evidence"};
  const reasons=object(coverage.reasons);
  const details=reasons ? Object.entries(reasons).filter(([,count])=>typeof count==="number" && Number.isSafeInteger(count) && count>0)
    .map(([reason,count])=>`${count} ${labels[reason]??"unresolved source fields"}`).join("; ") : "";
  return `${counts[0]} included / ${counts[1]} policy-excluded / ${counts[2]} unknown${typeof coverage.unscoped_excluded==="number" && coverage.unscoped_excluded>0 ? ` / ${coverage.unscoped_excluded} exclusions with unknown dates` : ""}${details ? ` | ${details}` : ""}`;
}
export function normalizeRecordedGrossStats(payload:SourcePayload|null,basis:GrossBasis="recorded_gross"):RangeRowMetrics[] {
  if(!recordedGrossContract(payload,basis) || !Array.isArray(payload?.rows))return [];
  return payload.rows.flatMap(value=>{
    const row=object(value), coverage=object(row?.coverage), best=object(row?.topTrade), worst=object(row?.worstLoss);
    if(!row || !coverage || !best || !worst || !["d","w","m","y","all"].includes(String(row.key)))return [];
    const counts=[coverage.included,coverage.excluded,coverage.unknown];
    if(counts.some(count=>typeof count!=="number" || !Number.isSafeInteger(count) || count<0) || row.tradesCount!==counts[0])return [];
    return [{key:row.key as RangeRowMetrics["key"],label:row.label as RangeRowMetrics["label"],
      winRatePct:amount(row.winRatePct),totalPL:amount(row.totalPL),tradesCount:counts[0] as number,
      topTrade:{symbol:String(best.symbol??"--"),pl:amount(best.pl)},
      worstLoss:{symbol:String(worst.symbol??"--"),pl:amount(worst.pl)},
      coverage:{included:counts[0] as number,excluded:counts[1] as number,unknown:counts[2] as number}}];
  });
}
export function normalizeRecordedGrossLeaders(payload:SourcePayload|null,basis:GrossBasis="recorded_gross"):{rows:LeaderRow[];rejected:number} {
  if(!recordedGrossContract(payload,basis) || !Array.isArray(payload?.rows))return {rows:[],rejected:0};
  const rows=payload.rows.flatMap(value=>{
    const row=object(value), pl=amount(row?.pl);
    if(!row || typeof row.symbol!=="string" || !row.symbol.trim() || pl===null || typeof row.rank!=="number" || !Number.isSafeInteger(row.rank) || row.rank<1)return [];
    return [{rank:row.rank,symbol:row.symbol,pl}];
  });
  return {rows,rejected:payload.rows.length-rows.length};
}

// A native unavailable response is an unsuccessful CURRENT read, not proof that
// the last contract-valid historical observation was empty. Retain only this
// exact read-only envelope, with its original time and explicit stale labeling.
const unavailableKeys = new Set(["ok", "status", "source", "view_scope", "pnl_basis", "reason", "rows", "coverage",
  "fees_status", "net_status", "net_eligible", "accounting_complete", "recovery_complete", "entry_ready"]);
export const executionGrossReadPolicy: SourceReadPolicy = {
  accepts: payload => !!payload && payload.ok === true && recordedGrossContract(payload, "execution_gross"),
  retainUnavailable: payload => !!payload && Object.keys(payload).length === unavailableKeys.size &&
    Object.keys(payload).every(key => unavailableKeys.has(key)) && payload.ok === false && payload.status === "unavailable" &&
    payload.reason === "CURRENT_EXECUTION_GROSS_INPUTS_UNAVAILABLE" && payload.source === "postgres" &&
    payload.view_scope === "EXECUTION_RECONSTRUCTED_GROSS" && payload.pnl_basis === "POSTGRES_MATCHED_EXECUTION_GROSS_V1" &&
    Array.isArray(payload.rows) && payload.rows.length === 0 && payload.coverage === null &&
    payload.fees_status === "UNKNOWN" && payload.net_status === "UNKNOWN" && payload.net_eligible === false &&
    payload.accounting_complete === false && payload.recovery_complete === false && payload.entry_ready === false &&
    payload.performance_status === undefined && payload.policy_validation_status === undefined,
};
