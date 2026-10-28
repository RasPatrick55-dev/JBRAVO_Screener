import { finiteNumber, type SourcePayload } from "../ui/sourceState";
import type { LeaderRow, RangeRowMetrics } from "./types";

export const recordedGrossContract=(payload:SourcePayload|null)=> {
  const coverage=object(payload?.coverage);
  const counts=coverage ? [coverage.included,coverage.excluded,coverage.unknown] : [];
  return payload?.source==="postgres" && payload.view_scope==="RECORDED_GROSS" &&
  payload.pnl_basis==="PAPER_GROSS_RECORDED_PRICES_AND_AUDITED_FILLS_V1" && payload.net_eligible===false &&
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
    QTY_UNKNOWN_OR_INVALID:"missing/invalid quantity",ENTRY_PRICE_UNKNOWN_OR_INVALID:"missing/invalid entry price",EXIT_PRICE_UNKNOWN_OR_INVALID:"missing/invalid exit price"};
  const reasons=object(coverage.reasons);
  const details=reasons ? Object.entries(reasons).filter(([,count])=>typeof count==="number" && Number.isSafeInteger(count) && count>0)
    .map(([reason,count])=>`${count} ${labels[reason]??"unresolved source fields"}`).join("; ") : "";
  return `${counts[0]} included / ${counts[1]} policy-excluded / ${counts[2]} unknown${typeof coverage.unscoped_excluded==="number" && coverage.unscoped_excluded>0 ? ` / ${coverage.unscoped_excluded} exclusions with unknown dates` : ""}${details ? ` | ${details}` : ""}`;
}
export function normalizeRecordedGrossStats(payload:SourcePayload|null):RangeRowMetrics[] {
  if(!recordedGrossContract(payload) || !Array.isArray(payload?.rows))return [];
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
export function normalizeRecordedGrossLeaders(payload:SourcePayload|null):{rows:LeaderRow[];rejected:number} {
  if(!recordedGrossContract(payload) || !Array.isArray(payload?.rows))return {rows:[],rejected:0};
  const rows=payload.rows.flatMap(value=>{
    const row=object(value), pl=amount(row?.pl);
    if(!row || typeof row.symbol!=="string" || !row.symbol.trim() || pl===null || typeof row.rank!=="number" || !Number.isSafeInteger(row.rank) || row.rank<1)return [];
    return [{rank:row.rank,symbol:row.symbol,pl}];
  });
  return {rows,rejected:payload.rows.length-rows.length};
}
