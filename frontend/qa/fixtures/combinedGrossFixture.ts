// Synthetic fixtures only. The base JSON is produced by the Python projection.
import wire from "./combinedGrossWire.synthetic.json";
export const combinedGrossFixture=():Record<string,any>=>structuredClone(wire);
export function combinedGrossEmptyFixture(){
  const p=combinedGrossFixture();
  const coverage={included:0,excluded:0,unknown:0,deduplicated:0,strategy_not_closed:0,by_source:{legacy:{included:0,excluded:0,unknown:0},strategy:{included:0,excluded:0,unknown:0}}};
  p.status="empty";p.rows=[];p.coverage=coverage;p.source_as_of=null;
  p.stats=p.stats.map((row:Record<string,unknown>)=>({...row,tradesCount:0,totalPL:"0",winRatePct:null,topTrade:{symbol:"--",pl:null},worstLoss:{symbol:"--",pl:null},coverage:structuredClone(coverage)}));
  p.rankings=Object.fromEntries(Object.keys(p.rankings).map(key=>[key,{winners:[],losers:[]}]));
  p.strategy_history={...p.strategy_history,rows:[],status:"empty",source_as_of:null,has_more:false};return p;
}
export function combinedGrossPaginationFixture(){
  // QA-only pagination extension: no new eligible contributions or calculations.
  const p=combinedGrossFixture();
  const excluded=p.rows.find((row:Record<string,unknown>)=>row.source_kind === "LEGACY_EXECUTION" && row.inclusion === "POLICY_EXCLUDED");
  for(let i=0;i<25;i++)p.rows.push({...excluded,trade_id:10000+i,row_id:`legacy:${10000+i}`,symbol:`PAGE${i}`});
  p.cohort_id="b".repeat(64);p.coverage.excluded+=25;p.coverage.by_source.legacy.excluded+=25;
  p.stats=p.stats.map((row:Record<string,any>)=>({...row,coverage:{...row.coverage,excluded:row.coverage.excluded+25,
    by_source:{...row.coverage.by_source,legacy:{...row.coverage.by_source.legacy,excluded:row.coverage.by_source.legacy.excluded+25}}}}));
  return p;
}
