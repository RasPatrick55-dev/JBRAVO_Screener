import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { metricRows, strategyLabel } from "../src/components/screener/metricsRows";
import type { MetricsResponse } from "../src/components/screener/types";
import fixture from "./unifiedMetricsFixture.json";

const payload=()=>structuredClone(fixture.populated) as MetricsResponse;
const single=():MetricsResponse=>{
  const data=payload(),row=data.rows![0];
  return {...data,...row,metrics_result_id:row.metrics_result_id,metrics_status:"COMPLETE",rows:[row]};
};
afterEach(()=>{vi.unstubAllGlobals();vi.restoreAllMocks();vi.doUnmock("react");vi.resetModules();});

describe("retained unified Metrics wire contract",()=>{
  it("accepts row-bound results without fabricating top-level result IDs",()=>{
    const data=payload(),rows=metricRows(data);
    expect(data.backtest_result_id).toBeUndefined();expect(data.metrics_result_id).toBeUndefined();
    expect(rows).toHaveLength(2);
    for(let i=0;i<rows.length;i++) {
      for(const key of ["strategy_id","strategy_version","definition_sha256","signal_id","backtest_result_id","metrics_result_id"] as const)
        expect(rows[i][key]).toBe(data.rows![i][key]);
    }
    expect(rows[0].strategy_id).not.toBe(rows[1].strategy_id);
    expect(strategyLabel(data.strategy_id!,data.strategy_version!)).toBe("All strategies (per-setup results)");
  });
  it("retains original single-strategy matching and legitimate empty contracts",()=>{
    expect(metricRows(single())).toHaveLength(1);
    expect(metricRows({...single(),rows:[]})).toEqual([]);
    expect(metricRows(fixture.empty)).toEqual([]);
    const legacy=single();delete legacy.rows![0].metrics_result_id;
    expect(metricRows(legacy)).toHaveLength(1);
  });
  it.each(["strategy_id","strategy_version","definition_sha256","signal_id","backtest_result_id","metrics_result_id"] as const)("rejects absent/malformed row %s",key=>{
    for(const value of [undefined,null,"",false,123," "]) {
      const data=payload();(data.rows![0] as unknown as Record<string,unknown>)[key]=value;
      expect(()=>metricRows(data)).toThrow();
    }
    if(key.endsWith("id") && key!=="strategy_id" || key==="definition_sha256") {
      const data=payload();(data.rows![0] as unknown as Record<string,unknown>)[key]="not-a-sha256";
      expect(()=>metricRows(data)).toThrow();
    }
  });
  it("rejects contradictory row bindings within one strategy instead of pooling them",()=>{
    const data=payload();data.rows![1]={...data.rows![0],symbol:"OTHER",signal_id:"f".repeat(64),metrics_result_id:"a".repeat(64)};
    expect(()=>metricRows(data)).toThrow();
  });
  it.each(["metrics_result_id","backtest_result_id"] as const)("rejects shared %s across conflicting strategy contracts",key=>{
    const data=payload();data.rows![1][key]=data.rows![0][key];
    expect(()=>metricRows(data)).toThrow("Metrics do not match");
  });
  it("preserves multiple distinct signals from one report and separate ID namespaces",()=>{
    const data=payload(),first=data.rows![0];
    data.rows![1]={...first,symbol:"OTHER",signal_id:"f".repeat(64)};
    expect(metricRows(data)).toHaveLength(2);
    // A Metrics ID and an unrelated backtest ID are different namespaces.
    expect(payload().rows![0].metrics_result_id).toBe(payload().rows![1].backtest_result_id);
    expect(metricRows(payload())).toHaveLength(2);
  });
  it.each(["same-row","different-symbol"])("rejects a repeated current signal identity (%s)",shape=>{
    const data=payload();data.rows![1]={...data.rows![0],...(shape==="different-symbol"?{symbol:"OTHER"}:{})};
    expect(()=>metricRows(data)).toThrow("Metrics do not match");
  });
  it("rejects invalid envelopes, absent row collections, and aggregate identity as row strategy",()=>{
    for(const update of [{ok:false},{strategy_version:"2"},{definition_sha256:"bad"},{metric_basis:"LEDGER_NET"},
      {metrics_status:"METRICS_PENDING"},{strategy_priority:true},{backtest_result_id:"b".repeat(64)},
      {metrics_result_id:"c".repeat(64)},{rows:undefined},{rows:null}])
      expect(()=>metricRows({...payload(),...update} as MetricsResponse)).toThrow();
    const data=payload();data.rows![0].strategy_id="ALL_STRATEGIES";expect(()=>metricRows(data)).toThrow();
  });
  it("retains single-strategy envelope and optional row Metrics ID validation",()=>{
    for(const key of ["backtest_result_id","metrics_result_id","definition_sha256"] as const) {
      const data=single();data[key]="bad";expect(()=>metricRows(data)).toThrow();
    }
    const data=single();data.rows![0].metrics_result_id="a".repeat(64);expect(()=>metricRows(data)).toThrow();
  });
});

async function renderLoaded(data:MetricsResponse) {
  vi.resetModules();let cursor=0;const states:unknown[]=[],effects:Array<()=>void|(()=>void)>=[];
  vi.stubGlobal("window",globalThis);
  vi.stubGlobal("fetch",vi.fn().mockResolvedValue(new Response(JSON.stringify(data),{status:200})));
  vi.doMock("react",async()=>{
    const actual=await vi.importActual<typeof import("react")>("react");
    return {...actual,useState:(initial:unknown)=>{const index=cursor++;if(!(index in states))states[index]=initial;
      return [states[index],(next:unknown)=>{states[index]=next;}];},useEffect:(effect:()=>void|(()=>void))=>{effects.push(effect);},
      useMemo:(calculate:()=>unknown)=>calculate(),useRef:(value:unknown)=>({current:value})};
  });
  const {default:Card}=await import("../src/components/screener/MetricsResultsCard");
  renderToStaticMarkup(<Card/>);
  const stop=effects[1]();
  try {
    for(let i=0;i<40;i++)await Promise.resolve();
    cursor=0;
    return renderToStaticMarkup(<Card/>);
  } finally {if(typeof stop==="function")stop();}
}

describe("actual Metrics request-adapter-state-render flow",()=>{
  it.each(["metrics_result_id","backtest_result_id"] as const)("renders contradictory %s as error instead of populated/empty success",async key=>{
    const data=payload();data.rows![1][key]=data.rows![0][key];
    const html=await renderLoaded(data);
    expect(html).toContain('role="alert"');expect(html).toContain("Metrics do not match");
    expect(html).not.toContain("No Screener picks in this batch.");expect(html).not.toContain("$12.34");
  });
  it("renders multiple signals sharing a valid report",async()=>{
    const data=payload();data.rows![1]={...data.rows![0],symbol:"OTHER",signal_id:"f".repeat(64)};
    const html=await renderLoaded(data);
    expect(html).toContain("SYN");expect(html).toContain("OTHER");expect(html).not.toContain('role="alert"');
  });
  it("renders populated unified results with each row's strategy and no summed metric",async()=>{
    const html=await renderLoaded(payload());
    expect(html).toContain("All strategies (per-setup results)");
    expect(html).toContain("SYN");expect(html).toContain("FIX");
    expect(html).toContain("Pullback/Reclaim v2");expect(html).toContain("bull flag companion v1 v1");
    expect(html).toContain("$12.34");expect(html).toContain("$6.17");
    expect(html).not.toContain("$18.51");expect(html).not.toContain("No Screener picks");
    expect(html).not.toContain('role="alert"');expect(html).toContain("jbravo-panel-emerald");
  });
  it("renders valid empty unified and original single-strategy results",async()=>{
    expect(await renderLoaded(fixture.empty)).toContain("No Screener picks in this batch.");
    vi.doUnmock("react");vi.resetModules();
    const html=await renderLoaded(single());expect(html).toContain("SYN");expect(html).not.toContain('role="alert"');
  });
  it.each(["missing","malformed"])("renders %s row ID as unavailable, never successful empty",async damage=>{
    const data=payload();data.rows![0].metrics_result_id=damage==="missing"?undefined:"invalid";
    const html=await renderLoaded(data);
    expect(html).toContain('role="alert"');expect(html).toContain("Metrics do not match");
    expect(html).not.toContain("No Screener picks in this batch.");expect(html).not.toContain("$12.34");
  });
});
