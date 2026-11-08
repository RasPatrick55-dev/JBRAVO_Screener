import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { buyFillSummary, type ExecuteOrdersSummaryResponse } from "../src/components/dashboard/buyFillSummary";
import ExecuteTradesStatusCard from "../src/components/dashboard/ExecuteTradesStatusCard";
import PythonAnywhereResources, { resourceRows } from "../src/components/dashboard/PythonAnywhereResources";
import PythonAnywhereTasksTable, { scheduledTaskRows } from "../src/components/dashboard/PythonAnywhereTasksTable";
import { normalizeOpenOrders } from "../src/components/account/AccountTab";
import OpenOrdersCard from "../src/components/account/OpenOrdersCard";
import OrdersTableCard from "../src/components/execute/OrdersTableCard";
import TrailingStopsCard from "../src/components/execute/TrailingStopsCard";
import ExecutionSummaryCard from "../src/components/execute/ExecutionSummaryCard";
import { filterOrdersClient, filterTrailingClient, normalizeOrderStatus, normalizeTrailingStatus } from "../src/components/execute/utils";
import MonitoringLogsPanel from "../src/components/positions/MonitoringLogsPanel";
import { toMonitoringLog } from "../src/screens/PositionsMonitoring";
import { BacktestCompletionBadge } from "../src/components/screener/BacktestResultsCard";
import { computationComplete, computationStatus, ComputationScope } from "../src/components/screener/completion";
import { predictionFreshness, mlStagePresentation, successfulPredictionExecution } from "../src/components/ml/freshness";
import type { MlStage } from "../src/components/ml/useMlOverview";
import { StageTile } from "../src/screens/MLPipeline";

const fillSummary: ExecuteOrdersSummaryResponse = {
  ok: true, status: "empty", source: "alpaca", fill_count: 0, total_value: 0,
  source_endpoint: "/v2/account/activities/FILL", account_scope: "configured_paper_broker",
  count_unit: "buy_fill_activity_events", value_unit: "USD_buy_fill_notional", coverage: "requested_24h_fill_activity_window",
  coverage_scope: "observed_broker_fill_activities", window_basis: "transaction_time",
  window_boundary: "since_inclusive_until_exclusive", accounting_final: false,
  since_utc: "2026-02-03T12:00:00Z", until_utc: "2026-02-04T12:00:00Z",
};

describe("trailing 24-hour buy fill reporting", () => {
  it("keeps a genuine zero count and zero value together, regardless of old submissions", () => {
    expect(buyFillSummary(fillSummary)).toEqual({ count: 0, value: 0 });
    const source = readFileSync(new URL("../src/screens/DashboardHealth.tsx", import.meta.url), "utf8");
    expect(source).toContain("const ordersPlaced = fills24h.count");
    expect(source).toContain("totalValue: fills24h.value");
    expect(source).not.toContain("const ordersPlaced = typeof ordersSubmitted");
    const html = renderToStaticMarkup(<ExecuteTradesStatusCard lastRun={{date:"",start:"",end:"",duration:""}}
      ordersPlaced={0} totalValue={0} successRate={NaN} isCycleComplete={false} />);
    expect(html).toContain("Buy fill executions (24h)");
    expect(html).toContain("Buy notional (USD)");
    expect(html).toContain("Observed broker executions; late-reported fills may appear on refresh.");
    expect(html).toContain("$0.00");
  });
  it.each([
    null, { ...fillSummary, ok: false }, { ...fillSummary, source: "logs" },
    { ...fillSummary, coverage: "at_least_500_orders_incomplete" },
    { ...fillSummary, coverage: "requested_24h_window" },
    { ...fillSummary, coverage: "incomplete", observed_fill_count: 13 },
    { ...fillSummary, status: "partial" }, { ...fillSummary, status: "unavailable" },
    { ...fillSummary, status: null }, { ...fillSummary, status: "ready" },
    { ...fillSummary, status: "empty", fill_count: 1 },
    { ...fillSummary, source_endpoint: "/v2/orders" },
    { ...fillSummary, window_basis: "submitted_at" },
    { ...fillSummary, window_boundary: null }, { ...fillSummary, coverage_scope: null },
    { ...fillSummary, account_scope: "unknown" },
    { ...fillSummary, count_unit: "completed_buy_orders" }, { ...fillSummary, count_unit: null },
    { ...fillSummary, value_unit: "USD_all_orders" },
    { ...fillSummary, since_utc: null }, { ...fillSummary, until_utc: "2026-02-05T12:00:00Z" },
  ])("holds count and value when broker window or scope is not verified: %j", summary => {
    const result = buyFillSummary(summary);
    expect(result.count).toBeNaN(); expect(result.value).toBeNaN();
  });
  it("holds both totals when a fully validated count or notional is unavailable", () => {
    expect(buyFillSummary({ ...fillSummary, fill_count: 1, total_value: null })).toEqual({ count: NaN, value: NaN });
    expect(buyFillSummary({ ...fillSummary, fill_count: null, total_value: 107 })).toEqual({ count: NaN, value: NaN });
  });
  it("shows fill activity executions, including partials, rather than a completed-order count", () => {
    const executions = { ...fillSummary, status: "ready", fill_count: 3, orders_filled: 1, total_value: 107 };
    const result = buyFillSummary(executions);
    expect(result).toEqual({ count: 3, value: 107 });
    const html = renderToStaticMarkup(<ExecuteTradesStatusCard lastRun={{date:"",start:"",end:"",duration:""}}
      ordersPlaced={result.count} totalValue={result.value} successRate={NaN} isCycleComplete={false} />);
    expect(html).toContain("Buy fill executions (24h)"); expect(html).toContain(">3<"); expect(html).toContain("$107.00");
  });
  it("never uses the old order count or an observed incomplete lower bound as the headline", () => {
    const legacy = { ...fillSummary, fill_count: undefined, orders_filled: 7 };
    expect(buyFillSummary(legacy)).toEqual({ count: NaN, value: NaN });
    expect(buyFillSummary({ ...fillSummary, ok:false, status:"partial", coverage:"incomplete", fill_count:null, observed_fill_count:13, total_value:null }))
      .toEqual({ count: NaN, value: NaN });
  });
});

describe("broker status and replacement lineage", () => {
  it.each(["canceled", "rejected", "expired", "done_for_day", "pending_replace", "pending_cancel", "replaced", "new"])("preserves %s without inventing fills or triggers", raw => {
    expect(normalizeOrderStatus(raw)).toBe(raw.toUpperCase());
    expect(normalizeTrailingStatus(raw)).toBe(raw.toUpperCase());
  });
  it("unknown and closed are not fill evidence", () => {
    expect(normalizeOrderStatus(null)).toBe("UNKNOWN");
    expect(normalizeTrailingStatus(undefined)).toBe("UNKNOWN");
    expect(normalizeOrderStatus("closed")).toBe("CLOSED");
    expect(normalizeOrderStatus("partially_filled")).toBe("PARTIAL");
    expect(normalizeTrailingStatus("filled")).toBe("FILLED");
  });
  it("filters preserved lifecycle states without dropping replacements", () => {
    const rows = [{ raw_status: "pending_replace", status: "PENDING", order_id: "order-old", replaced_by: "order-next" }, { raw_status: "canceled" }];
    expect(filterOrdersClient(rows, "order-next", "open", "all")).toHaveLength(1);
    expect(filterOrdersClient(rows, "", "closed", "all")).toHaveLength(1);
    expect(filterTrailingClient(rows, "", "open", "all")).toHaveLength(1);
    expect(filterTrailingClient(rows, "", "closed", "all")).toHaveLength(1);
  });
  it("shows synthetic linked orders in Account and Execute without calling them duplicates", () => {
    const raw = { symbol: "TEST", side: "sell", type: "stop", qty: 2, price_or_stop: 12, submitted_at: "2026-02-04T12:00:00Z", raw_status: "pending_replace", order_id: "order-old", replaced_by: "order-next" };
    const rows = normalizeOpenOrders({ ok: true, rows: [raw] });
    expect(rows[0].replaced_by).toBe("order-next");
    for (const html of [
      renderToStaticMarkup(<OpenOrdersCard rows={rows} sourceKind="ready" />),
      renderToStaticMarkup(<OrdersTableCard rows={[raw]} isLoading={false} hasError={false} />),
      renderToStaticMarkup(<TrailingStopsCard rows={[raw]} isLoading={false} hasError={false} />),
    ]) {
      expect(html).toContain("pending_replace"); expect(html).toContain("Replacement chain");
      expect(html).toContain("order-old"); expect(html).toContain("order-next");
      expect(html.toLowerCase()).not.toContain("duplicate"); expect(html).not.toContain("TRIGGERED");
    }
  });
});

describe("resource snapshot provenance", () => {
  it("uses unknown, not 10/35/10 percent defaults", () => {
    const rows = resourceRows(null);
    expect(rows.map(row => row.value)).toEqual([null, null, null]);
    const html = renderToStaticMarkup(<PythonAnywhereResources resources={rows} />);
    expect(html).toContain("Unknown"); expect(html).not.toMatch(/(?:10|35)%/);
    expect(html).toContain("Captured: unknown");
  });
  it("retains over-quota text while capping only the bar and labels a stale snapshot", () => {
    const html = renderToStaticMarkup(<PythonAnywhereResources resources={resourceRows([
      { label: "File Storage", value: 143.2, used: 1432, quota: 1000, unit: "bytes", source: "saved_snapshot", captured_at: "2026-02-01T00:00:00Z", status: "stale", stale: true },
    ])} />);
    expect(html).toContain("143.2%"); expect(html).toContain("width:100%");
    expect(html).toContain("STALE"); expect(html).toContain("Over quota in this snapshot");
    expect(html).toContain("1,432 / 1,000 bytes"); expect(html).toContain("2026-02-01T00:00:00Z");
  });
  it("does not call a timestamp-less historic value fresh", () => {
    const html = renderToStaticMarkup(<PythonAnywhereResources resources={[{ label: "CPU Usage", value: 5, status: "fresh" }]} />);
    expect(html).toContain("UNKNOWN"); expect(html).not.toContain(">FRESH<");
  });
});

describe("scheduled task provenance", () => {
  it.each([null, {ok:false,tasks:[{name:"Prior run",frequency:"Daily",time:"03:45 UTC"}]}, {ok:true,tasks:[]}, {ok:true,tasks:[null,{command:"unmapped"}]}])("does not invent schedules when unavailable or unrecognized: %j", payload => {
    const rows = scheduledTaskRows(payload);
    expect(rows).toEqual([]);
    const html = renderToStaticMarkup(<PythonAnywhereTasksTable rows={rows}/>);
    expect(html).toContain("Current schedule is unknown");
    expect(html).not.toContain("No tasks scheduled"); expect(html).not.toContain("Run Pipeline");
    expect(html).not.toContain("03:45"); expect(html).not.toContain("04:00");
  });
  it("preserves real rows and leaves omitted time or frequency unknown", () => {
    const rows = scheduledTaskRows({ok:true,tasks:[{name:"Verified task",frequency:"Daily",time:"08:15 UTC"},{name:"Task with partial metadata"}]});
    expect(rows).toEqual([{name:"Verified task",frequency:"Daily",time:"08:15 UTC"},{name:"Task with partial metadata",frequency:"Unknown",time:"Unknown"}]);
    const html = renderToStaticMarkup(<PythonAnywhereTasksTable rows={rows}/>);
    expect(html).toContain("Verified task"); expect(html).toContain("08:15 UTC");
    expect(html).not.toContain("Current schedule is unknown");
    const source = readFileSync(new URL("../src/screens/DashboardHealth.tsx", import.meta.url), "utf8");
    expect(source).not.toContain("pythonAnywhereTasksDefaults"); expect(source).toContain("scheduledTaskRows(pythonAnywhereTasksPayload)");
  });
});

describe("execution metadata and independent attempts", () => {
  it("does not turn absent or legacy in-window metadata into NO", () => {
    const html = renderToStaticMarkup(<ExecutionSummaryCard summary={{ok:true,in_window:false}} isLoading={false} hasError={false} />);
    expect(html).toContain("UNKNOWN"); expect(html).toContain("Entry window unknown");
    expect(html).not.toContain("07:00-09:30"); expect(html).not.toContain(">NO<");
  });
  it("renders verified release metadata and guarded historical attempt separately", () => {
    const html = renderToStaticMarkup(<ExecutionSummaryCard summary={{ok:true,last_run_utc:"2026-02-03T09:00:00Z",execution_schedule:{
      status:"verified",in_window:false,entry_start_local:"04:00",entry_end_local:"04:01",timezone:"America/New_York",
      lifecycle_owner:"MONITOR_POSITIONS",cancellation_policy:"monitor_exchange_regular_close",legacy_cancel_after_min:35,legacy_cancel_after_min_is_deadline:false,
    }}} taskSource={{kind:"ready",freshness:"current",receivedAt:"2026-02-04T12:00:00Z"}} lastExecutionReceipt={{ok:true,source:"executor_receipt",status:"STALE",finished_utc:"2026-02-03T09:00:00Z"}} lastScheduledAttempt={{ok:true,status:"GUARDED_NO_ENTRY",started_utc:"2026-02-04T07:55:00Z",rc:1,orders_submitted:0,release_attribution:"unverified_historical_attempt"}}
      isLoading={false} hasError={false} />);
    for (const text of ["04:00-04:01 America/New_York", "MONITOR_POSITIONS", "Deadline: unknown", "not the cancellation deadline", "Last execution receipt", "GUARDED_NO_ENTRY", "not verified against the current release"]) expect(html).toContain(text);
    expect(html).toContain(">NO<");
  });
});

describe("monitor row provenance", () => {
  it("keeps report observation distinct from broker events and current protection", () => {
    const row = toMonitoringLog({ timestamp: "2026-02-04", message:"replacement remains pending", source:"monitor/latest.json",source_kind:"monitor_receipt",run_id:"monitor-test",ts_utc:"2026-02-04T12:00:00Z",timestamp_kind:"monitor_report_observation",stale:true,protection_confirmed:false,old_order_id:"order-old",replacement_order_id:"order-next",native_stops_trigger_only_regular_hours:true,extended_hours_native_protection:false });
    const html = renderToStaticMarkup(<MonitoringLogsPanel logs={[row]} />);
    for (const text of ["Monitor report observation", "Observed/report UTC", "STALE", "unconfirmed", "not a current protection guarantee", "order-old", "order-next", "Extended-hours native protection: not provided"]) expect(html).toContain(text);
    expect(html).not.toContain("Always On Track"); expect(html).not.toContain("Event UTC");
  });
  it("marks absent provenance unknown", () => {
    const html = renderToStaticMarkup(<MonitoringLogsPanel logs={[toMonitoringLog({message:"unattributed event"})]} />);
    expect(html).toContain("Unverified provenance"); expect(html).toContain("Source: unknown");
  });
});

describe("computation is not permission to trade", () => {
  const payload = { ok:true,status:"COMPLETE",completion_status:"verified",computation_complete:true,research_qualified:false,order_submission_authorized:false,entry_readiness_scope:"PREPARED_PAPER_HANDOFF_ONLY",broker_preflight_required:true };
  it.each(["COMPLETE", "EMPTY"])("uses the shared verified contract for %s", status => {
    expect(computationComplete({...payload,status})).toBe(true);
    expect(computationStatus({...payload,status})).toBe("COMPUTE COMPLETE");
    const html = renderToStaticMarkup(<><BacktestCompletionBadge payload={{...payload,status}} kind="ready" /><ComputationScope payload={payload}/></>);
    for (const text of ["COMPUTE COMPLETE", "Research qualification: not qualified", "Order submission: not authorized", "Broker preflight required"]) expect(html).toContain(text);
  });
  it.each([{computation_complete:undefined}, {completion_status:undefined}, {ok:false}, {computation_complete:false}])("fails closed on incomplete evidence %j", patch => {
    expect(computationComplete({...payload,...patch})).toBe(false);
  });
  it("does not render the broken scope separator", () => {
    const source = readFileSync(new URL("../src/components/screener/ScreenerPicksCard.tsx", import.meta.url), "utf8");
    expect(source).not.toContain("\uFFFD"); expect(source).not.toContain("{statusLabel} ? {scopeLabel}");
  });
});

describe("ML freshness and artifacts", () => {
  it.each(["fresh", "stale", "unknown"])("retains a failed prediction with rc=1 when freshness is %s", freshness_status => {
    const stage: MlStage = { label:"Predict", status:"blocked", tone:"error", detail:"ranker_predict failed (rc=1)", artifact_present:true, execution_status:"unverified", freshness_status };
    expect(mlStagePresentation(stage)).toBe(stage);
    const html = renderToStaticMarkup(<StageTile stage={stage} />);
    expect(html).toContain(">blocked<"); expect(html).toContain("ranker_predict failed (rc=1)");
    expect(html).toContain("Execution: unverified"); expect(html).toContain(`Freshness: ${freshness_status}`);
    expect(html).not.toContain("Present / unverified"); expect(html).not.toContain(">Stale<");
  });
  it.each([
    {label:"Monitor", status:"warn", tone:"warning", detail:"guard requires recalibration"},
    {label:"Monitor", status:"blocked", tone:"error", detail:"retrain"},
    {label:"Features", status:"blocked", tone:"neutral", detail:"No features artifact available"},
    {label:"Auto-remediate", status:"skipped", tone:"neutral", detail:"No remediation executed"},
  ] as MlStage[])("preserves authoritative $label $status and action", input => {
    const stage: MlStage = {...input, artifact_present:true, execution_status:"unverified", freshness_status:"stale"};
    expect(mlStagePresentation(stage)).toBe(stage);
    const html = renderToStaticMarkup(<StageTile stage={stage} />);
    expect(html).toContain(`>${input.status}<`); expect(html).toContain(input.detail!);
    expect(html).toContain("Execution: unverified"); expect(html).toContain("Freshness: stale");
  });
  it("only downgrades positive claims and preserves non-success source status", () => {
    const positive: MlStage = {label:"Eval",status:"available",tone:"neutral",artifact_present:true,execution_status:"unverified",freshness_status:"fresh"};
    expect(mlStagePresentation(positive)).toMatchObject({status:"Present / unverified",tone:"neutral"});
    const verified: MlStage = {...positive,status:"success",tone:"success",execution_status:"verified"};
    expect(mlStagePresentation(verified)).toBe(verified);
  });
  it("does not relabel a recent log update as a successful ML run", () => {
    const evidence = { ok:true, updated_at:"2026-02-05T12:00:00Z", status:{last_ml_run:"2026-02-05T12:00:00Z"} };
    expect(successfulPredictionExecution(evidence)).toBeNull();
    expect(successfulPredictionExecution({...evidence,last_successful_execution:"2026-02-04T10:00:00Z"})).toBeNull();
    expect(successfulPredictionExecution({...evidence,last_successful_execution:"2026-02-04T10:00:00Z",execution_evidence_scope:"PREDICTION_STAGE_ONLY"})).toBe("2026-02-04T10:00:00Z");
    expect(successfulPredictionExecution({...evidence,last_successful_execution:"invalid",execution_evidence_scope:"PREDICTION_STAGE_ONLY"})).toBeNull();
    expect(successfulPredictionExecution({...evidence,ok:false,last_successful_execution:"2026-02-04T10:00:00Z",execution_evidence_scope:"PREDICTION_STAGE_ONLY"})).toBeNull();
    for (const file of ["../src/screens/MLPipeline.tsx", "../src/components/dashboard/MLDailyImprovements.tsx"]) {
      const source = readFileSync(new URL(file, import.meta.url), "utf8");
      expect(source).toContain("Last ML evidence"); expect(source).not.toContain("Last ML run");
    }
  });
  it("stale backend state wins over an old Fresh label", () => {
    expect(predictionFreshness({label:"Fresh",status:"stale",stale:true})).toBe("Stale");
    expect(predictionFreshness({label:"Fresh",pred_compatible:true})).toBe("Unknown");
    expect(predictionFreshness({status:"fresh",stale:false})).toBe("Fresh");
  });
  it("does not equate artifact presence with execution success", () => {
    const stage = { label:"Predict",status:"Success",tone:"success" as const,artifact_present:true,execution_status:"unverified",freshness_status:"stale" };
    expect(mlStagePresentation(stage).status).toBe("Stale");
    const html = renderToStaticMarkup(<StageTile stage={stage} />);
    expect(html).toContain("Artifact: present"); expect(html).toContain("Execution: unverified"); expect(html).toContain("Freshness: stale");
    expect(html).not.toContain(">Success<");
  });
});
