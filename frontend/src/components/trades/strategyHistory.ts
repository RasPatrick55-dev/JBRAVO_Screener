import { canDisplay, type SourceKind, type SourceResult, type SourceReadPolicy } from "../ui/sourceState";

export const STRATEGY_HISTORY_LIMIT = 25;
export const STRATEGY_HISTORY_PATH = `/api/trades/strategy-history?limit=${STRATEGY_HISTORY_LIMIT}`;
const SCHEMA = "jbravo.strategy-history.v1";
const BASIS = "STORED_BROKER_ORDER_OBSERVATIONS";
type ObjectValue = Record<string, unknown>;
const object = (value: unknown): ObjectValue | null => value !== null && typeof value === "object" && !Array.isArray(value) ? value as ObjectValue : null;
const text = (value: unknown): string | null => typeof value === "string" && value.trim() ? value.trim() : null;
const numeric = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const nonnegative = (value: unknown): number | null => { const number = numeric(value); return number !== null && number >= 0 ? number : null; };
const timestamp = (value: unknown): string | null => {
  const stamp = text(value);
  if (!stamp || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(stamp) || !Number.isFinite(Date.parse(stamp))) return null;
  const [year,month,day]=stamp.slice(0,10).split("-").map(Number);
  const leap=year%4===0 && (year%100!==0 || year%400===0);
  if(month<1 || month>12 || day<1 || day>[31,leap?29:28,31,30,31,30,31,31,30,31,30,31][month-1]) return null;
  return stamp;
};
const ids = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && !!item.trim()) : [];
const completeIds = (value: unknown): value is string[] => Array.isArray(value) && value.length > 0 &&
  value.every(item => typeof item === "string" && !!item && item === item.trim()) && new Set(value).size === value.length;
export const strategyEvidenceLabel = (value: string | null) => value ? value.replace(/_/g, " ") : "Unknown";

export type StrategyHistoryRow = {
  signalId: string; symbol: string; strategyId: string | null; state: string;
  status: "complete" | "partial" | "incomplete"; reason: string | null;
  completion: "FULL_EXIT_OBSERVED" | "PARTIAL_EXIT_OBSERVED" | "NO_EXIT_OBSERVED" | "NO_ENTRY_FILL_OBSERVED" | "UNVERIFIED";
  entryQty: number | null; exitedQty: number | null; remainingQty: number | null;
  entryPrice: number | null; exitPrice: number | null; gross: number | null;
  entryTime: string | null; exitTime: string | null; updatedAt: string | null;
  entryOrders: string[]; exitOrders: string[];
  ownership: "UNKNOWN" | "STRATEGY_SIGNAL" | "MANUAL_PAPER_TEST" | "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS";
};
export type StrategyHistoryView = {
  rows: StrategyHistoryRow[]; kind: SourceKind; error: string | null; omitted: number;
  sourceAsOf: string | null; observedAt: string | null; hasMore: boolean | null;
  legacyOverlap: number | null;
};

// Python owns fill binding, lifecycle classification and gross arithmetic.
// This adapter only checks the presentation contract; it never joins the legacy ledger.
export function strategyHistoryView(result: SourceResult): StrategyHistoryView {
  const data = result.payload;
  const unavailable = (error: string | null, kind: SourceKind = "unavailable"): StrategyHistoryView => ({ rows: [], kind, error, omitted: 0, sourceAsOf: null, observedAt: null, hasMore: null, legacyOverlap: null });
  if (result.kind === "loading") return unavailable(null, "loading");
  if (!canDisplay(result.kind) || result.freshness === "unknown") return unavailable("Strategy history unavailable. The strategy count and financial results are unknown.");
  if (!data || data.ok !== true || data.schema_version !== SCHEMA || data.source !== "postgres_strategy_execution" ||
      data.trade_view !== "STRATEGY_EXECUTION_HISTORY" || data.evidence_basis !== BASIS ||
      data.performance_status !== "NOT_QUALIFIED_STRATEGY_HISTORY" || data.accounting_complete !== false ||
      data.fees_status !== "UNKNOWN_UNALLOCATED" || data.net_status !== "UNKNOWN" ||
      data.account_scope !== "configured_paper_broker" || data.freshness !== "stored_observations_not_live" ||
      !["ready", "partial", "empty"].includes(String(data.status)) || !timestamp(data.observed_at) || !Array.isArray(data.rows) || data.rows.length>50) {
    return unavailable("Strategy history unavailable: the source contract could not be verified. This is not an empty trading history.");
  }
  const rows: StrategyHistoryRow[] = [];
  let omitted = 0;
  for (const value of data.rows) {
    const row = object(value);
    const signalId = typeof row?.signal_id === "string" ? row.signal_id : null;
    const symbol = typeof row?.symbol === "string" ? row.symbol : null;
    if (!row || !signalId || !/^[a-f0-9]{64}$/.test(signalId) || !symbol || !/^[A-Z][A-Z0-9.\-]{0,15}$/.test(symbol) || rows.some(item => item.signalId === signalId)) { omitted++; continue; }
    const state = text(row.state) ?? "Unknown";
    const bound = data.broker_account_match === "verified" && row.evidence_basis === BASIS &&
      row.fees_status === "UNKNOWN_UNALLOCATED" && row.fees === null && row.net_realized_pnl === null;
    const supported = ["FULL_EXIT_OBSERVED", "PARTIAL_EXIT_OBSERVED", "NO_EXIT_OBSERVED", "NO_ENTRY_FILL_OBSERVED"].includes(String(row.completion_status));
    const known = bound && supported && ["complete", "partial"].includes(String(row.status));
    const entryTime = timestamp(row.entry_time), exitTime = timestamp(row.exit_time);
    const entryOrders = row.entry_order_ids, exitOrders = row.exit_order_ids;
    const completeOrderIds = completeIds(entryOrders) && completeIds(exitOrders) &&
      entryOrders.every(id => !exitOrders.includes(id));
    const completeFields = [row.entry_qty,row.exited_qty,row.entry_avg_price,row.exit_avg_price].every(value=>typeof value==="number" && Number.isFinite(value) && value>0) &&
      row.entry_qty === row.exited_qty && row.remaining_qty === 0 && entryTime !== null && exitTime !== null &&
      Date.parse(entryTime) <= Date.parse(exitTime) && completeOrderIds;
    const complete = known && row.status === "complete" && row.completion_status === "FULL_EXIT_OBSERVED" && state === "CLOSED" && completeFields;
    const inconsistentComplete = row.status === "complete" && !complete;
    const available = known && !inconsistentComplete;
    rows.push({
      signalId, symbol, strategyId: text(row.strategy_id), state,
      status: complete ? "complete" : available ? "partial" : "incomplete",
      reason: !bound || inconsistentComplete ? "source_evidence_unverified" : text(row.reason),
      completion: available ? row.completion_status as StrategyHistoryRow["completion"] : "UNVERIFIED",
      entryQty: available ? nonnegative(row.entry_qty) : null,
      exitedQty: available ? nonnegative(row.exited_qty) : null,
      remainingQty: available ? nonnegative(row.remaining_qty) : null,
      entryPrice: available ? nonnegative(row.entry_avg_price) : null,
      exitPrice: available ? nonnegative(row.exit_avg_price) : null,
      gross: complete ? numeric(row.gross_realized_pnl) : null,
      entryTime: available ? entryTime : null, exitTime: available ? exitTime : null,
      updatedAt: timestamp(row.updated_at), entryOrders: ids(row.entry_order_ids), exitOrders: ids(row.exit_order_ids),
      ownership: row.ownership_basis === "STORED_SIGNAL_PAYLOAD_NOT_REVALIDATED" && ["STRATEGY_SIGNAL", "MANUAL_PAPER_TEST", "ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS"].includes(String(row.ownership_scope)) ? row.ownership_scope as StrategyHistoryRow["ownership"] : "UNKNOWN",
    });
  }
  // An open/unfilled signal is a valid lifecycle observation, not a failed read.
  const partial = omitted > 0 || rows.some(row => row.status === "incomplete" || row.reason === "native_state_reconciliation_required") || data.status === "partial" || data.broker_account_match !== "verified";
  return { rows, kind: result.freshness === "stale" ? "stale" : partial ? "partial" : rows.length ? "ready" : "empty",
    error: omitted ? `${omitted} source ${omitted === 1 ? "record was" : "records were"} not displayed because its identity was invalid or duplicated. Coverage is incomplete.` : null,
    omitted, sourceAsOf: timestamp(data.source_as_of), observedAt: timestamp(data.observed_at), hasMore: typeof data.has_more === "boolean" ? data.has_more : null,
    legacyOverlap: typeof data.legacy_overlap_count === "number" && Number.isSafeInteger(data.legacy_overlap_count) && data.legacy_overlap_count >= 0 ? data.legacy_overlap_count : null };
}

export const strategyNumber = (value: number | null) => value === null ? "Unknown" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 8 }).format(value);
export const strategyCurrency = (value: number | null) => value === null ? "Unknown" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 8 }).format(value);
export const strategyTime = (value: string | null) => timestamp(value) === null ? "Unknown" : new Date(value!).toISOString().replace("T", " ").replace(/\.\d{3}Z$/, " UTC");
export const strategyCompletion = (value: StrategyHistoryRow["completion"]) => ({FULL_EXIT_OBSERVED: "Full exit observed", PARTIAL_EXIT_OBSERVED: "Partial exit observed", NO_EXIT_OBSERVED: "No exit observed", NO_ENTRY_FILL_OBSERVED: "Entry unfilled in stored observation", UNVERIFIED: "Fill evidence incomplete"})[value];
export const strategyOwnership = (value: StrategyHistoryRow["ownership"]) => ({UNKNOWN: "Ownership unknown", STRATEGY_SIGNAL: "Stored: strategy enrollment", MANUAL_PAPER_TEST: "Stored: manual paper test", ENTRY_RECOGNITION_ONLY_NO_POSITION_ORDERS: "Stored: entry recognition only (no position orders)"})[value];

// Do not advance a successful-data timestamp for an HTTP200 unavailable or
// malformed envelope. Strategy failures still hide rows; no new retention here.
export const strategyHistoryReadPolicy: SourceReadPolicy = {
  accepts: payload => canDisplay(strategyHistoryView({payload, kind: "ready", validity: "ready", freshness: "current"}).kind),
  retainUnavailable: () => false,
};
