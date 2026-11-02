import type { SourceResult } from "../ui/sourceState";

type ObjectValue = Record<string, unknown>;
const object = (value: unknown): ObjectValue | null => value !== null && typeof value === "object" && !Array.isArray(value) ? value as ObjectValue : null;
const revision = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value > 0;

// Display server-validated Decimal text without binary floating-point arithmetic.
// The fill validation and gross calculation remain canonical Python operations.
function decimal(value: unknown): string | null {
  if (typeof value !== "string" || value.length > 96 || !/^-?\d+(?:\.\d+)?$/.test(value)) return null;
  const negative = value.startsWith("-");
  const [whole, fraction = ""] = (negative ? value.slice(1) : value).split(".");
  const integer = whole.replace(/^0+(?=\d)/, "");
  const digits = fraction.replace(/0+$/, "");
  const absolute = integer + (digits ? `.${digits}` : "");
  return negative && absolute !== "0" ? `-${absolute}` : absolute;
}
export const formatAllocationGrossCurrency = (value: string) => {
  const negative = value.startsWith("-");
  const [whole, fraction = ""] = (negative ? value.slice(1) : value).split(".");
  return `${negative ? "-" : ""}$${whole}.${fraction.padEnd(2, "0")}`;
};
export type GrossAllocation = { tradeId: number; allocationRevision: number; entryQuantity: string; exitQuantity: string; gross: string };
export function allocationGrossRows(result: SourceResult): { policyRevision: number; rows: GrossAllocation[] } | null {
  const data = result.payload;
  if (!data || result.validity === "unavailable" || result.validity === "loading" || result.freshness === "unknown" ||
      data.source !== "postgres" || data.policy_validation_status !== "VALIDATED_POLICY") return null;
  const policy = object(object(data.trade_policy)?.policy);
  const overlay = object(policy?.allocation_overlay);
  if (policy?.scope !== "VALIDATED_PERFORMANCE" || !revision(policy.policy_revision) || overlay?.status !== "VALIDATED" ||
      overlay.policy_revision !== policy.policy_revision || !Array.isArray(overlay.allocations) || overlay.allocations.length > 16 ||
      overlay.active_count !== overlay.allocations.length) return null;
  const rows: GrossAllocation[] = [];
  for (const value of overlay.allocations) {
    const allocation = object(value);
    if (!allocation || !revision(allocation.trade_id) || !revision(allocation.allocation_revision) || allocation.state !== "ACTIVE" ||
        allocation.financial_eligible !== false || allocation.fees !== null || allocation.net !== null ||
        allocation.fees_status !== "UNKNOWN" || allocation.net_status !== "UNKNOWN") return null;
    const entry = decimal(allocation.entry_quantity), exit = decimal(allocation.exit_quantity), gross = decimal(allocation.gross_result);
    if (entry === null || exit === null || gross === null || entry === "0" || entry.startsWith("-") || entry !== exit ||
        rows.some(row => row.tradeId === allocation.trade_id)) return null;
    rows.push({tradeId: allocation.trade_id, allocationRevision: allocation.allocation_revision, entryQuantity: entry, exitQuantity: exit, gross});
  }
  return {policyRevision: policy.policy_revision, rows: rows.sort((left, right) => left.tradeId - right.tradeId)};
}

export default function AllocationGrossPanel({ result }: { result: SourceResult }) {
  const calculated = allocationGrossRows(result);
  return <section className="rounded-xl border border-slate-600 p-4 space-y-2" aria-label="Audited fill calculations">
    <h3>Audited fill calculations — gross only</h3>
    <p className="text-sm text-secondary">Calculated from the validated PostgreSQL fill allocation. Original ledger records are preserved. These calculations do not qualify net performance or establish account-wide coverage.</p>
    {!calculated ? <p>Allocation calculations unavailable: current validated allocation evidence is required.</p> : <>
      <p>Source: PostgreSQL allocation · Policy revision {calculated.policyRevision} · {result.freshness === "stale" ? "STALE — last successfully checked evidence; current state unverified" : "Evidence checked on the latest successful response; execution dates are historical"}</p>
      {calculated.rows.length ? <div className="overflow-x-auto"><table className="min-w-full text-sm" aria-label="Gross allocation calculations">
        <thead><tr>{["Trade", "Allocation revision", "Entry shares", "Exit shares", "Gross P/L before fees", "Fees", "Net P/L", "Financial qualification"].map(label => <th key={label}>{label}</th>)}</tr></thead>
        <tbody>{calculated.rows.map(row => <tr key={row.tradeId}><td>{row.tradeId}</td><td>{row.allocationRevision}</td><td>{row.entryQuantity}</td><td>{row.exitQuantity}</td><td>{formatAllocationGrossCurrency(row.gross)}</td><td>Unknown</td><td>Unknown</td><td>Not eligible</td></tr>)}</tbody>
      </table></div> : <p>No active audited fill allocations in the checked policy.</p>}
    </>}
  </section>;
}
