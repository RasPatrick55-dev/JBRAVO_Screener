import type { SourceKind, SourcePayload, SourceFreshness } from "./sourceState";

export default function SourceNotice({ label, kind, payload, freshness, receivedAt }: { label: string; kind: SourceKind; payload: SourcePayload | null; freshness?: SourceFreshness; receivedAt?: string }) {
  return <div role={kind === "unavailable" || kind === "held" ? "alert" : "status"} className="rounded-lg border border-slate-500/40 p-3 text-sm text-secondary">
    <strong>{label}: {kind.toUpperCase()}</strong>
    {kind === "held" ? " — validated performance is held; physical history below remains separate." : null}
    {kind === "unavailable" ? " — source could not establish current data; this is not a zero balance or empty ledger." : null}
    {kind === "stale" ? " — last known values; current source freshness is unverified." : null}
    {freshness === "stale" && kind !== "stale" ? <div>Last successful read is stale; business status remains {kind.toUpperCase()}.</div> : null}
    <div>Source: {String(payload?.source ?? "unknown")} | Last successful read (UTC): {receivedAt ?? "unknown"}</div>
    {payload?.observed_at ? <div>Source observed (UTC): {String(payload.observed_at)}</div> : null}
    {payload?.source_as_of ? <div>Source data as of (UTC): {String(payload.source_as_of)} | Freshness: {String(payload.freshness ?? "unknown")}</div> : null}
    {payload?.account_scope ? <div>Scope: {String(payload.account_scope)} · Broker association: {String(payload.broker_account_match ?? "not combined")}</div> : null}
  </div>;
}
