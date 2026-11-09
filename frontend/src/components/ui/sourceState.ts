import { useEffect, useState } from "react";
import { pollRead, readJson, READ_TIMEOUT_MS } from "./readLifecycle";
export { deadlineRead } from "./readLifecycle";

export type SourceKind = "loading" | "ready" | "empty" | "held" | "partial" | "unavailable" | "stale";
export type SourcePayload = Record<string, unknown>;
export const finiteNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};
export type SourceValidity = Exclude<SourceKind, "stale">;
export type SourceFreshness = "current" | "stale" | "unknown";
export function sourceValidity(value: unknown): SourceValidity {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "unavailable";
  const data = value as SourcePayload;
  // Financial validity precedes all successful-envelope and freshness fields.
  if (data.performance_status === "HELD_UNVALIDATED_PERFORMANCE") {
    return data.policy_validation_status === "UNAVAILABLE" ? "unavailable" : "held";
  }
  if (String(data.status ?? "").toLowerCase() === "held") return "held";
  if (data.ok !== true) return "unavailable";
  if (data.status === "partial") return "partial";
  if (data.status === "empty") return "empty";
  const rows = data.rows ?? data.positions;
  return Array.isArray(rows) && rows.length === 0 ? "empty" : "ready";
}
const displayKind = (validity: SourceValidity, freshness: SourceFreshness): SourceKind =>
  ["ready", "empty", "partial"].includes(validity) && freshness === "stale" ? "stale" : validity;
export function sourceKind(value: unknown): SourceKind {
  const data = value as SourcePayload | null;
  return displayKind(sourceValidity(value), data?.status === "stale" || data?.stale === true ? "stale" : "current");
}
export type SourceResult = {payload: SourcePayload | null; kind: SourceKind; validity: SourceValidity; freshness: SourceFreshness; receivedAt?: string};
export function watchSource(path: string, publish: (result: SourceResult) => void, interval = 20000, transport: typeof fetch = fetch, maxAgeMs = 60000, timeoutMs = READ_TIMEOUT_MS) {
  let active = true;
  let result: SourceResult = {payload: null, kind: "loading", validity: "loading", freshness: "unknown"};
  let observed = 0;
  const emit = (next: SourceResult) => {if (active) {result = next; publish(next);}};
  emit(result);
  const polling = pollRead({
    read: async signal => await readJson(path, signal, transport) as SourcePayload,
    timeout: timeoutMs,
    interval,
    success: payload => {
      const stamp = payload?.observed_at ?? payload?.taken_at_utc;
      observed = stamp ? Date.parse(String(stamp)) : Date.now();
      let validity = sourceValidity(payload);
      let freshness: SourceFreshness = payload?.status === "stale" || payload?.stale === true || Date.now() - observed >= maxAgeMs ? "stale" : "current";
      if (!Number.isFinite(observed) || observed - Date.now() > 5000) {
        freshness = "unknown";
        if (["ready", "empty", "partial"].includes(validity)) validity = "unavailable";
      }
      emit({payload, validity, freshness, kind: displayKind(validity, freshness), receivedAt: new Date().toISOString()});
    },
    failure: () => {
      const validity = result.validity === "loading" ? "unavailable" : result.validity;
      const freshness = result.payload ? "stale" : "unknown";
      emit({payload: result.payload, validity, freshness, kind: displayKind(validity, freshness), receivedAt: result.receivedAt});
    },
  });
  const expiry = globalThis.setInterval(() => {
    if (result.payload && result.freshness !== "stale" && Date.now() - observed >= maxAgeMs) emit({...result, freshness: "stale", kind: displayKind(result.validity, "stale")});
  }, Math.min(1000, maxAgeMs));
  return () => {active = false; polling.stop(); globalThis.clearInterval(expiry);};
}
export function useSource(path: string, interval = 20000, timeoutMs = READ_TIMEOUT_MS, maxAgeMs = 60000) {
  const initial: SourceResult = {payload: null, kind: "loading", validity: "loading", freshness: "unknown"};
  const [state, setState] = useState({path, result: initial});
  useEffect(() => watchSource(path, result => setState({path, result}), interval, fetch, maxAgeMs, timeoutMs), [path, interval, timeoutMs, maxAgeMs]);
  return state.path === path ? state.result : initial;
}

export const canDisplay = (kind: SourceKind) => ["ready", "empty", "partial", "stale"].includes(kind);
