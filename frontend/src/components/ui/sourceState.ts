import { useEffect, useState } from "react";

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
// Deadline includes body decoding; a transport that ignores abort cannot publish a late result.
export async function deadlineRead<T>(operation: (signal: AbortSignal) => Promise<T>, milliseconds = 10000): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([operation(controller.signal), new Promise<never>((_resolve, reject) => {
      timer = globalThis.setTimeout(() => {controller.abort(); reject(new Error("source_timeout"));}, milliseconds);
    })]);
  } finally { globalThis.clearTimeout(timer); }
}
export function watchSource(path: string, publish: (result: SourceResult) => void, interval = 20000, transport: typeof fetch = fetch, maxAgeMs = 60000) {
  let active = true, inFlight = false;
  let result: SourceResult = {payload: null, kind: "loading", validity: "loading", freshness: "unknown"};
  let observed = 0;
  const emit = (next: SourceResult) => {if (active) {result = next; publish(next);}};
  emit(result);
  const load = async () => {
    if (inFlight || !active) return;
    inFlight = true;
    try {
      const payload = await deadlineRead(async signal => {
        const response = await transport(path, {cache: "no-store", signal, headers: {Accept: "application/json"}});
        if (!response.ok) throw new Error("source_http_unavailable");
        return await response.json() as SourcePayload;
      });
      const stamp = payload?.observed_at ?? payload?.taken_at_utc;
      observed = stamp ? Date.parse(String(stamp)) : Date.now();
      let validity = sourceValidity(payload);
      let freshness: SourceFreshness = payload?.status === "stale" || payload?.stale === true || Date.now() - observed >= maxAgeMs ? "stale" : "current";
      if (!Number.isFinite(observed) || observed - Date.now() > 5000) {
        freshness = "unknown";
        if (["ready", "empty", "partial"].includes(validity)) validity = "unavailable";
      }
      emit({payload, validity, freshness, kind: displayKind(validity, freshness), receivedAt: new Date().toISOString()});
    } catch {
      const validity = result.validity === "loading" ? "unavailable" : result.validity;
      const freshness = result.payload ? "stale" : "unknown";
      emit({payload: result.payload, validity, freshness, kind: displayKind(validity, freshness), receivedAt: result.receivedAt});
    } finally {inFlight = false;}
  };
  void load();
  const poll = globalThis.setInterval(() => void load(), interval);
  const expiry = globalThis.setInterval(() => {
    if (result.payload && result.freshness !== "stale" && Date.now() - observed >= maxAgeMs) emit({...result, freshness: "stale", kind: displayKind(result.validity, "stale")});
  }, Math.min(1000, maxAgeMs));
  return () => {active = false; globalThis.clearInterval(poll); globalThis.clearInterval(expiry);};
}
export function useSource(path: string, interval = 20000) {
  const [result, setResult] = useState<SourceResult>({payload: null, kind: "loading", validity: "loading", freshness: "unknown"});
  useEffect(() => watchSource(path, setResult, interval), [path, interval]);
  return result;
}

export const canDisplay = (kind: SourceKind) => ["ready", "empty", "partial", "stale"].includes(kind);
