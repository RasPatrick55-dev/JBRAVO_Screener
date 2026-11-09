import { useEffect, useState } from "react";
import { pollRead, READ_TIMEOUT_MS } from "../ui/readLifecycle";
import type { LogReceipt } from "./logEvidence";

export interface ProviderLogSource extends LogReceipt {
  source: string;
  text: string;
  entries?: Array<{ raw: string; ts_utc?: string | null; provenance_label?: string }>;
}
export type ProviderLogs = {
  kind: "loading" | "ready" | "empty" | "stale" | "unavailable";
  sources: ProviderLogSource[];
  receivedAt?: string;
};
const initial: ProviderLogs = { kind: "loading", sources: [] };
export const PROVIDER_READ_TIMEOUT_MS = READ_TIMEOUT_MS;

// One bounded provider read at a time. A failed refresh never substitutes unrelated
// local files, nor claims that no activity occurred. Cached provider evidence keeps
// its original receipt timestamps and is explicitly marked stale.
export function watchProviderLogs(
  publish: (value: ProviderLogs) => void,
  transport: typeof fetch = fetch,
  interval = 20000,
  timeout = PROVIDER_READ_TIMEOUT_MS,
) {
  let active = true;
  let state = initial;
  const emit = (next: ProviderLogs) => { if (active) { state = next; publish(next); } };
  emit(initial);
  const polling = pollRead({
    interval, timeout,
    read: async signal => {
        const response = await transport("/api/pythonanywhere/logs", {
          signal, cache: "no-store", headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error("provider_http_unavailable");
        const data = await response.json();
        if (data?.ok !== true || data.source !== "pythonanywhere" || !Array.isArray(data.sources)
          || data.sources.some((row: unknown) => !row || typeof row !== "object"
            || typeof (row as ProviderLogSource).source !== "string"
            || typeof (row as ProviderLogSource).text !== "string"
            || ((row as ProviderLogSource).entries !== undefined
              && (!Array.isArray((row as ProviderLogSource).entries)
                || (row as ProviderLogSource).entries!.some(entry => !entry || typeof entry.raw !== "string"
                  || (entry.ts_utc != null && typeof entry.ts_utc !== "string")
                  || (entry.provenance_label != null && typeof entry.provenance_label !== "string")))))) {
          throw new Error("provider_logs_unavailable");
        }
        return data.sources as ProviderLogSource[];
    },
    success: sources => {
      emit({ kind: sources.some(row => row.text.trim()) ? "ready" : "empty", sources,
        receivedAt: new Date().toISOString() });
    },
    failure: () => {
      emit({ ...state, kind: state.receivedAt ? "stale" : "unavailable" });
    },
  });
  return () => { active = false; polling.stop(); };
}

export function useProviderLogs() {
  const [state, setState] = useState<ProviderLogs>(initial);
  useEffect(() => watchProviderLogs(setState), []);
  return state;
}

export function providerLogNotice(state: ProviderLogs): string {
  if (state.kind === "loading") return "Loading PythonAnywhere task logs. Activity is not yet known.";
  if (state.kind === "unavailable") return "PythonAnywhere task logs unavailable. No local-file fallback is used; current activity is unknown.";
  if (state.kind === "stale") return `PythonAnywhere refresh failed. Showing the last successfully received provider sample (${state.receivedAt || "time unknown"}), not current activity. No local-file fallback is used.`;
  if (state.kind === "empty") return "PythonAnywhere returned an empty log sample. This does not establish that no activity occurred.";
  return "Source: PythonAnywhere API task-log sample. Fetch time is separate from event time; this is not a complete activity inventory.";
}
