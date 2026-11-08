export interface LogReceipt {
  source?: string | null;
  timestamp_basis?: string | null;
  observed_at?: string | null;
  latest_event_at?: string | null;
  unknown_timestamp_count?: number;
}
export const eventFreshness = (timestamp?: string | null, now = Date.now()) => {
  const time = timestamp ? Date.parse(timestamp) : NaN;
  if (!Number.isFinite(time) || time > now) return "Age unknown";
  const hours = (now - time) / 3600000;
  return `${hours > 48 ? "Historical / stale" : "Recent (within 48h)"} · ${hours < 1 ? "<1" : Math.floor(hours)}h old`;
};
export function LogSourceReceipt({ receipt }: { receipt: LogReceipt }) {
  return <p className="mt-1 break-words text-[10px] text-secondary">
    {receipt.source || "Source unknown"} (returned log sample): {eventFreshness(receipt.latest_event_at)}.
    {" Latest event: "}{receipt.latest_event_at || "unknown"}.
    {" Fetched: "}{receipt.observed_at || "unknown"}.
    {" "}{receipt.timestamp_basis || "Timestamp basis unknown"}.
    {" Untimed entries: "}{receipt.unknown_timestamp_count ?? "unknown"}.
  </p>;
}
