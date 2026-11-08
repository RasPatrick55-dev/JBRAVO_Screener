// Calendar/session identifiers are not instants. Keep their source date and year.
export function formatRunDate(value: unknown): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return "Unknown";
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
    ? value : "Unknown";
}

function parseInstant(value: unknown): Date | null {
  // Require an explicit offset rather than interpreting incomplete evidence in
  // the browser's timezone. The overview API normalizes instants to UTC.
  if (typeof value !== "string") return null;
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.exec(value);
  if (!match || formatRunDate(match[1]) === "Unknown"
    || Number(match[2]) > 23 || Number(match[3]) > 59 || Number(match[4]) > 59) return null;
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

export function formatMlTimestamp(value: unknown): string {
  const parsed = parseInstant(value);
  if (!parsed) return "Unknown";
  return `${parsed.toLocaleString("en-US", {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC",
  })} UTC`;
}

export function formatMlInstantDate(value: unknown): string {
  const parsed = parseInstant(value);
  return parsed ? `${parsed.toISOString().slice(0, 10)} UTC` : "Unknown";
}

export function formatMlBoolean(value: unknown): string {
  return value === true ? "true" : value === false ? "false" : "Unknown";
}
