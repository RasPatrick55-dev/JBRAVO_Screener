// These displays accept finite numbers and decimal API strings, never JavaScript
// coercions such as true, whitespace, [] or hexadecimal text becoming zero.
export function evidenceNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string" || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function evidenceCount(value: unknown): number | null {
  const number = evidenceNumber(value);
  return number !== null && Number.isSafeInteger(number) && number >= 0 ? number : null;
}

export const evidenceText = (value: unknown): string => typeof value === "string" ? value.trim() : "";
export const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
export const isDigest = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
export const isStrategyId = (value: unknown): value is string => typeof value === "string" && /^[a-z][a-z0-9_]{0,127}$/.test(value);
export const isVersion = (value: unknown): value is string => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value);
export const evidenceLabel = (value: unknown): string => evidenceText(value).replace(/_/g, " ").toLowerCase() || "unknown";
export const formatRatio = (value: number | null): string => value === null ? "--" : value.toFixed(2);
export const evidenceTone = (value: number | null): string => value === null || value === 0 ? "text-primary"
  : value > 0 ? "jbravo-text-success" : "jbravo-text-error";
