import type {
  ExecuteLogRow,
  ExecuteLsxFilter,
  ExecuteOrderRow,
  ExecuteStatusScope,
  ExecuteTrailingStopRow,
} from "./types";

export const LSX_CHIPS: Array<{ key: ExecuteLsxFilter; label: string }> = [
  { key: "l", label: "L" },
  { key: "s", label: "S" },
  { key: "e", label: "E" },
  { key: "x", label: "X" },
];

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

export const fetchJsonNoStore = async <T,>(path: string, timeoutMs = 15_000): Promise<T | null> => {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), Math.max(1_000, timeoutMs));
  try {
    const response = await fetch(path, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
};

export const parseSseJson = <T,>(raw: string): T | null => {
  const normalized = String(raw ?? "").trim();
  if (!normalized) {
    return null;
  }

  const candidates = [normalized];
  if (normalized.includes("\n")) {
    const dataLines = normalized
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.toLowerCase().startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .filter(Boolean);
    if (dataLines.length > 0) {
      candidates.unshift(dataLines.join("\n"));
    }
  }

  try {
    return JSON.parse(candidates[0]) as T;
  } catch {
    try {
      return JSON.parse(candidates[candidates.length - 1]) as T;
    } catch {
      return null;
    }
  }
};

export const auditSeverityChipClass = (severity: string): string => {
  const normalized = String(severity || "").trim().toLowerCase();
  if (normalized === "high" || normalized === "error") {
    return "bg-rose-500/15 text-rose-300 outline-rose-400/50";
  }
  if (normalized === "warning" || normalized === "warn") {
    return "bg-amber-500/15 text-amber-300 outline-amber-400/55";
  }
  return "bg-sky-500/15 text-sky-300 outline-sky-400/55";
};

export const formatAgeSeconds = (value: number | null | undefined): string => {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  const total = Math.max(0, Math.trunc(Number(value)));
  if (total < 60) {
    return `${total}s`;
  }
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes < 60) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return `${hours}h ${String(remMinutes).padStart(2, "0")}m`;
};

export const parseNumber = (value: string | number | null | undefined): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  const numeric = Number(String(value).trim());
  return Number.isFinite(numeric) ? numeric : null;
};

export const formatNumber = (value: string | number | null | undefined): string => {
  const numeric = parseNumber(value);
  if (numeric === null) {
    return "--";
  }
  return numberFormatter.format(numeric);
};

export const formatCurrency = (value: string | number | null | undefined): string => {
  const numeric = parseNumber(value);
  if (numeric === null) {
    return "--";
  }
  return currencyFormatter.format(numeric);
};

export const formatSignedCurrency = (value: string | number | null | undefined): string => {
  const numeric = parseNumber(value);
  if (numeric === null) {
    return "--";
  }
  return `${numeric >= 0 ? "+" : "-"}${currencyFormatter.format(Math.abs(numeric))}`;
};

export const formatTimeUtc = (value: string | null | undefined): string => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
  }
  return parsed.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
  });
};

export const formatDateUtc = (value: string | null | undefined): string => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
  }
  const year = parsed.getUTCFullYear();
  const month = String(parsed.getUTCMonth() + 1).padStart(2, "0");
  const day = String(parsed.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export const formatDateTimeUtc = (value: string | null | undefined): { date: string; time: string } => ({
  date: formatDateUtc(value),
  time: formatTimeUtc(value),
});

export const normalizeSide = (value: string | null | undefined): string => {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (normalized === "BUY" || normalized === "LONG") {
    return "BUY";
  }
  if (normalized === "SELL" || normalized === "SHORT") {
    return "SELL";
  }
  return normalized;
};

export const normalizeOrderStatus = (value: string | null | undefined): string => {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (normalized.includes("REJECT") || normalized.includes("CANCEL") || normalized.includes("EXPIRE")) {
    return "REJECTED";
  }
  if (normalized.includes("PARTIAL")) {
    return "PARTIAL";
  }
  if (normalized.includes("FILL") || normalized.includes("CLOSE")) {
    return "FILLED";
  }
  if (
    normalized.includes("PEND") ||
    normalized.includes("NEW") ||
    normalized.includes("ACCEPT") ||
    normalized.includes("OPEN")
  ) {
    return "PENDING";
  }
  return normalized || "PENDING";
};

export const normalizeTrailingStatus = (value: string | null | undefined): string => {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (
    normalized.includes("TRIGGER") ||
    normalized.includes("FILL") ||
    normalized.includes("CANCEL") ||
    normalized.includes("CLOSE")
  ) {
    return "TRIGGERED";
  }
  return "ACTIVE";
};

export const normalizeLogLevel = (value: string | null | undefined): string => {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (normalized === "WARN") {
    return "WARNING";
  }
  if (normalized === "ERR") {
    return "ERROR";
  }
  if (normalized === "SUCCESS") {
    return "SUCCESS";
  }
  if (normalized === "ERROR") {
    return "ERROR";
  }
  if (normalized === "WARNING") {
    return "WARNING";
  }
  return "INFO";
};

export const sideChipClass = (side: string): string => {
  if (side === "BUY") {
    return "bg-emerald-500/15 text-emerald-300 outline-emerald-400/50";
  }
  if (side === "SELL") {
    return "bg-rose-500/15 text-rose-300 outline-rose-400/50";
  }
  return "bg-slate-500/15 text-slate-300 outline-slate-500/50";
};

export const orderStatusChipClass = (status: string): string => {
  if (status === "FILLED") {
    return "bg-emerald-500/15 text-emerald-300 outline-emerald-400/50";
  }
  if (status === "PENDING") {
    return "bg-sky-500/15 text-sky-300 outline-sky-400/55";
  }
  if (status === "PARTIAL") {
    return "bg-amber-500/15 text-amber-300 outline-amber-400/55";
  }
  if (status === "REJECTED") {
    return "bg-rose-500/15 text-rose-300 outline-rose-400/50";
  }
  return "bg-slate-500/15 text-slate-300 outline-slate-500/50";
};

export const trailingStatusChipClass = (status: string): string => {
  if (status === "ACTIVE") {
    return "bg-emerald-500/15 text-emerald-300 outline-emerald-400/50";
  }
  return "bg-amber-500/15 text-amber-300 outline-amber-400/55";
};

export const logLevelChipClass = (level: string): string => {
  if (level === "INFO") {
    return "bg-sky-500/15 text-sky-300 outline-sky-400/55";
  }
  if (level === "SUCCESS") {
    return "bg-emerald-500/15 text-emerald-300 outline-emerald-400/50";
  }
  if (level === "WARNING") {
    return "bg-amber-500/15 text-amber-300 outline-amber-400/55";
  }
  if (level === "ERROR") {
    return "bg-rose-500/15 text-rose-300 outline-rose-400/50";
  }
  return "bg-slate-500/15 text-slate-300 outline-slate-500/50";
};

export const cycleStatusScope = (scope: ExecuteStatusScope): ExecuteStatusScope => {
  if (scope === "all") {
    return "open";
  }
  if (scope === "open") {
    return "closed";
  }
  return "all";
};

export const statusScopeLabel = (scope: ExecuteStatusScope): string => {
  if (scope === "open") {
    return "Open";
  }
  if (scope === "closed") {
    return "Closed";
  }
  return "All";
};

const matchesLsx = (filter: ExecuteLsxFilter, text: string, side: string): boolean => {
  if (filter === "all") {
    return true;
  }
  const haystack = text.toLowerCase();
  const normalizedSide = side.toLowerCase();
  if (filter === "l") {
    return normalizedSide === "buy" || haystack.includes(" long") || haystack.includes("entry");
  }
  if (filter === "s") {
    return normalizedSide === "sell" || haystack.includes(" short") || haystack.includes("trail");
  }
  if (filter === "e") {
    return haystack.includes("entry") || haystack.includes("submit") || normalizedSide === "buy";
  }
  return haystack.includes("exit") || haystack.includes("close") || normalizedSide === "sell";
};

const matchesStatusScope = (scope: ExecuteStatusScope, status: string): boolean => {
  if (scope === "all") {
    return true;
  }
  if (scope === "open") {
    return ["PENDING", "PARTIAL", "ACTIVE"].includes(status);
  }
  return ["FILLED", "REJECTED", "TRIGGERED"].includes(status);
};

const includesQuery = (query: string, text: string): boolean => {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  return text.toLowerCase().includes(normalized);
};

export const filterOrdersClient = (
  rows: ExecuteOrderRow[],
  query: string,
  statusScope: ExecuteStatusScope,
  lsx: ExecuteLsxFilter
): ExecuteOrderRow[] => {
  return rows.filter((row) => {
    const side = normalizeSide(row.side);
    const status = normalizeOrderStatus(row.status);
    const text = [
      row.ts_utc,
      row.symbol,
      side,
      row.type,
      row.qty,
      row.limit_stop_trail,
      status,
      row.filled_avg,
      row.order_id,
      row.notes,
    ]
      .map((value) => String(value ?? ""))
      .join(" ");
    return (
      includesQuery(query, text) &&
      matchesStatusScope(statusScope, status) &&
      matchesLsx(lsx, text, side)
    );
  });
};

export const filterTrailingClient = (
  rows: ExecuteTrailingStopRow[],
  query: string,
  statusScope: ExecuteStatusScope,
  lsx: ExecuteLsxFilter
): ExecuteTrailingStopRow[] => {
  return rows.filter((row) => {
    const status = normalizeTrailingStatus(row.status);
    const text = [row.symbol, row.qty, row.trail, row.stop_price, status, row.parent_leg]
      .map((value) => String(value ?? ""))
      .join(" ");
    const sideHint = status === "ACTIVE" ? "sell" : "";
    return (
      includesQuery(query, text) &&
      matchesStatusScope(statusScope, status) &&
      matchesLsx(lsx, text, sideHint)
    );
  });
};

export const filterLogsClient = (
  rows: ExecuteLogRow[],
  query: string,
  lsx: ExecuteLsxFilter
): ExecuteLogRow[] => {
  return rows.filter((row) => {
    const level = normalizeLogLevel(row.level);
    const text = [row.ts_utc, level, row.message].map((value) => String(value ?? "")).join(" ");
    return includesQuery(query, text) && matchesLsx(lsx, text, "");
  });
};
