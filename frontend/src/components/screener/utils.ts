import type { BacktestRow, LogLevel, MetricsRow, ScreenerLogRow, ScreenerPickRow } from "./types";

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const compactNumberFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 2,
});

export const withTs = (path: string) =>
  `${path}${path.includes("?") ? "&" : "?"}ts=${Date.now()}`;

export const parseNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

export const asString = (value: unknown, fallback = ""): string => {
  const normalized = String(value ?? "").trim();
  return normalized || fallback;
};

export const normalizeSymbol = (value: unknown): string => {
  return asString(value, "--").toUpperCase();
};

export const formatCurrency = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return currencyFormatter.format(value);
};

export const formatSignedCurrency = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${currencyFormatter.format(value)}`;
};

export const formatNumber = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return numberFormatter.format(value);
};

export const formatCompactNumber = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return compactNumberFormatter.format(value);
};

export const normalizePercent = (value: number | null | undefined): number | null => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return null;
  }
  if (Math.abs(value) <= 1) {
    return value * 100;
  }
  return value;
};

export const formatPercent = (value: number | null | undefined, signed = false): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const normalized = normalizePercent(value);
  if (normalized === null) {
    return "--";
  }
  const sign = signed && normalized > 0 ? "+" : "";
  return `${sign}${percentFormatter.format(normalized)}%`;
};

export const formatUtcDateTime = (value: string | null | undefined): string => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  const date = parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    timeZone: "UTC",
  });
  const time = parsed.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
  return `${date} ${time}`;
};

export const formatRunBadge = (value: string | null | undefined): string => {
  if (!value) {
    return "RUN: --";
  }
  return `RUN: ${formatUtcDateTime(value)} UTC`;
};

const includesQuery = (values: unknown[], query: string): boolean => {
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return true;
  }
  return values.some((value) =>
    String(value ?? "")
      .toLowerCase()
      .includes(needle)
  );
};

export const picksMatchQuery = (row: ScreenerPickRow, query: string): boolean => {
  return includesQuery(
    [
      row.rank,
      row.symbol,
      row.exchange,
      row.screened_at_utc,
      row.final_score,
      row.volume,
      row.price,
      row.entry_price,
      row.sma_ema_pct,
    ],
    query
  );
};

export const backtestMatchQuery = (row: BacktestRow, query: string): boolean => {
  return includesQuery(
    [
      row.symbol,
      row.window,
      row.trades,
      row.win_rate_pct,
      row.avg_return_pct,
      row.pl_ratio,
      row.max_dd_pct,
      row.avg_hold_days,
      row.total_pl_usd,
    ],
    query
  );
};

export const metricsMatchQuery = (row: MetricsRow, query: string): boolean => {
  return includesQuery(
    [
      row.symbol,
      row.score_breakdown_short,
      row.liquidity_gate,
      row.volatility_gate,
      row.trend_gate,
      row.bars_complete,
      row.confidence,
      row.source_label,
    ],
    query
  );
};

export const normalizeLogLevel = (value: unknown): LogLevel => {
  const normalized = String(value ?? "")
    .trim()
    .toUpperCase();
  if (normalized.startsWith("ERR")) {
    return "ERROR";
  }
  if (normalized.startsWith("WARN")) {
    return "WARN";
  }
  if (normalized === "SUCCESS" || normalized === "OK") {
    return "SUCCESS";
  }
  return "INFO";
};

export const logMatchQuery = (row: ScreenerLogRow, query: string): boolean => {
  return includesQuery([row.ts_utc, row.level, row.message], query);
};

export const compareNullableNumbers = (
  left: number | null | undefined,
  right: number | null | undefined
): number => {
  const a = left ?? Number.NEGATIVE_INFINITY;
  const b = right ?? Number.NEGATIVE_INFINITY;
  return a - b;
};
