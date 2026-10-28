import type { LeaderRow, LeaderboardMode, RangeKey } from "./types";

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  return value as Record<string, unknown>;
};

const parseNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const normalizeSymbol = (value: unknown): string => {
  const normalized = String(value ?? "")
    .trim()
    .toUpperCase();
  return normalized || "--";
};

const normalizeLeaderboardRow = (value: unknown, index: number): LeaderRow | null => {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const symbol = normalizeSymbol(record.symbol ?? record.ticker);
  if (symbol === "--") {
    return null;
  }

  const pl = parseNumber(record.pl ?? record.totalPL ?? record.total_pl ?? record.pnl ?? record.net_pnl);
  if (pl === null) return null;
  return {
    rank: Math.max(1, Math.trunc(parseNumber(record.rank) ?? index + 1)),
    symbol,
    pl,
  };
};

export const normalizeLeaderboardPayload = (payload: unknown): LeaderRow[] => {
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(asRecord(payload)?.rows)
      ? (asRecord(payload)?.rows as unknown[])
      : Array.isArray(asRecord(payload)?.leaderboard)
        ? (asRecord(payload)?.leaderboard as unknown[])
        : [];

  return rows
    .map((row, index) => normalizeLeaderboardRow(row, index))
    .filter((row): row is LeaderRow => Boolean(row));
};

export const buildLeaderboardRequestPath = (
  range: RangeKey,
  mode: LeaderboardMode,
  limit: number,
  basis?: "recorded_gross"
): string => {
  const query = new URLSearchParams({
    range,
    mode,
    limit: String(limit),
  });
  if (basis) query.set("basis", basis);
  return `/api/trades/leaderboard?${query.toString()}`;
};
