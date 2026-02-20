import { useEffect, useMemo, useRef, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import LatestTradesTable from "./LatestTradesTable";
import TradesLeaderboard from "./TradesLeaderboard";
import TradesPerformanceBoard from "./TradesPerformanceBoard";
import { buildLeaderboardRequestPath, normalizeLeaderboardPayload } from "./leaderboardData";
import type {
  LatestTradeRow,
  LeaderRow,
  LeaderboardMode,
  RangeKey,
  RangeRowMetrics,
} from "./types";

const rangeOrder: RangeKey[] = ["d", "w", "m", "y", "all"];
const rangeLabels: Record<RangeKey, RangeRowMetrics["label"]> = {
  d: "DAILY",
  w: "WEEKLY",
  m: "MONTHLY",
  y: "YEARLY",
  all: "ALL",
};

const mockStatsRows: RangeRowMetrics[] = [
  {
    key: "d",
    label: "DAILY",
    winRatePct: 68.23,
    totalPL: 1100.0,
    topTrade: { symbol: "VTRS", pl: 475.5 },
    worstLoss: { symbol: "LC", pl: -165.05 },
    tradesCount: 3,
  },
  {
    key: "w",
    label: "WEEKLY",
    winRatePct: 52.5,
    totalPL: 900.0,
    topTrade: { symbol: "TSLA", pl: 820.0 },
    worstLoss: { symbol: "AMD", pl: -340.5 },
    tradesCount: 8,
  },
  {
    key: "m",
    label: "MONTHLY",
    winRatePct: 61.8,
    totalPL: 3420.75,
    topTrade: { symbol: "NVDA", pl: 1250.0 },
    worstLoss: { symbol: "PLTR", pl: -580.25 },
    tradesCount: 34,
  },
  {
    key: "y",
    label: "YEARLY",
    winRatePct: 58.4,
    totalPL: 24580.5,
    topTrade: { symbol: "AAPL", pl: 3200.0 },
    worstLoss: { symbol: "SNAP", pl: -1150.0 },
    tradesCount: 187,
  },
  {
    key: "all",
    label: "ALL",
    winRatePct: 59.2,
    totalPL: 48920.25,
    topTrade: { symbol: "MSFT", pl: 4500.0 },
    worstLoss: { symbol: "BYND", pl: -2100.0 },
    tradesCount: 428,
  },
];

const mockLatestTrades: LatestTradeRow[] = [
  {
    symbol: "VTRS",
    buyDate: "2026-02-01",
    sellDate: "2026-02-20",
    totalDays: 9,
    totalShares: 20,
    avgEntryPrice: 30.5,
    priceSold: 94.0,
    totalPL: 888.0,
  },
  {
    symbol: "AAPL",
    buyDate: "2026-01-15",
    sellDate: "2026-02-10",
    totalDays: 26,
    totalShares: 15,
    avgEntryPrice: 178.2,
    priceSold: 185.5,
    totalPL: 109.5,
  },
  {
    symbol: "TSLA",
    buyDate: "2026-01-22",
    sellDate: "2026-02-05",
    totalDays: 14,
    totalShares: 8,
    avgEntryPrice: 242.75,
    priceSold: 255.3,
    totalPL: 100.4,
  },
  {
    symbol: "AMD",
    buyDate: "2026-01-08",
    sellDate: "2026-01-28",
    totalDays: 20,
    totalShares: 25,
    avgEntryPrice: 145.8,
    priceSold: 138.2,
    totalPL: -190.0,
  },
  {
    symbol: "NVDA",
    buyDate: "2025-12-20",
    sellDate: "2026-01-18",
    totalDays: 29,
    totalShares: 12,
    avgEntryPrice: 485.0,
    priceSold: 520.5,
    totalPL: 426.0,
  },
];

const REFRESH_INTERVAL_MS = 15_000;
const LEADERBOARD_LIMIT = 12;

const parseNumber = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  return value as Record<string, unknown>;
};

const normalizeSymbol = (value: unknown): string => {
  const normalized = String(value ?? "")
    .trim()
    .toUpperCase();
  return normalized || "--";
};

const normalizeRangeKey = (value: unknown): RangeKey | null => {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();
  switch (normalized) {
    case "d":
    case "day":
    case "daily":
      return "d";
    case "w":
    case "week":
    case "weekly":
      return "w";
    case "m":
    case "month":
    case "monthly":
      return "m";
    case "y":
    case "year":
    case "yearly":
      return "y";
    case "a":
    case "all":
      return "all";
    default:
      return null;
  }
};

const normalizeStatsRow = (value: unknown, keyHint?: RangeKey): RangeRowMetrics | null => {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const key = normalizeRangeKey(record.key ?? record.range ?? record.window ?? keyHint);
  if (!key) {
    return null;
  }

  const topTradeRecord = asRecord(record.topTrade ?? record.top_trade);
  const worstLossRecord = asRecord(record.worstLoss ?? record.worst_loss);

  const winRateRaw =
    parseNumber(record.winRatePct ?? record.win_rate_pct ?? record.winRate ?? record.win_rate) ?? 0;
  const winRatePct = winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;

  return {
    key,
    label: rangeLabels[key],
    winRatePct,
    totalPL: parseNumber(record.totalPL ?? record.total_pl ?? record.net_pnl ?? record.pnl) ?? 0,
    topTrade: {
      symbol: normalizeSymbol(
        topTradeRecord?.symbol ?? topTradeRecord?.ticker ?? record.topTradeSymbol ?? record.top_trade_symbol
      ),
      pl: parseNumber(topTradeRecord?.pl ?? topTradeRecord?.pnl ?? record.topTradePL ?? record.top_trade_pl) ?? 0,
    },
    worstLoss: {
      symbol: normalizeSymbol(
        worstLossRecord?.symbol ??
          worstLossRecord?.ticker ??
          record.worstLossSymbol ??
          record.worst_loss_symbol
      ),
      pl:
        parseNumber(worstLossRecord?.pl ?? worstLossRecord?.pnl ?? record.worstLossPL ?? record.worst_loss_pl) ??
        0,
    },
    tradesCount:
      Math.max(
        0,
        Math.trunc(
          parseNumber(record.tradesCount ?? record.trades_count ?? record.totalTrades ?? record.total_trades) ?? 0
        )
      ) || 0,
  };
};

const normalizeStatsPayload = (payload: unknown): RangeRowMetrics[] => {
  const mapped = new Map<RangeKey, RangeRowMetrics>();

  const pushRow = (value: unknown, keyHint?: RangeKey) => {
    const row = normalizeStatsRow(value, keyHint);
    if (row) {
      mapped.set(row.key, row);
    }
  };

  if (Array.isArray(payload)) {
    payload.forEach((row) => pushRow(row));
  } else {
    const record = asRecord(payload);
    if (record) {
      const listPayload = record.rows ?? record.stats ?? record.data;
      if (Array.isArray(listPayload)) {
        listPayload.forEach((row) => pushRow(row));
      }

      const mapPayload = asRecord(record.byRange ?? record.ranges);
      if (mapPayload) {
        Object.entries(mapPayload).forEach(([key, row]) => {
          const normalizedKey = normalizeRangeKey(key);
          if (normalizedKey) {
            pushRow(row, normalizedKey);
          }
        });
      }

      rangeOrder.forEach((key) => {
        if (record[key] !== undefined) {
          pushRow(record[key], key);
        }
      });
    }
  }

  return rangeOrder.map((key) => mapped.get(key)).filter((row): row is RangeRowMetrics => Boolean(row));
};

const normalizeLatestRow = (value: unknown): LatestTradeRow | null => {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const symbol = normalizeSymbol(record.symbol ?? record.ticker);
  if (symbol === "--") {
    return null;
  }

  const buyDate = String(record.buyDate ?? record.buy_date ?? record.entry_time ?? "").trim();
  const sellDate = String(record.sellDate ?? record.sell_date ?? record.exit_time ?? "").trim();
  const totalDays = Math.max(
    0,
    Math.trunc(parseNumber(record.totalDays ?? record.total_days ?? record.hold_days) ?? 0)
  );
  const totalShares = Math.max(
    0,
    Math.trunc(parseNumber(record.totalShares ?? record.total_shares ?? record.qty) ?? 0)
  );

  return {
    symbol,
    buyDate,
    sellDate,
    totalDays,
    totalShares,
    avgEntryPrice: parseNumber(record.avgEntryPrice ?? record.avg_entry_price ?? record.entry_price) ?? 0,
    priceSold: parseNumber(record.priceSold ?? record.price_sold ?? record.exit_price) ?? 0,
    totalPL: parseNumber(record.totalPL ?? record.total_pl ?? record.realized_pnl ?? record.pnl) ?? 0,
  };
};

const normalizeLatestPayload = (payload: unknown): LatestTradeRow[] => {
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(asRecord(payload)?.rows)
      ? (asRecord(payload)?.rows as unknown[])
      : Array.isArray(asRecord(payload)?.trades)
        ? (asRecord(payload)?.trades as unknown[])
        : [];

  return rows.map((row) => normalizeLatestRow(row)).filter((row): row is LatestTradeRow => Boolean(row));
};

const withCacheBust = (path: string): string => {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}ts=${Date.now()}`;
};

const fetchJson = async <T,>(path: string): Promise<T | null> => {
  try {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
};

type TradesTabProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function TradesTab({ onSyncStateChange }: TradesTabProps) {
  const [selectedLeaderboardRange, setSelectedLeaderboardRange] = useState<RangeKey>("all");
  const [leaderboardMode, setLeaderboardMode] = useState<LeaderboardMode>("winners");

  const [statsRows, setStatsRows] = useState<RangeRowMetrics[]>([]);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState(false);
  const [leaderboardRows, setLeaderboardRows] = useState<LeaderRow[]>([]);
  const [leaderboardLoading, setLeaderboardLoading] = useState(true);
  const [leaderboardError, setLeaderboardError] = useState<string | null>(null);
  const [latestTradesRows, setLatestTradesRows] = useState<LatestTradeRow[]>([]);
  const [latestTradesLoading, setLatestTradesLoading] = useState(true);
  const [latestTradesError, setLatestTradesError] = useState(false);

  const hasLoadedStatsRef = useRef(false);
  const hasLoadedLeaderboardRef = useRef(false);
  const hasLoadedLatestRef = useRef(false);

  const isDev = import.meta.env.DEV;

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      if (!hasLoadedStatsRef.current) {
        setStatsLoading(true);
      }
      const payload = await fetchJson<unknown>(withCacheBust("/api/trades/stats?range=all"));
      if (!isMounted) {
        return;
      }
      const normalized = normalizeStatsPayload(payload);
      setStatsError(payload === null);
      setStatsRows(normalized.length > 0 ? normalized : isDev ? mockStatsRows : []);
      hasLoadedStatsRef.current = true;
      setStatsLoading(false);
    };

    load();
    const intervalId = window.setInterval(load, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [isDev]);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      if (!hasLoadedLeaderboardRef.current) {
        setLeaderboardLoading(true);
      }
      const payload = await fetchJson<unknown>(
        buildLeaderboardRequestPath(selectedLeaderboardRange, leaderboardMode, LEADERBOARD_LIMIT)
      );
      if (!isMounted) {
        return;
      }
      if (payload === null) {
        setLeaderboardError("Unable to load leaderboard");
        hasLoadedLeaderboardRef.current = true;
        setLeaderboardLoading(false);
        return;
      }
      const normalized = normalizeLeaderboardPayload(payload);
      setLeaderboardRows(normalized.slice(0, LEADERBOARD_LIMIT));
      setLeaderboardError(null);
      hasLoadedLeaderboardRef.current = true;
      setLeaderboardLoading(false);
    };

    load();
    const intervalId = window.setInterval(load, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [selectedLeaderboardRange, leaderboardMode]);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      if (!hasLoadedLatestRef.current) {
        setLatestTradesLoading(true);
      }
      const payload = await fetchJson<unknown>(withCacheBust("/api/trades/latest?limit=25"));
      if (!isMounted) {
        return;
      }
      const normalized = normalizeLatestPayload(payload);
      setLatestTradesError(payload === null);
      setLatestTradesRows(normalized.length > 0 ? normalized : isDev ? mockLatestTrades : []);
      hasLoadedLatestRef.current = true;
      setLatestTradesLoading(false);
    };

    load();
    const intervalId = window.setInterval(load, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, [isDev]);

  const normalizedStatsRows = useMemo(() => {
    if (statsRows.length) {
      return statsRows;
    }
    return [];
  }, [statsRows]);

  useEffect(() => {
    if (!onSyncStateChange) {
      return;
    }
    if (statsLoading || leaderboardLoading || latestTradesLoading) {
      onSyncStateChange("loading");
      return;
    }
    if (statsError || latestTradesError || Boolean(leaderboardError)) {
      onSyncStateChange("error");
      return;
    }
    onSyncStateChange("ready");
  }, [
    latestTradesError,
    latestTradesLoading,
    leaderboardError,
    leaderboardLoading,
    onSyncStateChange,
    statsError,
    statsLoading,
  ]);

  return (
    <section className="space-y-4" aria-label="Trades tab">
      <div className="grid gap-md xl:grid-cols-3 xl:items-stretch">
        <div className="contents xl:col-span-2 xl:block xl:space-y-4">
          <div className="order-1">
            <TradesPerformanceBoard rows={normalizedStatsRows} isLoading={statsLoading} />
          </div>
          <div className="order-3">
            <LatestTradesTable rows={latestTradesRows} isLoading={latestTradesLoading} />
          </div>
        </div>

        <div className="order-2 xl:order-none xl:col-span-1 xl:h-full xl:min-h-0">
          <TradesLeaderboard
            rows={leaderboardRows}
            selectedRange={selectedLeaderboardRange}
            onRangeChange={setSelectedLeaderboardRange}
            mode={leaderboardMode}
            onModeChange={setLeaderboardMode}
            isLoading={leaderboardLoading}
            errorMessage={leaderboardError}
          />
        </div>
      </div>
    </section>
  );
}
