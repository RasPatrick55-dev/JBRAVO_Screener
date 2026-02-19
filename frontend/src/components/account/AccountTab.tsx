import { useEffect, useMemo, useRef, useState } from "react";
import AccountBreakdownCard from "./AccountBreakdownCard";
import AccountPerformanceCard from "./AccountPerformanceCard";
import DailyOrderLogsCard from "./DailyOrderLogsCard";
import EquityCurveCard from "./EquityCurveCard";
import OpenOrdersCard from "./OpenOrdersCard";
import type {
  AccountPerformanceRow,
  AccountSummary,
  AccountTotal,
  EquityCurvePoint,
  OpenOrderRow,
  OrderLogLevel,
  OrderLogRow,
} from "./types";

const REFRESH_INTERVAL_MS = 20_000;
const FETCH_TIMEOUT_MS = 20_000;
const performancePeriods = ["Daily", "Weekly", "Monthly", "Yearly"];

const withTs = (path: string) => `${path}${path.includes("?") ? "&" : "?"}ts=${Date.now()}`;

const fetchJsonAttempt = async <T,>(url: string): Promise<T | null> => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timeoutId);
  }
};

const localApiOrigins = (): string[] => {
  const explicit = String(import.meta.env.VITE_API_BASE_URL ?? "").trim();
  if (explicit) {
    return [explicit.replace(/\/+$/, "")];
  }
  if (typeof window === "undefined") {
    return [];
  }
  const hostname = window.location.hostname;
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return ["http://127.0.0.1:8050", "http://localhost:8050"];
  }
  return [];
};

const fetchJson = async <T,>(path: string): Promise<T | null> => {
  const primary = await fetchJsonAttempt<T>(path);
  if (primary !== null) {
    return primary;
  }
  if (/^https?:\/\//i.test(path)) {
    return null;
  }
  const apiOrigins = localApiOrigins();
  for (const origin of apiOrigins) {
    const fallback = await fetchJsonAttempt<T>(`${origin}${path}`);
    if (fallback !== null) {
      return fallback;
    }
  }
  return null;
};

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
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

const normalizePeriod = (value: unknown): string | null => {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();
  switch (normalized) {
    case "d":
    case "day":
    case "daily":
      return "Daily";
    case "w":
    case "week":
    case "weekly":
      return "Weekly";
    case "m":
    case "month":
    case "monthly":
      return "Monthly";
    case "y":
    case "year":
    case "yearly":
      return "Yearly";
    default:
      return null;
  }
};

const normalizeOrderLogLevel = (value: unknown): OrderLogLevel => {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase();
  if (normalized === "success") {
    return "success";
  }
  if (normalized === "warning" || normalized === "warn" || normalized === "error") {
    return "warning";
  }
  return "info";
};

const normalizeEquityBasis = (
  value: unknown
): AccountTotal["equityBasis"] => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "live") {
    return "live";
  }
  if (normalized === "last_close") {
    return "last_close";
  }
  return undefined;
};

const normalizePerformanceBasis = (
  value: unknown
): AccountTotal["performanceBasis"] => {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "live_vs_close_baselines") {
    return "live_vs_close_baselines";
  }
  if (normalized === "close_to_close") {
    return "close_to_close";
  }
  return undefined;
};

const normalizeSummary = (payload: unknown): AccountSummary | null => {
  const record = asRecord(payload);
  if (!record) {
    return null;
  }
  return {
    equity: parseNumber(record.equity) ?? 0,
    cash: parseNumber(record.cash) ?? 0,
    buyingPower: parseNumber(record.buying_power) ?? 0,
    openPositionsValue: parseNumber(record.open_positions_value) ?? 0,
    cashToPositionsRatio: parseNumber(record.cash_to_positions_ratio),
    takenAtUtc: String(record.taken_at_utc ?? ""),
  };
};

const normalizePerformanceRows = (payload: unknown): { rows: AccountPerformanceRow[]; total: AccountTotal | null } => {
  const record = asRecord(payload);
  if (!record) {
    return { rows: [], total: null };
  }

  const list = Array.isArray(record.rows) ? record.rows : [];
  const rows: AccountPerformanceRow[] = list
    .map((item) => {
      const row = asRecord(item);
      if (!row) {
        return null;
      }
      const period = normalizePeriod(row.period);
      if (!period) {
        return null;
      }
      return {
        period,
        netChangePct: parseNumber(row.netChangePct) ?? 0,
        netChangeUsd: parseNumber(row.netChangeUsd) ?? 0,
      };
    })
    .filter((row): row is AccountPerformanceRow => Boolean(row));

  const totalRecord = asRecord(record.accountTotal);
  const total = totalRecord
      ? {
          equity: parseNumber(totalRecord.equity) ?? 0,
          netChangePct: parseNumber(totalRecord.netChangePct) ?? 0,
          netChangeUsd: parseNumber(totalRecord.netChangeUsd) ?? 0,
          equityBasis: normalizeEquityBasis(totalRecord.equityBasis),
          asOfUtc: String(totalRecord.asOfUtc ?? "").trim() || undefined,
          performanceBasis: normalizePerformanceBasis(totalRecord.performanceBasis),
        }
    : null;

  return { rows, total };
};

const normalizePortfolioPoints = (payload: unknown): EquityCurvePoint[] => {
  const record = asRecord(payload);
  if (!record || !Array.isArray(record.points)) {
    return [];
  }
  return record.points
    .map((item) => {
      const row = asRecord(item);
      if (!row) {
        return null;
      }
      const equity = parseNumber(row.equity);
      const t = String(row.t ?? "").trim();
      if (equity === null || !t) {
        return null;
      }
      return { t, equity };
    })
    .filter((row): row is EquityCurvePoint => Boolean(row));
};

const normalizeOpenOrders = (payload: unknown): OpenOrderRow[] => {
  const record = asRecord(payload);
  if (!record || !Array.isArray(record.rows)) {
    return [];
  }
  return record.rows
    .map((item) => {
      const row = asRecord(item);
      if (!row) {
        return null;
      }
      return {
        symbol: String(row.symbol ?? "")
          .trim()
          .toUpperCase(),
        type: String(row.type ?? "market")
          .trim()
          .toLowerCase(),
        side: String(row.side ?? "buy")
          .trim()
          .toLowerCase(),
        qty: parseNumber(row.qty) ?? 0,
        priceOrStop: parseNumber(row.price_or_stop),
        submittedAt: String(row.submitted_at ?? "").trim(),
      };
    })
    .filter((row): row is OpenOrderRow => Boolean(row));
};

const normalizeOrderLogs = (payload: unknown): OrderLogRow[] => {
  const record = asRecord(payload);
  if (!record || !Array.isArray(record.rows)) {
    return [];
  }
  return record.rows
    .map((item) => {
      const row = asRecord(item);
      if (!row) {
        return null;
      }
      const ts = String(row.ts ?? "").trim();
      const message = String(row.message ?? "").trim();
      if (!ts || !message) {
        return null;
      }
      return {
        ts,
        level: normalizeOrderLogLevel(row.level),
        message,
      };
    })
    .filter((row): row is OrderLogRow => Boolean(row));
};

interface LoadingState {
  summary: boolean;
  performance: boolean;
  history: boolean;
  openOrders: boolean;
  logs: boolean;
}

const defaultLoadingState: LoadingState = {
  summary: true,
  performance: true,
  history: true,
  openOrders: true,
  logs: true,
};

export default function AccountTab() {
  const [summary, setSummary] = useState<AccountSummary | null>(null);
  const [performanceRows, setPerformanceRows] = useState<AccountPerformanceRow[]>([]);
  const [accountTotal, setAccountTotal] = useState<AccountTotal>({
    equity: 0,
    netChangePct: 0,
    netChangeUsd: 0,
    equityBasis: "last_close",
    performanceBasis: "close_to_close",
  });
  const [portfolioPoints, setPortfolioPoints] = useState<EquityCurvePoint[]>([]);
  const [openOrders, setOpenOrders] = useState<OpenOrderRow[]>([]);
  const [orderLogs, setOrderLogs] = useState<OrderLogRow[]>([]);
  const [loading, setLoading] = useState<LoadingState>(defaultLoadingState);
  const [hasError, setHasError] = useState(false);

  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      if (!hasLoadedRef.current) {
        setLoading(defaultLoadingState);
      } else {
        setLoading((previous) => ({ ...previous, performance: true }));
      }

      const [summaryResult, performanceResult, historyResult, openOrdersResult, logsResult] =
        await Promise.allSettled([
          fetchJson<unknown>(withTs("/api/account/summary")),
          fetchJson<unknown>(withTs("/api/account/performance?range=all")),
          fetchJson<unknown>(withTs("/api/account/portfolio_history?period=1Y&timeframe=1D")),
          fetchJson<unknown>(withTs("/api/account/open_orders?limit=50")),
          fetchJson<unknown>(withTs("/api/account/order_logs?limit=100")),
        ]);

      if (!isMounted) {
        return;
      }

      const summaryPayload = summaryResult.status === "fulfilled" ? summaryResult.value : null;
      const performancePayload = performanceResult.status === "fulfilled" ? performanceResult.value : null;
      const historyPayload = historyResult.status === "fulfilled" ? historyResult.value : null;
      const openOrdersPayload = openOrdersResult.status === "fulfilled" ? openOrdersResult.value : null;
      const logsPayload = logsResult.status === "fulfilled" ? logsResult.value : null;

      const normalizedSummary = normalizeSummary(summaryPayload);
      const normalizedPerformance = normalizePerformanceRows(performancePayload);
      const normalizedPoints = normalizePortfolioPoints(historyPayload);
      const normalizedOpenOrders = normalizeOpenOrders(openOrdersPayload);
      const normalizedLogs = normalizeOrderLogs(logsPayload);

      if (summaryPayload) {
        setSummary(normalizedSummary);
      }
      if (performancePayload) {
        setPerformanceRows(normalizedPerformance.rows);
        setAccountTotal(
          normalizedPerformance.total ?? {
            equity: normalizedSummary?.equity ?? 0,
            netChangePct: 0,
            netChangeUsd: 0,
            equityBasis: "live",
            asOfUtc: normalizedSummary?.takenAtUtc ?? "",
            performanceBasis: "close_to_close",
          }
        );
      } else if (summaryPayload && normalizedSummary) {
        setAccountTotal((previous) => ({
          ...previous,
          equity: normalizedSummary.equity,
          equityBasis: "live",
          asOfUtc: normalizedSummary.takenAtUtc,
        }));
      }
      if (historyPayload) {
        setPortfolioPoints(normalizedPoints);
      }
      if (openOrdersPayload) {
        setOpenOrders(normalizedOpenOrders);
      }
      if (logsPayload) {
        setOrderLogs(normalizedLogs);
      }
      setHasError(!summaryPayload || !performancePayload || !historyPayload || !openOrdersPayload || !logsPayload);
      setLoading({
        summary: false,
        performance: false,
        history: false,
        openOrders: false,
        logs: false,
      });
      hasLoadedRef.current = true;
    };

    load();
    const intervalId = window.setInterval(load, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const tableRows = useMemo(() => {
    const byPeriod = new Map<string, AccountPerformanceRow>();
    performanceRows.forEach((row) => {
      byPeriod.set(row.period.toLowerCase(), row);
    });

    return performancePeriods.map((period) => {
      const existing = byPeriod.get(period.toLowerCase());
      if (existing) {
        return existing;
      }
      return { period, netChangePct: 0, netChangeUsd: 0 };
    });
  }, [performanceRows]);

  return (
    <section className="space-y-4" aria-label="Account tab">
      <header className="space-y-1">
        <h1 className="text-3xl font-semibold tracking-tight text-financial sm:text-4xl">Account</h1>
        <p className="text-sm text-secondary sm:text-base">Performance, equity trend, open orders, and daily logs</p>
      </header>

      {hasError ? (
        <div
          role="alert"
          className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-300/35 dark:bg-amber-500/10 dark:text-amber-200"
        >
          Some account data could not be refreshed. Showing the latest available values.
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-12 xl:items-start">
        <div className="space-y-4 xl:col-span-5">
          <AccountPerformanceCard
            rows={tableRows}
            total={accountTotal}
            isLoading={loading.performance}
          />
          <AccountBreakdownCard summary={summary} isLoading={loading.summary} />
        </div>

        <div className="space-y-4 xl:col-span-7">
          <EquityCurveCard points={portfolioPoints} isLoading={loading.history} />
          <OpenOrdersCard rows={openOrders} isLoading={loading.openOrders} />
          <DailyOrderLogsCard rows={orderLogs} isLoading={loading.logs} />
        </div>
      </div>
    </section>
  );
}
