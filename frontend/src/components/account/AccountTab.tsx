import { useSource, canDisplay } from "../ui/sourceState";
import SourceNotice from "../ui/SourceNotice";
import { useEffect, useMemo } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
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

const performancePeriods = ["Daily", "Weekly", "Monthly", "Yearly"];

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
  if (!record || record.ok !== true) {
    return null;
  }
  return {
    equity: parseNumber(record.equity),
    cash: parseNumber(record.cash),
    buyingPower: parseNumber(record.buying_power),
    openPositionsValue: parseNumber(record.open_positions_value),
    cashToPositionsRatio: parseNumber(record.cash_to_positions_ratio),
    takenAtUtc: String(record.taken_at_utc ?? ""),
  };
};

const normalizePerformanceRows = (payload: unknown): { rows: AccountPerformanceRow[]; total: AccountTotal | null } => {
  const record = asRecord(payload);
  if (!record || record.ok !== true) {
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
        netChangePct: parseNumber(row.netChangePct),
        netChangeUsd: parseNumber(row.netChangeUsd),
      };
    })
    .filter((row): row is AccountPerformanceRow => Boolean(row));

  const totalRecord = asRecord(record.accountTotal);
  const total = totalRecord
      ? {
          equity: parseNumber(totalRecord.equity),
          netChangePct: parseNumber(totalRecord.netChangePct),
          netChangeUsd: parseNumber(totalRecord.netChangeUsd),
          equityBasis: normalizeEquityBasis(totalRecord.equityBasis),
          asOfUtc: String(totalRecord.asOfUtc ?? "").trim() || undefined,
          performanceBasis: normalizePerformanceBasis(totalRecord.performanceBasis),
        }
    : null;

  return { rows, total };
};

const normalizePortfolioPoints = (payload: unknown): EquityCurvePoint[] => {
  const record = asRecord(payload);
  if (!record || record.ok !== true || !Array.isArray(record.points)) {
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

export const normalizeOpenOrders = (payload: unknown): OpenOrderRow[] => {
  const record = asRecord(payload);
  if (!record || record.ok !== true || !Array.isArray(record.rows)) {
    return [];
  }
  return record.rows
    .map((item): OpenOrderRow | null => {
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
        qty: parseNumber(row.qty),
        priceOrStop: parseNumber(row.price_or_stop),
        submittedAt: String(row.submitted_at ?? "").trim(),
        status: String(row.status ?? "").trim(),
        raw_status: String(row.raw_status ?? "").trim(),
        order_id: String(row.order_id ?? "").trim(),
        replaces: String(row.replaces ?? "").trim(),
        replaced_by: String(row.replaced_by ?? "").trim(),
      };
    })
    .filter((row): row is OpenOrderRow => Boolean(row));
};

const normalizeOrderLogs = (payload: unknown): OrderLogRow[] => {
  const record = asRecord(payload);
  if (!record || record.ok !== true || !Array.isArray(record.rows)) {
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

type AccountTabProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function AccountTab({ onSyncStateChange }: AccountTabProps) {
  const summarySource = useSource("/api/account/summary");
  const performanceSource = useSource("/api/account/performance?range=all");
  const historySource = useSource("/api/account/portfolio_history?period=1Y&timeframe=1D");
  const logsSource = useSource("/api/account/order_logs?limit=100");
  const ordersSource = useSource("/api/account/open_orders?limit=50");
  const summary = normalizeSummary(canDisplay(summarySource.kind) ? summarySource.payload : null);
  const performance = normalizePerformanceRows(canDisplay(performanceSource.kind) ? performanceSource.payload : null);
  const performanceRows = performance.rows;
  const accountTotal = performance.total ?? { equity: null, netChangePct: null, netChangeUsd: null };
  const portfolioPoints = normalizePortfolioPoints(canDisplay(historySource.kind) ? historySource.payload : null);
  const orderLogs = normalizeOrderLogs(canDisplay(logsSource.kind) ? logsSource.payload : null);
  const openOrders = normalizeOpenOrders(canDisplay(ordersSource.kind) ? ordersSource.payload : null);
  const sources = [summarySource, performanceSource, historySource, logsSource, ordersSource];
  const anyLoading = sources.some(source => source.kind === "loading");
  const hasError = sources.some(source => !["loading", "ready", "empty"].includes(source.kind));

  useEffect(() => {
    if (!onSyncStateChange) {
      return;
    }
    if (anyLoading) {
      onSyncStateChange("loading");
      return;
    }
    if (hasError || !["ready", "empty"].includes(ordersSource.kind)) {
      onSyncStateChange("error");
      return;
    }
    onSyncStateChange("ready");
  }, [hasError, anyLoading, ordersSource.kind, onSyncStateChange]);

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
      return { period, netChangePct: null, netChangeUsd: null };
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
          Some account data is unavailable, partial, or stale. Unknown amounts are shown as --; stale samples are labeled with their last successful read. Source: configured Alpaca paper broker; historical trade accounting remains separately qualified.
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-12 xl:items-start">
        <div className="space-y-4 xl:col-span-5">
          <SourceNotice label="Account performance" {...performanceSource} />
          <AccountPerformanceCard
            rows={tableRows}
            total={accountTotal}
            isLoading={performanceSource.kind === "loading"}
          />
          <SourceNotice label="Account summary" {...summarySource} />
          <AccountBreakdownCard summary={summary} isLoading={summarySource.kind === "loading"} />
        </div>

        <div className="space-y-4 xl:col-span-7">
          <SourceNotice label="Account equity history" {...historySource} />
          <EquityCurveCard points={portfolioPoints} isLoading={historySource.kind === "loading"} sourceKind={historySource.kind} />
          <SourceNotice label="Open broker orders" {...ordersSource} />
          <OpenOrdersCard rows={openOrders} isLoading={ordersSource.kind === "loading"} sourceKind={ordersSource.kind} />
          <SourceNotice label="Account order logs" {...logsSource} />
          <DailyOrderLogsCard rows={orderLogs} isLoading={logsSource.kind === "loading"} sourceKind={logsSource.kind} />
        </div>
      </div>
    </section>
  );
}
