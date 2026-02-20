import { useEffect, useMemo, useRef, useState } from "react";
import { buildNavbarBadges, type LiveDataSyncState, useLiveTradingStatus } from "../components/navbar/liveStatus";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import type { MonitoringLogItem } from "../components/positions/MonitoringLogsPanel";
import type { PositionRowProps } from "../components/positions/PositionRow";
import type { PositionSummaryProps } from "../components/positions/PositionSummary";
import PositionsTab from "../components/positions/PositionsTab";

type PositionsMonitoringProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

type MonitoringPosition = {
  symbol?: string | null;
  logoUrl?: string | null;
  qty?: number | string | null;
  entryPrice?: number | string | null;
  currentPrice?: number | string | null;
  dollarPL?: number | string | null;
  daysHeld?: number | string | null;
  trailingStop?: number | string | null;
  capturedPL?: number | string | null;
};

type MonitoringSummary = {
  totalShares?: number | string | null;
  totalOpenPL?: number | string | null;
  avgDaysHeld?: number | string | null;
  totalCapturedPL?: number | string | null;
};

type MonitoringPositionsResponse = {
  ok?: boolean;
  source?: string;
  calculationSource?: string;
  positions?: MonitoringPosition[];
  summary?: MonitoringSummary;
};

type PositionsLogsResponse = {
  ok?: boolean;
  source?: string;
  logs?: {
    timestamp?: string | null;
    type?: string | null;
    message?: string | null;
  }[];
};

const navLabels = [
  "Dashboard",
  "Account",
  "Trades",
  "Positions",
  "Execute",
  "Screener",
  "ML Pipeline",
];
const REFRESH_INTERVAL_MS = 20_000;
const REQUEST_TIMEOUT_MS = 15_000;

const parseNumber = (value: string | number | null | undefined): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : null;
};

const normalizeSymbol = (symbol: string | null | undefined): string => {
  const normalized = (symbol ?? "").trim().toUpperCase();
  return normalized || "--";
};

const normalizeLogType = (type: string | null | undefined): MonitoringLogItem["type"] => {
  if (type === "success" || type === "warning" || type === "info") {
    return type;
  }
  return "info";
};

const fetchJson = async <T,>(path: string): Promise<T | null> => {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
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
    globalThis.clearTimeout(timeoutId);
  }
};

const toPositionRow = (position: MonitoringPosition): PositionRowProps => {
  const symbol = normalizeSymbol(position.symbol);
  const shares = parseNumber(position.qty) ?? 0;
  const entryPrice = parseNumber(position.entryPrice) ?? 0;
  const currentPrice = parseNumber(position.currentPrice) ?? entryPrice;
  const openPL = parseNumber(position.dollarPL) ?? 0;
  const daysHeldRaw = parseNumber(position.daysHeld) ?? 0;
  const daysHeld = Math.max(0, Math.round(daysHeldRaw));
  const trailingStop = parseNumber(position.trailingStop) ?? currentPrice;
  const capturedPL = parseNumber(position.capturedPL) ?? 0;

  return {
    symbol,
    logoUrl: position.logoUrl ?? "",
    shares,
    entryPrice,
    currentPrice,
    openPL,
    daysHeld,
    trailingStop,
    capturedPL,
  };
};

const summaryFromRows = (rows: PositionRowProps[]): PositionSummaryProps => {
  if (rows.length === 0) {
    return {
      totalShares: 0,
      totalOpenPL: 0,
      avgDaysHeld: 0,
      totalCapturedPL: 0,
    };
  }

  const totals = rows.reduce(
    (accumulator, row) => ({
      shares: accumulator.shares + row.shares,
      openPL: accumulator.openPL + row.openPL,
      days: accumulator.days + row.daysHeld,
      capturedPL: accumulator.capturedPL + row.capturedPL,
    }),
    { shares: 0, openPL: 0, days: 0, capturedPL: 0 }
  );

  return {
    totalShares: totals.shares,
    totalOpenPL: totals.openPL,
    avgDaysHeld: totals.days / rows.length,
    totalCapturedPL: totals.capturedPL,
  };
};

export default function PositionsMonitoring({ activeTab, onTabSelect }: PositionsMonitoringProps) {
  const [monitoringSnapshot, setMonitoringSnapshot] = useState<MonitoringPositionsResponse | null>(null);
  const [logsSnapshot, setLogsSnapshot] = useState<PositionsLogsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const liveTradingStatus = useLiveTradingStatus();
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let isMounted = true;
    let inFlight = false;

    const load = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      setIsLoading(!hasLoadedRef.current);
      const [monitoringPayload, logsPayload] = await Promise.all([
        fetchJson<MonitoringPositionsResponse>(`/api/positions/monitoring?ts=${Date.now()}`),
        fetchJson<PositionsLogsResponse>(`/api/positions/logs?limit=200&ts=${Date.now()}`),
      ]);

      if (!isMounted) {
        inFlight = false;
        return;
      }

      if (monitoringPayload) {
        setMonitoringSnapshot(monitoringPayload);
      } else if (!hasLoadedRef.current) {
        setMonitoringSnapshot({ ok: false, positions: [], summary: {} });
      }

      if (logsPayload) {
        setLogsSnapshot(logsPayload);
      } else if (!hasLoadedRef.current) {
        setLogsSnapshot({ ok: false, logs: [] });
      }

      setHasError(!monitoringPayload || !logsPayload);
      setIsLoading(false);
      hasLoadedRef.current = true;
      inFlight = false;
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, REFRESH_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const currentTab = activeTab ?? "Positions";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const pageSyncState: LiveDataSyncState = isLoading ? "loading" : hasError ? "error" : "ready";
  const rightBadges = useMemo(
    () => buildNavbarBadges(liveTradingStatus, pageSyncState),
    [liveTradingStatus, pageSyncState]
  );

  const positionRows = useMemo<PositionRowProps[]>(() => {
    const rows = (monitoringSnapshot?.positions ?? []).map(toPositionRow);
    return rows.sort((left, right) => left.symbol.localeCompare(right.symbol));
  }, [monitoringSnapshot?.positions]);

  const summary = useMemo<PositionSummaryProps>(() => {
    const source = monitoringSnapshot?.summary;
    if (source) {
      return {
        totalShares: parseNumber(source.totalShares) ?? 0,
        totalOpenPL: parseNumber(source.totalOpenPL) ?? 0,
        avgDaysHeld: parseNumber(source.avgDaysHeld) ?? 0,
        totalCapturedPL: parseNumber(source.totalCapturedPL) ?? 0,
      };
    }
    return summaryFromRows(positionRows);
  }, [monitoringSnapshot?.summary, positionRows]);

  const logs = useMemo<MonitoringLogItem[]>(() => {
    const rows = logsSnapshot?.logs ?? [];
    return rows.map((row) => ({
      timestamp: String(row.timestamp ?? "--"),
      type: normalizeLogType(row.type),
      message: String(row.message ?? ""),
    }));
  }, [logsSnapshot?.logs]);

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />
      <main className="relative pb-12 pt-[calc(var(--app-nav-height,208px)+16px)]">
        <div className="pointer-events-none absolute -top-28 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-cyan-300/18 via-sky-200/10 to-transparent blur-3xl dark:from-cyan-500/16 dark:via-blue-500/12 dark:to-transparent" />
        <div className="pointer-events-none absolute left-0 top-56 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-300/18 via-amber-200/10 to-transparent blur-3xl dark:from-emerald-500/14 dark:via-amber-500/10 dark:to-transparent" />
        <div className="relative mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8">
          <PositionsTab
            positions={positionRows}
            summary={summary}
            logs={logs}
            isLoading={isLoading}
            hasError={hasError}
          />
        </div>
      </main>
    </div>
  );
}
