import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import LogViewer from "../components/panels/LogViewer";
import type { LogEntry, StatusTone, SystemStatusItem } from "../types/ui";

type HealthOverviewResponse = {
  ok?: boolean;
  metrics_summary_present?: boolean;
  trades_log_present?: boolean;
  trades_log_rows?: number | null;
  kpis?: Record<string, number | string | null>;
};

type ApiHealthResponse = {
  trading_ok?: boolean | null;
  trading_status?: number | null;
  data_ok?: boolean | null;
  data_status?: number | null;
  feed?: string | null;
  last_run_utc?: string | null;
  pipeline_rc?: number | null;
  rows_final?: number | null;
  rows_premetrics?: number | null;
  latest_source?: string | null;
  freshness?: {
    age_seconds?: number | null;
    freshness_level?: string | null;
  };
  run_type?: string | null;
  buying_power?: number | null;
};

type AccountOverviewResponse = {
  ok?: boolean;
  snapshot?: AccountSnapshot;
};

type AccountSnapshot = {
  account_id?: string;
  status?: string;
  equity?: number | null;
  cash?: number | null;
  buying_power?: number | null;
  portfolio_value?: number | null;
  taken_at?: string;
  source?: string;
};

type TradesMetrics = {
  total_trades?: number | null;
  win_rate?: number | null;
  net_pnl?: number | null;
  profit_factor?: number | null;
  last_run_utc?: string | null;
};

type TradeRecord = {
  trade_id?: string | number;
  symbol?: string;
  qty?: number | null;
  status?: string;
  entry_time?: string | null;
  exit_time?: string | null;
  realized_pnl?: number | null;
  updated_at?: string | null;
  created_at?: string | null;
};

type TradesOverviewResponse = {
  ok?: boolean;
  metrics?: TradesMetrics;
  trades?: TradeRecord[];
  open_positions?: {
    count?: number | null;
    realized_pnl?: number | null;
  };
};

type OpenPositionsSummary = {
  count: number | null;
  pnl: number | null;
  pnlPct: number | null;
  filledCount: number | null;
  totalCount: number;
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

const statusDotTone: Record<StatusTone, string> = {
  success: "bg-emerald-500",
  warning: "bg-amber-500",
  error: "bg-rose-500",
  info: "bg-sky-500",
  neutral: "bg-slate-400",
};

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

const parseNumber = (value: string | number | null | undefined) => {
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

const formatCurrency = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return currencyFormatter.format(value);
};

const formatSignedCurrency = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${currencyFormatter.format(value)}`;
};

const formatNumber = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return numberFormatter.format(value);
};

const formatPercent = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${percentFormatter.format(value)}%`;
};

const formatSignedPercent = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${percentFormatter.format(value)}%`;
};

const formatDateTime = (value: string | null | undefined) => {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "n/a";
  }
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatAge = (seconds: number | null | undefined) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "n/a";
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s ago`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h ago`;
  }
  return `${Math.round(seconds / 86400)}d ago`;
};

const normalizeWinRate = (value: number | null) => {
  if (value === null || value === undefined) {
    return null;
  }
  return value <= 1 ? value * 100 : value;
};

const isFilledStatus = (status: string | undefined) => {
  if (!status) {
    return false;
  }
  const normalized = status.toLowerCase();
  return normalized.includes("fill") || normalized === "closed" || normalized === "filled";
};

const isOpenStatus = (status: string | undefined) => {
  if (!status) {
    return false;
  }
  return status.toLowerCase() === "open";
};

const fetchJson = async <T,>(path: string): Promise<T | null> => {
  try {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
};

const fetchText = async (path: string): Promise<string | null> => {
  try {
    const response = await fetch(path, {
      headers: { Accept: "text/plain" },
    });
    if (!response.ok) {
      return null;
    }
    return await response.text();
  } catch {
    return null;
  }
};

type ParsedLogEntry = LogEntry & { timestampMs: number };

const normalizeLogLevel = (rawLevel: string | undefined): LogEntry["level"] => {
  if (!rawLevel) {
    return "INFO";
  }
  const level = rawLevel.toUpperCase();
  if (level.startsWith("WARN")) {
    return "WARN";
  }
  if (level === "ERROR") {
    return "ERROR";
  }
  if (level === "SUCCESS") {
    return "SUCCESS";
  }
  return "INFO";
};

const buildLogEntries = (sources: Array<{ text: string | null; source: string }>, limit = 8) => {
  const entries: ParsedLogEntry[] = [];
  const pattern =
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?(?:\s+-\s+[^-]+\s+-\s+)?\s*(?:\[(\w+)\])?\s*(.*)$/;

  sources.forEach(({ text }) => {
    if (!text) {
      return;
    }
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const sample = lines.slice(-Math.max(limit, 12));
    sample.forEach((line) => {
      const match = line.match(pattern);
      if (match) {
        const [, date, time, level, message] = match;
        const timestampMs = new Date(`${date}T${time}`).getTime() || 0;
        entries.push({
          time,
          level: normalizeLogLevel(level),
          message: message.trim() || "(no message)",
          timestampMs,
        });
      } else {
        entries.push({
          time: "--:--:--",
          level: "INFO",
          message: line.trim(),
          timestampMs: 0,
        });
      }
    });
  });

  entries.sort((a, b) => b.timestampMs - a.timestampMs);
  return entries.slice(0, limit).map(({ timestampMs, ...entry }) => entry);
};

const computePipelineScore = (health: ApiHealthResponse | null) => {
  if (!health) {
    return null;
  }
  const checks = [
    typeof health.pipeline_rc === "number" ? health.pipeline_rc === 0 : null,
    typeof health.trading_ok === "boolean" ? health.trading_ok : null,
    typeof health.data_ok === "boolean" ? health.data_ok : null,
  ].filter((value): value is boolean => typeof value === "boolean");

  if (!checks.length) {
    return null;
  }
  const passed = checks.filter(Boolean).length;
  const score = (passed / checks.length) * 100;
  return Math.round(score * 10) / 10;
};

const pipelineStatusFromScore = (score: number | null) => {
  if (score === null) {
    return { label: "Unknown", tone: "neutral" as StatusTone };
  }
  if (score >= 90) {
    return { label: "Healthy", tone: "success" as StatusTone };
  }
  if (score >= 70) {
    return { label: "Watch", tone: "warning" as StatusTone };
  }
  return { label: "Degraded", tone: "error" as StatusTone };
};

type DashboardHealthProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

export default function DashboardHealth({ activeTab, onTabSelect }: DashboardHealthProps) {
  const [healthSnapshot, setHealthSnapshot] = useState<ApiHealthResponse | null>(null);
  const [overviewSnapshot, setOverviewSnapshot] = useState<HealthOverviewResponse | null>(null);
  const [accountSnapshot, setAccountSnapshot] = useState<AccountSnapshot | null>(null);
  const [tradesOverview, setTradesOverview] = useState<TradesOverviewResponse | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      const [overview, health, accountPayload, tradesPayload, pipelineLog, executeLog] =
        await Promise.all([
          fetchJson<HealthOverviewResponse>("/health/overview"),
          fetchJson<ApiHealthResponse>("/api/health"),
          fetchJson<AccountOverviewResponse>("/api/account/overview"),
          fetchJson<TradesOverviewResponse>("/api/trades/overview"),
          fetchText("/logs/pipeline.log"),
          fetchText("/logs/execute_trades.log"),
        ]);

      if (!isMounted) {
        return;
      }

      setOverviewSnapshot(overview);
      setHealthSnapshot(health);
      setAccountSnapshot(accountPayload?.snapshot ?? null);
      setTradesOverview(tradesPayload);
      setLogEntries(
        buildLogEntries([
          { source: "pipeline", text: pipelineLog },
          { source: "execute", text: executeLog },
        ])
      );
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const rightBadges = useMemo(() => {
    const liveTone: StatusTone =
      healthSnapshot?.trading_ok === true
        ? "success"
        : healthSnapshot?.trading_ok === false
          ? "error"
          : "neutral";
    return [
      { label: "Paper Trading", tone: "warning" as const },
      {
        label: "Live",
        tone: liveTone as StatusTone,
        showDot: healthSnapshot?.trading_ok === true,
      },
    ];
  }, [healthSnapshot?.trading_ok]);

  const pipelineScore = useMemo(
    () => computePipelineScore(healthSnapshot),
    [healthSnapshot]
  );

  const pipelineStatus = useMemo(
    () => pipelineStatusFromScore(pipelineScore),
    [pipelineScore]
  );

  const pipelineFootnote = useMemo(() => {
    const lastRun = formatDateTime(healthSnapshot?.last_run_utc);
    const source = healthSnapshot?.latest_source
      ? `Source: ${healthSnapshot.latest_source}`
      : null;
    return source ? `Last run: ${lastRun} / ${source}` : `Last run: ${lastRun}`;
  }, [healthSnapshot?.last_run_utc, healthSnapshot?.latest_source]);

  const tradesList = useMemo(() => tradesOverview?.trades ?? [], [tradesOverview]);

  const tradeStats = useMemo(() => {
    if (!tradesList.length) {
      return {
        wins: 0,
        losses: 0,
        netPnl: null,
        winRate: null,
        filledCount: null,
      };
    }
    let wins = 0;
    let losses = 0;
    let netPnl = 0;
    let filledCount = 0;
    tradesList.forEach((trade) => {
      const pnl = parseNumber(trade.realized_pnl ?? null);
      if (pnl === null) {
        return;
      }
      netPnl += pnl;
      if (pnl > 0) {
        wins += 1;
      } else if (pnl < 0) {
        losses += 1;
      }
      if (isFilledStatus(trade.status)) {
        filledCount += 1;
      }
    });
    const total = wins + losses;
    const winRate = total > 0 ? (wins / total) * 100 : null;
    return { wins, losses, netPnl, winRate, filledCount };
  }, [tradesList]);

  const openPositions = useMemo<OpenPositionsSummary>(() => {
    const openCountFromApi = tradesOverview?.open_positions?.count;
    const openCountFallback = tradesList.length
      ? tradesList.filter((trade) => isOpenStatus(trade.status)).length
      : null;
    const count = typeof openCountFromApi === "number" ? openCountFromApi : openCountFallback;
    const pnl = parseNumber(tradesOverview?.open_positions?.realized_pnl ?? null);
    const filledCount = tradesList.length ? tradeStats.filledCount : null;
    return {
      count,
      pnl,
      pnlPct: null,
      filledCount,
      totalCount: tradesList.length,
    };
  }, [tradesOverview, tradesList, tradeStats.filledCount]);

  const tradesMetrics = useMemo(() => {
    const kpis = tradesOverview?.metrics ?? overviewSnapshot?.kpis ?? {};
    const totalTrades =
      parseNumber(kpis.total_trades) ??
      (tradesList.length ? tradesList.length : null) ??
      parseNumber(overviewSnapshot?.trades_log_rows ?? null);
    const winRateRaw = parseNumber(kpis.win_rate) ?? tradeStats.winRate ?? null;
    const winRate = normalizeWinRate(winRateRaw);
    const netPnl = parseNumber(kpis.net_pnl) ?? tradeStats.netPnl ?? null;
    return { totalTrades, winRate, netPnl };
  }, [overviewSnapshot, tradeStats.netPnl, tradeStats.winRate, tradesList.length, tradesOverview]);

  const buyingPowerValue = parseNumber(
    healthSnapshot?.buying_power ?? accountSnapshot?.buying_power ?? accountSnapshot?.equity ?? null
  );

  const openPositionsDetail = useMemo(() => {
    if (!openPositions || openPositions.pnl === null) {
      return "Open P/L: --";
    }
    const pctDetail =
      openPositions.pnlPct !== null
        ? ` (${formatSignedPercent(openPositions.pnlPct)})`
        : "";
    return `Open P/L: ${formatSignedCurrency(openPositions.pnl)}${pctDetail}`;
  }, [openPositions]);

  const openPositionsTone: StatusTone = useMemo(() => {
    if (!openPositions || openPositions.pnl === null) {
      return "neutral";
    }
    return openPositions.pnl < 0 ? "error" : "success";
  }, [openPositions]);

  const kpiCards = useMemo(
    () => [
      {
        title: "Pipeline Health",
        value: formatPercent(pipelineScore),
        detail: pipelineStatus.label,
        detailTone: pipelineStatus.tone,
        footnote: pipelineFootnote,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
            <path
              d="M4 12h3l2-5 4 10 2-5h5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.1s_ease-out_1]",
      },
      {
        title: "Buying Power",
        value: formatCurrency(buyingPowerValue),
        detail: `Cash: ${formatCurrency(parseNumber(accountSnapshot?.cash ?? null))}`,
        detailTone: "info" as const,
        footnote: `Equity: ${formatCurrency(parseNumber(accountSnapshot?.equity ?? null))}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-sky-600" fill="none">
            <path
              d="M12 3v18M8 7h6a3 3 0 0 1 0 6H10a3 3 0 0 0 0 6h6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.25s_ease-out_1]",
      },
      {
        title: "Open Positions",
        value: formatNumber(openPositions?.count ?? null),
        detail: openPositionsDetail,
        detailTone: openPositionsTone,
        footnote: `Filled orders: ${formatNumber(openPositions?.filledCount ?? null)}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-violet-600" fill="none">
            <path
              d="M4 16l5-5 4 4 7-7"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M16 8h4v4"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.4s_ease-out_1]",
      },
      {
        title: "Recent Trades Summary",
        value: formatNumber(tradesMetrics.totalTrades),
        detail: `Win rate: ${formatPercent(tradesMetrics.winRate)}`,
        detailTone: "neutral" as const,
        footnote: `Net P/L: ${formatSignedCurrency(tradesMetrics.netPnl)}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-orange-600" fill="none">
            <path
              d="M5 12h3v7H5zM10.5 5h3v14h-3zM16 9h3v10h-3z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.55s_ease-out_1]",
      },
    ],
    [
      accountSnapshot?.cash,
      accountSnapshot?.equity,
      buyingPowerValue,
      openPositions,
      openPositionsDetail,
      openPositionsTone,
      pipelineFootnote,
      pipelineScore,
      pipelineStatus.label,
      pipelineStatus.tone,
      tradesMetrics.netPnl,
      tradesMetrics.totalTrades,
      tradesMetrics.winRate,
    ]
  );

  const systemStatus: SystemStatusItem[] = useMemo(() => {
    const tradingOk = healthSnapshot?.trading_ok ?? null;
    const dataOk = healthSnapshot?.data_ok ?? null;
    const pipelineRc = healthSnapshot?.pipeline_rc ?? null;
    const feed = healthSnapshot?.feed ? healthSnapshot.feed.toUpperCase() : null;

    const apiTone: StatusTone = tradingOk === false ? "error" : tradingOk === true ? "success" : "neutral";
    const dataTone: StatusTone = dataOk === false ? "warning" : dataOk === true ? "success" : "neutral";
    const pipelineTone: StatusTone =
      pipelineRc === 0 ? "success" : pipelineRc === null ? "neutral" : "warning";

    return [
      {
        title: "API Connection",
        status: tradingOk === true ? "Connected" : tradingOk === false ? "Offline" : "Unknown",
        tone: apiTone,
        description: tradingOk === true ? "Broker API responding" : "Broker API status unknown",
        meta: `Status: ${healthSnapshot?.trading_status ?? "n/a"}`,
      },
      {
        title: "Market Data Feed",
        status: dataOk === true ? "Streaming" : dataOk === false ? "Delayed" : "Unknown",
        tone: dataTone,
        description: feed ? `Feed: ${feed}` : "Market data feed",
        meta: `Status: ${healthSnapshot?.data_status ?? "n/a"}`,
      },
      {
        title: "ML Pipeline",
        status: pipelineRc === 0 ? "Ready" : pipelineRc === null ? "Unknown" : "Degraded",
        tone: pipelineTone,
        description: healthSnapshot?.run_type
          ? `Run type: ${healthSnapshot.run_type}`
          : "Run type: n/a",
        meta: `Rows: ${formatNumber(healthSnapshot?.rows_final ?? null)}`,
      },
    ];
  }, [
    healthSnapshot?.data_ok,
    healthSnapshot?.data_status,
    healthSnapshot?.feed,
    healthSnapshot?.pipeline_rc,
    healthSnapshot?.rows_final,
    healthSnapshot?.run_type,
    healthSnapshot?.trading_ok,
    healthSnapshot?.trading_status,
  ]);

  const updatedLabel = useMemo(() => {
    if (healthSnapshot?.freshness?.age_seconds !== undefined) {
      return `Updated ${formatAge(healthSnapshot.freshness.age_seconds)}`;
    }
    return "Updated n/a";
  }, [healthSnapshot?.freshness?.age_seconds]);

  const emptyEntries: LogEntry[] = [
    { time: "--:--:--", level: "INFO", message: "No recent activity logged." },
  ];

  const displayEntries = logEntries.length ? logEntries : emptyEntries;

  const currentTab = activeTab ?? "Dashboard";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-24 pb-12">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl" />

        <div className="relative mx-auto max-w-[1240px] px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900">Dashboard Overview</h1>
            <p className="mt-2 text-sm text-slate-500">Real-time system health and portfolio metrics</p>
          </header>

          <section className="mt-8 grid grid-cols-4 gap-6">
            {kpiCards.map((card) => (
              <div key={card.title} className={card.animation}>
                <KPICard
                  title={card.title}
                  value={card.value}
                  detail={card.detail}
                  detailTone={card.detailTone}
                  footnote={card.footnote}
                  icon={card.icon}
                />
              </div>
            ))}
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800">System Status</h2>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span>{updatedLabel}</span>
                <StatusBadge label="Stable" tone="success" size="sm" />
              </div>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="grid grid-cols-3 gap-6">
                {systemStatus.map((item) => (
                  <div key={item.title} className="flex items-start gap-3">
                    <span className={`mt-2 h-2 w-2 rounded-full ${statusDotTone[item.tone]}`} />
                    <div>
                      <div className="flex items-center gap-2">
                        <div className="text-sm font-semibold text-slate-800">{item.title}</div>
                        <StatusBadge label={item.status} tone={item.tone} size="sm" />
                      </div>
                      <div className="mt-1 text-xs text-slate-500">{item.description}</div>
                      <div className="mt-1 text-xs text-slate-400">{item.meta}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800">Activity Log</h2>
              <span className="text-sm font-semibold text-blue-600">View All</span>
            </div>
            <div className="mt-4">
              <LogViewer
                title="Live Activity Feed"
                statusLabel="Streaming"
                entries={displayEntries}
                actionLabel="Clear"
              />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
