import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import type { StatusTone } from "../types/ui";

type TradesOverviewProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

type MetricsSummary = {
  totalTrades: number | null;
  winRate: number | null;
  netPnl: number | null;
  profitFactor: number | null;
  lastRunUtc?: string | null;
};

type TradeRow = {
  symbol: string;
  entryTime: string;
  exitTime: string;
  qty: number | null;
  pnl: number | null;
  netPnl: number | null;
  exitReason: string;
  entryPrice: number | null;
  exitPrice: number | null;
  sortKey: number;
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
  exit_reason?: string | null;
  entry_price?: number | null;
  exit_price?: number | null;
  updated_at?: string | null;
  created_at?: string | null;
};

type TradesOverviewResponse = {
  ok?: boolean;
  metrics?: TradesMetrics;
  trades?: TradeRecord[];
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

const formatDateTime = (value: string | null | undefined) => {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

const normalizeMetrics = (metrics: TradesMetrics | null | undefined): MetricsSummary | null => {
  if (!metrics) {
    return null;
  }
  return {
    totalTrades: parseNumber(metrics.total_trades),
    winRate: parseNumber(metrics.win_rate),
    netPnl: parseNumber(metrics.net_pnl),
    profitFactor: parseNumber(metrics.profit_factor),
    lastRunUtc: metrics.last_run_utc ?? null,
  };
};

const parseTrades = (records: TradeRecord[] | null | undefined): TradeRow[] => {
  if (!records) {
    return [];
  }
  return records.map((record) => {
    const entryTime = record.entry_time ?? "";
    const exitTime = record.exit_time ?? "";
    const updatedTime = record.updated_at ?? record.created_at ?? "";
    const sortKey = Date.parse(exitTime || entryTime || updatedTime || "") || 0;
    return {
      symbol: record.symbol ?? "--",
      entryTime,
      exitTime,
      qty: parseNumber(record.qty ?? null),
      pnl: parseNumber(record.realized_pnl ?? null),
      netPnl: parseNumber(record.realized_pnl ?? null),
      exitReason: record.exit_reason ?? "n/a",
      entryPrice: parseNumber(record.entry_price ?? null),
      exitPrice: parseNumber(record.exit_price ?? null),
      sortKey,
    };
  });
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

export default function TradesOverview({ activeTab, onTabSelect }: TradesOverviewProps) {
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const payload = await fetchJson<TradesOverviewResponse>("/api/trades/overview");
      if (!isMounted) {
        return;
      }
      if (!payload) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      setMetrics(normalizeMetrics(payload.metrics));
      setTrades(parseTrades(payload.trades));
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "Trades";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "success" as const, showDot: true },
  ];

  const tradeStats = useMemo(() => {
    if (!trades.length) {
      return {
        total: 0,
        wins: 0,
        losses: 0,
        netPnl: 0,
        avgPnl: 0,
        profitFactor: null as number | null,
        lastUpdated: "n/a",
      };
    }
    let wins = 0;
    let losses = 0;
    let netPnl = 0;
    let positive = 0;
    let negative = 0;
    trades.forEach((trade) => {
      const pnl = trade.netPnl ?? trade.pnl ?? 0;
      netPnl += pnl;
      if (pnl > 0) {
        wins += 1;
        positive += pnl;
      } else if (pnl < 0) {
        losses += 1;
        negative += Math.abs(pnl);
      }
    });
    const avgPnl = trades.length ? netPnl / trades.length : 0;
    const profitFactor = negative > 0 ? positive / negative : null;
    const lastTrade = [...trades].sort((a, b) => b.sortKey - a.sortKey)[0];
    return {
      total: trades.length,
      wins,
      losses,
      netPnl,
      avgPnl,
      profitFactor,
      lastUpdated: formatDateTime(lastTrade?.exitTime || lastTrade?.entryTime),
    };
  }, [trades]);

  const summaryMetrics = useMemo(() => {
    const total = metrics?.totalTrades ?? tradeStats.total;
    const winRateRaw = metrics?.winRate ?? (tradeStats.total ? (tradeStats.wins / tradeStats.total) * 100 : null);
    const winRate = winRateRaw !== null && winRateRaw <= 1 ? winRateRaw * 100 : winRateRaw;
    const netPnl = metrics?.netPnl ?? tradeStats.netPnl;
    const profitFactor = metrics?.profitFactor ?? tradeStats.profitFactor;
    return { total, winRate, netPnl, profitFactor };
  }, [metrics, tradeStats]);

  const recentTrades = useMemo(() => {
    return [...trades].sort((a, b) => b.sortKey - a.sortKey).slice(0, 12);
  }, [trades]);

  const exitReasonCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    trades.forEach((trade) => {
      const reasons = trade.exitReason ? trade.exitReason.split(";") : ["n/a"];
      reasons.forEach((reason) => {
        const trimmed = reason.trim();
        if (!trimmed) {
          return;
        }
        counts[trimmed] = (counts[trimmed] ?? 0) + 1;
      });
    });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);
  }, [trades]);

  const resultState = useMemo(() => {
    if (isLoading) {
      return { label: "Loading", tone: "info" as StatusTone };
    }
    if (!trades.length) {
      return { label: "No Trades", tone: "neutral" as StatusTone };
    }
    if (tradeStats.wins > 0 && tradeStats.losses > 0) {
      return { label: "Mixed Results", tone: "warning" as StatusTone };
    }
    if (tradeStats.wins > 0 && tradeStats.losses === 0) {
      return { label: "All Wins", tone: "success" as StatusTone };
    }
    return { label: "Losses", tone: "error" as StatusTone };
  }, [isLoading, trades.length, tradeStats.losses, tradeStats.wins]);

  const kpiCards = [
    {
      title: "Total Trades",
      value: formatNumber(summaryMetrics.total),
      detail: `Last updated ${tradeStats.lastUpdated}`,
      detailTone: "neutral" as StatusTone,
      footnote: "Closed trades only",
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-sky-600" fill="none">
          <path
            d="M4 7h16v10H4z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M8 11h4"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Win Rate",
      value: formatPercent(summaryMetrics.winRate),
      detail: `Wins: ${tradeStats.wins} / Losses: ${tradeStats.losses}`,
      detailTone:
        summaryMetrics.winRate !== null && summaryMetrics.winRate >= 55
          ? ("success" as StatusTone)
          : ("warning" as StatusTone),
      footnote: "Rolling trade history",
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
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
    },
    {
      title: "Net P/L",
      value: formatSignedCurrency(summaryMetrics.netPnl),
      detail: `Avg trade: ${formatSignedCurrency(tradeStats.avgPnl)}`,
      detailTone:
        summaryMetrics.netPnl !== null && summaryMetrics.netPnl > 0
          ? ("success" as StatusTone)
          : ("error" as StatusTone),
      footnote: "After fees",
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-violet-600" fill="none">
          <path
            d="M5 12h3v7H5zM10.5 5h3v14h-3zM16 9h3v10h-3z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Profit Factor",
      value: summaryMetrics.profitFactor ? summaryMetrics.profitFactor.toFixed(2) : "--",
      detail: summaryMetrics.profitFactor && summaryMetrics.profitFactor >= 1.2 ? "Healthy" : "Monitor",
      detailTone:
        summaryMetrics.profitFactor !== null && summaryMetrics.profitFactor >= 1.2
          ? ("success" as StatusTone)
          : ("warning" as StatusTone),
      footnote: "Gross wins / losses",
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-orange-600" fill="none">
          <path
            d="M12 3v18M8 7h6a3 3 0 0 1 0 6H10a3 3 0 0 0 0 6h6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-32 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Trades Overview</h1>
            <p className="mt-2 text-sm text-slate-500">Trade history and performance results</p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Trade history could not be loaded. Showing placeholders.
            </div>
          ) : null}

          <section className="mt-8">
            {isLoading ? (
              <div className="grid grid-cols-4 gap-6">
                {Array.from({ length: 4 }).map((_, index) => (
                  <div key={index} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="h-4 w-24 rounded-full bg-slate-100" />
                    <div className="mt-6 h-7 w-32 rounded-full bg-slate-200" />
                    <div className="mt-3 h-3 w-28 rounded-full bg-slate-100" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-4 gap-6">
                {kpiCards.map((card) => (
                  <KPICard
                    key={card.title}
                    title={card.title}
                    value={card.value}
                    detail={card.detail}
                    detailTone={card.detailTone}
                    footnote={card.footnote}
                    icon={card.icon}
                  />
                ))}
              </div>
            )}
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Trade History</h2>
              <StatusBadge label={resultState.label} tone={resultState.tone} size="sm" />
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="grid grid-cols-[1.2fr_1fr_1fr_0.7fr_0.8fr_1.6fr_0.6fr] gap-3 border-b border-slate-100 px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <div>Symbol</div>
                <div>Entry</div>
                <div>Exit</div>
                <div>Qty</div>
                <div>P/L</div>
                <div>Exit Reason</div>
                <div>Status</div>
              </div>
              <div className="max-h-72 overflow-y-auto px-5 py-2">
                {recentTrades.length ? (
                  recentTrades.map((trade, index) => {
                    const pnlValue = trade.netPnl ?? trade.pnl ?? 0;
                    const pnlTone = pnlValue > 0 ? "text-emerald-600" : pnlValue < 0 ? "text-rose-600" : "text-slate-500";
                    const statusLabel = pnlValue > 0 ? "Win" : pnlValue < 0 ? "Loss" : "Flat";
                    const statusTone: StatusTone = pnlValue > 0 ? "success" : pnlValue < 0 ? "error" : "neutral";
                    return (
                      <div
                        key={`${trade.symbol}-${index}`}
                        className="grid grid-cols-[1.2fr_1fr_1fr_0.7fr_0.8fr_1.6fr_0.6fr] items-center gap-3 border-b border-slate-100 py-3 text-sm text-slate-700"
                      >
                        <div className="font-semibold text-slate-900 dark:text-slate-100">{trade.symbol}</div>
                        <div>{formatDateTime(trade.entryTime)}</div>
                        <div>{formatDateTime(trade.exitTime)}</div>
                        <div>{formatNumber(trade.qty)}</div>
                        <div className={`font-semibold ${pnlTone}`}>{formatSignedCurrency(pnlValue)}</div>
                        <div className="text-xs text-slate-500">{trade.exitReason}</div>
                        <div>
                          <StatusBadge label={statusLabel} tone={statusTone} size="sm" />
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="py-6 text-sm text-slate-500">No trades recorded yet.</div>
                )}
              </div>
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Results Summary</h2>
              <span className="text-xs text-slate-500">Derived from trade history</span>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-6">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="text-xs uppercase tracking-wide text-slate-400">Net Performance</div>
                <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                  {formatSignedCurrency(summaryMetrics.netPnl)}
                </div>
                <div className="mt-2 text-xs text-slate-500">
                  Average per trade {formatSignedCurrency(tradeStats.avgPnl)}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="text-xs uppercase tracking-wide text-slate-400">Win / Loss Split</div>
                <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                  {tradeStats.wins} / {tradeStats.losses}
                </div>
                <div className="mt-2 text-xs text-slate-500">Win rate {formatPercent(summaryMetrics.winRate)}</div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="text-xs uppercase tracking-wide text-slate-400">Top Exit Reasons</div>
                <div className="mt-3 space-y-2 text-sm text-slate-700">
                  {exitReasonCounts.length ? (
                    exitReasonCounts.map(([reason, count]) => (
                      <div key={reason} className="flex items-center justify-between">
                        <span className="text-xs text-slate-500">{reason}</span>
                        <span className="font-semibold text-slate-900 dark:text-slate-100">{count}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-xs text-slate-500">No exit data yet.</div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}


