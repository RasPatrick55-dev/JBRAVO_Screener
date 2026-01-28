import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import type { StatusTone } from "../types/ui";

type PositionsMonitoringProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
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

type AccountOverviewResponse = {
  ok?: boolean;
  snapshot?: AccountSnapshot;
};

type TradeRecord = {
  trade_id?: string | number;
  symbol?: string;
  qty?: number | null;
  status?: string;
  entry_time?: string | null;
  entry_price?: number | null;
  exit_time?: string | null;
  exit_price?: number | null;
  realized_pnl?: number | null;
  exit_reason?: string | null;
  updated_at?: string | null;
  created_at?: string | null;
};

type TradesOverviewResponse = {
  ok?: boolean;
  trades?: TradeRecord[];
};

type PositionRow = {
  symbol: string;
  entryTime: string;
  qty: number | null;
  entryPrice: number | null;
  costBasis: number | null;
  status: string;
  realizedPnl: number | null;
  exitPlan: string;
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

const formatDate = (value: string | null | undefined) => {
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

export default function PositionsMonitoring({ activeTab, onTabSelect }: PositionsMonitoringProps) {
  const [positions, setPositions] = useState<PositionRow[]>([]);
  const [accountSnapshot, setAccountSnapshot] = useState<AccountSnapshot | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const [tradesPayload, accountPayload] = await Promise.all([
        fetchJson<TradesOverviewResponse>("/api/trades/overview"),
        fetchJson<AccountOverviewResponse>("/api/account/overview"),
      ]);
      if (!isMounted) {
        return;
      }
      if (!tradesPayload || !accountPayload) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      const openPositions = (tradesPayload.trades ?? [])
        .filter((trade) => isOpenStatus(trade.status))
        .map((trade) => {
          const qty = parseNumber(trade.qty ?? null);
          const entryPrice = parseNumber(trade.entry_price ?? null);
          const costBasis =
            qty !== null && entryPrice !== null ? qty * entryPrice : null;
          return {
            symbol: trade.symbol ?? "--",
            entryTime: trade.entry_time ?? "",
            qty,
            entryPrice,
            costBasis,
            status: trade.status ?? "OPEN",
            realizedPnl: parseNumber(trade.realized_pnl ?? null),
            exitPlan: "Trailing stop 3%",
          };
        });
      setPositions(openPositions);
      setAccountSnapshot(accountPayload.snapshot ?? null);
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "Positions";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  const exposureSummary = useMemo(() => {
    if (!positions.length) {
      return { total: null, avg: null, largest: null, largestSymbol: "--", largestPct: null };
    }
    let total = 0;
    let largest = 0;
    let largestSymbol = "--";
    positions.forEach((position) => {
      const costBasis = position.costBasis ?? 0;
      total += costBasis;
      if (costBasis > largest) {
        largest = costBasis;
        largestSymbol = position.symbol;
      }
    });
    const avg = positions.length ? total / positions.length : null;
    const largestPct = total > 0 ? (largest / total) * 100 : null;
    return { total, avg, largest, largestSymbol, largestPct };
  }, [positions]);

  const openPnl = useMemo(() => {
    if (!positions.length) {
      return null;
    }
    let total = 0;
    let hasValue = false;
    positions.forEach((position) => {
      const pnl = position.realizedPnl;
      if (pnl !== null) {
        total += pnl;
        hasValue = true;
      }
    });
    return hasValue ? total : null;
  }, [positions]);

  const maxPositions = 5;
  const concentrationLimit = 20;
  const openCount = positions.length;
  const buyingPower = parseNumber(accountSnapshot?.buying_power ?? null);
  const lastUpdated = formatDate(accountSnapshot?.taken_at);
  const emptyState = !isLoading && !hasError && positions.length === 0;

  const kpiCards = [
    {
      title: "Open Positions",
      value: formatNumber(openCount),
      detail: `Max allowed: ${maxPositions}`,
      detailTone: openCount < maxPositions ? ("success" as StatusTone) : ("warning" as StatusTone),
      footnote: `Updated ${lastUpdated}`,
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
      title: "Net Exposure",
      value: formatCurrency(exposureSummary.total),
      detail: `Avg size: ${formatCurrency(exposureSummary.avg)}`,
      detailTone: "neutral" as StatusTone,
      footnote: "Cost basis only",
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
      title: "Largest Position",
      value: exposureSummary.largestSymbol,
      detail: formatCurrency(exposureSummary.largest),
      detailTone: "neutral" as StatusTone,
      footnote: `Concentration: ${formatPercent(exposureSummary.largestPct)}`,
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-violet-600" fill="none">
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
    {
      title: "Buying Power",
      value: formatCurrency(buyingPower),
      detail: buyingPower !== null && buyingPower > 0 ? "Available" : "Restricted",
      detailTone:
        buyingPower !== null && buyingPower > 0 ? ("success" as StatusTone) : ("warning" as StatusTone),
      footnote: "Broker snapshot",
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
    },
  ];

  const riskChecks = [
    {
      label: "Max Positions",
      value: `${openCount} / ${maxPositions}`,
      status: openCount < maxPositions ? "Within limit" : "At limit",
      tone: openCount < maxPositions ? ("success" as StatusTone) : ("warning" as StatusTone),
    },
    {
      label: "Concentration Limit",
      value: `${formatPercent(exposureSummary.largestPct)} of exposure`,
      status:
        exposureSummary.largestPct !== null && exposureSummary.largestPct <= concentrationLimit
          ? "Healthy"
          : "Review",
      tone:
        exposureSummary.largestPct !== null && exposureSummary.largestPct <= concentrationLimit
          ? ("success" as StatusTone)
          : ("warning" as StatusTone),
    },
    {
      label: "Open P/L",
      value: formatSignedCurrency(openPnl),
      status: openPnl !== null && openPnl >= 0 ? "Stable" : "Watch",
      tone: openPnl !== null && openPnl >= 0 ? ("success" as StatusTone) : ("warning" as StatusTone),
    },
    {
      label: "Risk Budget",
      value: "8% per trade",
      status: "Configured",
      tone: "info" as StatusTone,
    },
  ];

  const positionAlerts = [
    {
      title: "Exposure Check",
      message: openCount === 0 ? "No open positions yet." : "Open positions within configured limits.",
      tone: openCount === 0 ? ("neutral" as StatusTone) : ("success" as StatusTone),
    },
    {
      title: "Concentration Watch",
      message:
        exposureSummary.largestPct !== null && exposureSummary.largestPct > concentrationLimit
          ? "Largest position exceeds concentration threshold."
          : "No concentration breaches detected.",
      tone:
        exposureSummary.largestPct !== null && exposureSummary.largestPct > concentrationLimit
          ? ("warning" as StatusTone)
          : ("success" as StatusTone),
    },
    {
      title: "Broker Sync",
      message: accountSnapshot?.source
        ? `Data source: ${accountSnapshot.source}`
        : "Awaiting broker snapshot.",
      tone: accountSnapshot?.source ? ("info" as StatusTone) : ("neutral" as StatusTone),
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
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Positions Monitoring</h1>
            <p className="mt-2 text-sm text-slate-500">
              Live exposure, open positions, and risk posture.
            </p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Positions data could not be loaded. Showing placeholders.
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
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Open Positions</h2>
              <span className="text-xs text-slate-500">Read-only trade ledger</span>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="grid grid-cols-[1fr_1fr_0.7fr_0.8fr_0.9fr_1fr_0.8fr_0.9fr] gap-3 border-b border-slate-100 px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <div>Symbol</div>
                <div>Entry Date</div>
                <div>Qty</div>
                <div>Entry</div>
                <div>Cost Basis</div>
                <div>Status</div>
                <div>P/L</div>
                <div>Exit Plan</div>
              </div>
              <div className="max-h-80 overflow-y-auto px-5 py-2">
                {isLoading ? (
                  <div className="space-y-3 py-4">
                    {Array.from({ length: 5 }).map((_, index) => (
                      <div key={index} className="h-6 rounded-full bg-slate-100" />
                    ))}
                  </div>
                ) : positions.length ? (
                  positions.map((position, index) => {
                    const pnlTone =
                      position.realizedPnl !== null && position.realizedPnl < 0
                        ? "text-rose-600"
                        : position.realizedPnl !== null && position.realizedPnl > 0
                          ? "text-emerald-600"
                          : "text-slate-500";
                    return (
                      <div
                        key={`${position.symbol}-${index}`}
                        className="grid grid-cols-[1fr_1fr_0.7fr_0.8fr_0.9fr_1fr_0.8fr_0.9fr] items-center gap-3 border-b border-slate-100 py-3 text-sm text-slate-700"
                      >
                        <div className="font-semibold text-slate-900 dark:text-slate-100">{position.symbol}</div>
                        <div>{formatDate(position.entryTime)}</div>
                        <div>{formatNumber(position.qty)}</div>
                        <div>{formatCurrency(position.entryPrice)}</div>
                        <div>{formatCurrency(position.costBasis)}</div>
                        <div>
                          <StatusBadge label={position.status} tone="info" size="sm" />
                        </div>
                        <div className={`font-semibold ${pnlTone}`}>
                          {formatSignedCurrency(position.realizedPnl)}
                        </div>
                        <div className="text-xs text-slate-500">{position.exitPlan}</div>
                      </div>
                    );
                  })
                ) : (
                  <div className="py-6 text-sm text-slate-500">
                    {emptyState ? "No open positions yet." : "Positions unavailable."}
                  </div>
                )}
              </div>
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Risk Controls</h2>
              <span className="text-xs text-slate-500">Execution gates and limits</span>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="grid grid-cols-2 gap-4">
                {riskChecks.map((check) => (
                  <div key={check.label} className="flex items-center justify-between rounded-xl border border-slate-100 px-4 py-3">
                    <div>
                      <div className="text-xs uppercase tracking-wide text-slate-400">{check.label}</div>
                      <div className="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-200">{check.value}</div>
                    </div>
                    <StatusBadge label={check.status} tone={check.tone} size="sm" />
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="mt-10">
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Position Alerts</h2>
            <div className="mt-4 grid grid-cols-3 gap-4">
              {positionAlerts.map((alert) => (
                <div key={alert.title} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      {alert.title}
                    </div>
                    <StatusBadge label={alert.tone.toUpperCase()} tone={alert.tone} size="sm" />
                  </div>
                  <div className="mt-2 text-sm text-slate-600">{alert.message}</div>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}


