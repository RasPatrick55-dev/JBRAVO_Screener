import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import type { StatusTone } from "../types/ui";

type ExecuteTradesProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

type ExecutionSnapshot = {
  in_window?: boolean;
  buying_power?: number | null;
  open_positions?: number;
  orders_submitted?: number;
  orders_filled?: number;
  orders_rejected?: number;
  skip_counts?: Record<string, number>;
  last_execution?: string | null;
  ny_now?: string | null;
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

const formatNumber = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return numberFormatter.format(value);
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

type Constraint = {
  label: string;
  value: string;
  status: string;
  tone: StatusTone;
};

type ExecutionState = {
  title: string;
  message: string;
  tone: StatusTone;
};

export default function ExecuteTrades({ activeTab, onTabSelect }: ExecuteTradesProps) {
  const [snapshot, setSnapshot] = useState<ExecutionSnapshot | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const snap = await fetchJson<ExecutionSnapshot>("/api/execute/overview");
      if (!isMounted) {
        return;
      }
      if (!snap) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      setSnapshot(snap);
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "Execute";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  const inWindow = snapshot?.in_window ?? false;
  const buyingPower = parseNumber(snapshot?.buying_power ?? null);
  const openPositions = snapshot?.open_positions ?? 0;
  const allowedNewPositions = inWindow && (buyingPower ?? 0) > 0 ? 3 : 0;

  const readinessStatus: StatusTone = inWindow && (buyingPower ?? 0) > 0 ? "success" : "warning";
  const windowTone: StatusTone = inWindow ? "success" : "warning";
  const allowedTone: StatusTone = allowedNewPositions > 0 ? "success" : "error";

  const executionState = useMemo<ExecutionState>(() => {
    if (isLoading) {
      return { title: "Loading", message: "Fetching execution status...", tone: "info" };
    }
    if (!snapshot) {
      return { title: "Unavailable", message: "Execution snapshot not available yet.", tone: "neutral" };
    }
    if (!snapshot.last_execution) {
      return { title: "Not Run", message: "No execution attempted yet.", tone: "neutral" };
    }
    if (!snapshot.in_window) {
      return { title: "Skipped", message: "Outside trading window.", tone: "warning" };
    }
    if ((snapshot.buying_power ?? 0) <= 0) {
      return { title: "Skipped", message: "Insufficient buying power.", tone: "warning" };
    }
    if ((snapshot.orders_submitted ?? 0) === 0) {
      return { title: "Skipped", message: "No orders submitted.", tone: "warning" };
    }
    if ((snapshot.orders_filled ?? 0) > 0 && (snapshot.orders_rejected ?? 0) > 0) {
      return { title: "Partial", message: "Some orders filled, some rejected.", tone: "warning" };
    }
    return { title: "Executed", message: "Orders submitted successfully.", tone: "success" };
  }, [isLoading, snapshot]);

  const constraints = useMemo<Constraint[]>(() => {
    const maxPositions = 5;
    const currentOpen = openPositions;
    const allocation = "8% per trade";
    const trailStop = "3.0%";
    const extendedHoursEnabled = false;
    const extendedHoursValue = extendedHoursEnabled ? "Yes" : "No";

    return [
      {
        label: "Max Positions",
        value: `${maxPositions}`,
        status: currentOpen < maxPositions ? "OK" : "Limit reached",
        tone: currentOpen < maxPositions ? "success" : "warning",
      },
      {
        label: "Current Open Positions",
        value: `${currentOpen}`,
        status: currentOpen <= maxPositions ? "Within limit" : "Above limit",
        tone: currentOpen <= maxPositions ? "success" : "error",
      },
      {
        label: "Allocation per Trade",
        value: allocation,
        status: (buyingPower ?? 0) > 0 ? "Sufficient cash" : "Insufficient cash",
        tone: (buyingPower ?? 0) > 0 ? "success" : "warning",
      },
      {
        label: "Trailing Stop Percentage",
        value: trailStop,
        status: "Configured",
        tone: "info",
      },
      {
        label: "Extended Hours Enabled",
        value: extendedHoursValue,
        status: extendedHoursEnabled ? "Enabled" : "Disabled",
        tone: extendedHoursEnabled ? "success" : "neutral",
      },
    ];
  }, [buyingPower, openPositions]);

  const trailingStopStatus = (snapshot?.orders_filled ?? 0) > 0 ? "Confirmed" : "Pending";

  const timestamps = {
    lastAttempt: formatDateTime(snapshot?.last_execution ?? null),
    lastRefresh: formatDateTime(snapshot?.ny_now ?? null),
  };

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-36 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Execution Overview</h1>
            <p className="mt-2 text-sm text-slate-500">Execution readiness and order outcomes</p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Execution data could not be loaded. Showing placeholders.
            </div>
          ) : null}

          <section className="mt-8">
            <div className="grid grid-cols-4 gap-6">
              <KPICard
                title="Trading Mode"
                value="Paper"
                detail={inWindow ? "Ready" : "Paused"}
                detailTone={readinessStatus}
                footnote="Execution is disabled for live trades."
                icon={
                  <svg viewBox="0 0 24 24" className="h-5 w-5 text-amber-600" fill="none">
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
                }
              />
              <KPICard
                title="Time Window"
                value={inWindow ? "Inside" : "Outside"}
                detail={inWindow ? "Orders allowed" : "Outside trading window"}
                detailTone={windowTone}
                footnote={`Last check: ${timestamps.lastRefresh}`}
                icon={
                  <svg viewBox="0 0 24 24" className="h-5 w-5 text-sky-600" fill="none">
                    <path
                      d="M12 7v6l4 2"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
                  </svg>
                }
              />
              <KPICard
                title="Buying Power"
                value={formatCurrency(buyingPower)}
                detail={(buyingPower ?? 0) > 0 ? "Available" : "Unavailable"}
                detailTone={(buyingPower ?? 0) > 0 ? "success" : "warning"}
                footnote="Account liquidity snapshot"
                icon={
                  <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
                    <path
                      d="M12 3v18M8 7h6a3 3 0 0 1 0 6H10a3 3 0 0 0 0 6h6"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                }
              />
              <KPICard
                title="Allowed New Positions"
                value={formatNumber(allowedNewPositions)}
                detail={allowedNewPositions > 0 ? "Capacity available" : "Blocked"}
                detailTone={allowedTone}
                footnote={`Open positions: ${formatNumber(openPositions)}`}
                icon={
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
                }
              />
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Constraints & Gates</h2>
              <span className="text-xs text-slate-500">Informational only</span>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="grid grid-cols-2 gap-4">
                {constraints.map((constraint) => (
                  <div key={constraint.label} className="flex items-center justify-between rounded-xl border border-slate-100 px-4 py-3">
                    <div>
                      <div className="text-xs uppercase tracking-wide text-slate-400">{constraint.label}</div>
                      <div className="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-200">{constraint.value}</div>
                    </div>
                    <StatusBadge label={constraint.status} tone={constraint.tone} size="sm" />
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Execution Results</h2>
              <StatusBadge label={executionState.title} tone={executionState.tone} size="sm" />
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="grid grid-cols-3 gap-6">
                <div className="rounded-xl border border-slate-100 p-4">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Submitted Orders</div>
                  <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                    {formatNumber(snapshot?.orders_submitted ?? null)}
                  </div>
                  <div className="mt-2 text-xs text-slate-500">Last attempt: {timestamps.lastAttempt}</div>
                </div>
                <div className="rounded-xl border border-slate-100 p-4">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Filled Orders</div>
                  <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                    {formatNumber(snapshot?.orders_filled ?? null)}
                  </div>
                  <div className="mt-2 text-xs text-slate-500">
                    Rejected: {formatNumber(snapshot?.orders_rejected ?? null)}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-100 p-4">
                  <div className="text-xs uppercase tracking-wide text-slate-400">Trailing Stops</div>
                  <div className="mt-3 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                    {trailingStopStatus}
                  </div>
                  <div className="mt-2 text-xs text-slate-500">Configured at 3.0%</div>
                </div>
              </div>

              <div className="mt-6 rounded-xl border border-slate-100 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                {executionState.message}
              </div>

              <div className="mt-4 grid grid-cols-2 gap-4 text-xs text-slate-500">
                <div>
                  <div className="uppercase tracking-wide text-slate-400">Skip Reasons</div>
                  <div className="mt-2 text-sm text-slate-600">
                    {snapshot?.skip_counts && Object.keys(snapshot.skip_counts).length > 0
                      ? Object.entries(snapshot.skip_counts)
                          .map(([reason, count]) => `${reason}: ${count}`)
                          .join(" / ")
                      : "No skips recorded."}
                  </div>
                </div>
                <div>
                  <div className="uppercase tracking-wide text-slate-400">Last Refresh</div>
                  <div className="mt-2 text-sm text-slate-600">{timestamps.lastRefresh}</div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}


