import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import type { StatusTone } from "../types/ui";

type AccountOverviewProps = {
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

const parseCsv = (text: string | null) => {
  if (!text) {
    return [] as Array<Record<string, string>>;
  }
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    return [] as Array<Record<string, string>>;
  }
  const headers = lines[0].split(",").map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
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

type NoteTone = "info" | "warning";

const noteStyles: Record<NoteTone, { border: string; bg: string; text: string }> = {
  info: {
    border: "border-sky-200",
    bg: "bg-sky-50",
    text: "text-sky-700",
  },
  warning: {
    border: "border-amber-200",
    bg: "bg-amber-50",
    text: "text-amber-700",
  },
};

function NotePanel({
  title,
  message,
  tone,
}: {
  title: string;
  message: string;
  tone: NoteTone;
}) {
  const styles = noteStyles[tone];
  return (
    <div className={`rounded-xl border ${styles.border} ${styles.bg} p-4`}>
      <div className={`text-xs font-semibold uppercase tracking-wide ${styles.text}`}>{title}</div>
      <div className="mt-2 text-sm text-slate-700">{message}</div>
    </div>
  );
}

export default function AccountOverview({ activeTab, onTabSelect }: AccountOverviewProps) {
  const [accountSnapshot, setAccountSnapshot] = useState<AccountSnapshot | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const payload = await fetchJson<{ ok: boolean; snapshot?: AccountSnapshot }>("/api/account/overview");
      if (!isMounted) {
        return;
      }
      if (!payload || !payload.ok || !payload.snapshot) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      setAccountSnapshot(payload.snapshot);
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "Account";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const lastUpdated = formatDateTime(accountSnapshot?.taken_at);
  const emptyState = !isLoading && !hasError && !accountSnapshot;

  const kpiCards = [
    {
      title: "Buying Power",
      value: formatCurrency(parseNumber(accountSnapshot?.buying_power ?? null)),
      detail: `As of ${lastUpdated}`,
      detailTone: "info" as StatusTone,
      footnote: accountSnapshot?.source ? `Source: ${accountSnapshot.source}` : "Source: --",
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
    },
    {
      title: "Cash Balance",
      value: formatCurrency(parseNumber(accountSnapshot?.cash ?? null)),
      detail: `As of ${lastUpdated}`,
      detailTone: "neutral" as StatusTone,
      footnote: "Settled funds only",
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
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
      title: "Account Equity",
      value: formatCurrency(parseNumber(accountSnapshot?.equity ?? null)),
      detail: `As of ${lastUpdated}`,
      detailTone: "neutral" as StatusTone,
      footnote: "Net liquidation value",
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
    },
    {
      title: "Day Trades / PDT",
      value: formatNumber(0),
      detail: "PDT: Safe",
      detailTone: "success" as StatusTone,
      footnote: "Rolling 5-day window",
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

  const accountDetails = [
    { label: "Account ID", value: accountSnapshot?.account_id ?? "n/a" },
    { label: "Broker", value: "Alpaca Markets" },
    { label: "Account Type", value: accountSnapshot?.status ?? "n/a" },
    { label: "Base Currency", value: "USD" },
    { label: "Last Updated", value: lastUpdated },
  ];

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as StatusTone },
    { label: "Live", tone: "success" as StatusTone, showDot: true },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-28 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Account Overview</h1>
            <p className="mt-2 text-sm text-slate-500">Account health and capital status</p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Account data could not be loaded. Showing last known placeholders.
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
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Account Mode & Safety</h2>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span>Updated {lastUpdated}</span>
              </div>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-wrap items-center gap-6">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Trading Mode</div>
                  <div className="mt-2 flex items-center gap-2">
                    <StatusBadge label="Paper" tone="warning" showDot size="sm" />
                    <StatusBadge label="Live" tone="neutral" size="sm" />
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">Account Status</div>
                  <div className="mt-2 flex items-center gap-2">
                    <StatusBadge label="Healthy" tone="success" size="sm" showDot />
                    <StatusBadge label="Warning" tone="neutral" size="sm" />
                    <StatusBadge label="Restricted" tone="neutral" size="sm" />
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="mt-10">
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Account Details</h2>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              {emptyState ? (
                <div className="text-sm text-slate-500">Account details unavailable.</div>
              ) : (
                <dl className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
                  {accountDetails.map((detail) => (
                    <div key={detail.label} className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <dt className="text-slate-500">{detail.label}</dt>
                      <dd className="font-semibold text-slate-800 dark:text-slate-200">{detail.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          </section>

          <section className="mt-10">
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">System Notes & Restrictions</h2>
            <div className="mt-4 grid grid-cols-3 gap-4">
              <NotePanel
                title="Info"
                tone="info"
                message="Paper trading mode enabled for this account."
              />
              <NotePanel
                title="Warning"
                tone="warning"
                message="Outside trading window. Execution queues will pause."
              />
              <NotePanel
                title="Warning"
                tone="warning"
                message="Execution restricted by risk limits until next refresh."
              />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}


