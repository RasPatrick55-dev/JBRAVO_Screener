import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import LogViewer from "../components/panels/LogViewer";
import type { LogEntry, StatusTone } from "../types/ui";

type ScreenerOverviewProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

type ApiHealthResponse = {
  data_ok?: boolean | null;
  data_status?: number | null;
  feed?: string | null;
  last_run_utc?: string | null;
  latest_source?: string | null;
  pipeline_rc?: number | null;
  rows_final?: number | null;
  rows_premetrics?: number | null;
  symbols_in?: number | null;
  symbols_with_required_bars?: number | null;
  run_type?: string | null;
  freshness?: {
    age_seconds?: number | null;
    freshness_level?: string | null;
  };
};

type CandidateRecord = {
  symbol?: string | null;
  score?: number | null;
  exchange?: string | null;
  close?: number | null;
  volume?: number | null;
  entry_price?: number | null;
  adv20?: number | null;
  atrp?: number | null;
  source?: string | null;
  timestamp?: string | null;
  run_date?: string | null;
};

type ScreenerCandidatesResponse = {
  ok?: boolean;
  rows?: CandidateRecord[];
  rows_final?: number | null;
  run_date?: string | null;
};

type CandidateRow = {
  symbol: string;
  score: number | null;
  exchange: string;
  close: number | null;
  volume: number | null;
  entryPrice: number | null;
  adv20: number | null;
  atrp: number | null;
  source: string;
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

const buildLogEntries = (text: string | null, limit = 8) => {
  const entries: ParsedLogEntry[] = [];
  if (!text) {
    return [] as LogEntry[];
  }
  const pattern =
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?(?:\s+-\s+[^-]+\s+-\s+)?\s*(?:\[(\w+)\])?\s*(.*)$/;
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
  entries.sort((a, b) => b.timestampMs - a.timestampMs);
  return entries.slice(0, limit).map(({ timestampMs, ...entry }) => entry);
};

const pipelineStatus = (rc: number | null | undefined) => {
  if (rc === 0) {
    return { label: "Healthy", tone: "success" as StatusTone };
  }
  if (rc === null || rc === undefined) {
    return { label: "Unknown", tone: "neutral" as StatusTone };
  }
  return { label: "Degraded", tone: "warning" as StatusTone };
};

export default function ScreenerOverview({ activeTab, onTabSelect }: ScreenerOverviewProps) {
  const [healthSnapshot, setHealthSnapshot] = useState<ApiHealthResponse | null>(null);
  const [candidates, setCandidates] = useState<CandidateRow[]>([]);
  const [candidateCount, setCandidateCount] = useState<number | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const [health, candidatesPayload, pipelineLog] = await Promise.all([
        fetchJson<ApiHealthResponse>("/api/health"),
        fetchJson<ScreenerCandidatesResponse>("/api/screener/candidates"),
        fetchText("/api/logs/pipeline.log"),
      ]);
      if (!isMounted) {
        return;
      }
      if (!health && !candidatesPayload) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      const rows = (candidatesPayload?.rows ?? []).map((row) => ({
        symbol: row.symbol ?? "--",
        score: parseNumber(row.score ?? null),
        exchange: row.exchange ?? "--",
        close: parseNumber(row.close ?? null),
        volume: parseNumber(row.volume ?? null),
        entryPrice: parseNumber(row.entry_price ?? null),
        adv20: parseNumber(row.adv20 ?? null),
        atrp: parseNumber(row.atrp ?? null),
        source: row.source ?? "--",
      }));
      setHealthSnapshot(health);
      setCandidates(rows);
      setCandidateCount(
        parseNumber(candidatesPayload?.rows_final ?? null) ??
          parseNumber(health?.rows_final ?? null)
      );
      setLogEntries(buildLogEntries(pipelineLog, 8));
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "Screener";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  const coveragePct = useMemo(() => {
    const covered = parseNumber(healthSnapshot?.symbols_with_required_bars ?? null);
    const total = parseNumber(healthSnapshot?.symbols_in ?? null);
    if (!covered || !total) {
      return null;
    }
    return (covered / total) * 100;
  }, [healthSnapshot?.symbols_in, healthSnapshot?.symbols_with_required_bars]);

  const feedTone: StatusTone =
    healthSnapshot?.data_ok === true
      ? "success"
      : healthSnapshot?.data_ok === false
        ? "warning"
        : "neutral";

  const pipelineState = pipelineStatus(healthSnapshot?.pipeline_rc ?? null);
  const lastRun = formatDateTime(healthSnapshot?.last_run_utc);
  const freshness = formatAge(healthSnapshot?.freshness?.age_seconds ?? null);

  const kpiCards = [
    {
      title: "Candidates Ready",
      value: formatNumber(candidateCount),
      detail: `Last run: ${lastRun}`,
      detailTone: "neutral" as StatusTone,
      footnote: `Updated ${freshness}`,
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
      title: "Universe Coverage",
      value: formatNumber(parseNumber(healthSnapshot?.symbols_with_required_bars ?? null)),
      detail: `of ${formatNumber(parseNumber(healthSnapshot?.symbols_in ?? null))} symbols`,
      detailTone: "neutral" as StatusTone,
      footnote: `Coverage: ${formatPercent(coveragePct)}`,
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
      title: "Market Data Feed",
      value: healthSnapshot?.feed ? healthSnapshot.feed.toUpperCase() : "n/a",
      detail: healthSnapshot?.data_ok ? "Streaming" : "Delayed",
      detailTone: feedTone,
      footnote: `Status: ${healthSnapshot?.data_status ?? "n/a"}`,
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-violet-600" fill="none">
          <path
            d="M4 12h3l2-5 4 10 2-5h5"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ),
    },
    {
      title: "Pipeline Status",
      value: pipelineState.label,
      detail: healthSnapshot?.run_type ? `Run: ${healthSnapshot.run_type}` : "Run: n/a",
      detailTone: pipelineState.tone,
      footnote: healthSnapshot?.latest_source
        ? `Source: ${healthSnapshot.latest_source}`
        : "Source: n/a",
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

  const metadata = [
    { label: "Run Type", value: healthSnapshot?.run_type ?? "n/a" },
    { label: "Rows Final", value: formatNumber(parseNumber(healthSnapshot?.rows_final ?? null)) },
    {
      label: "Rows Pre-metrics",
      value: formatNumber(parseNumber(healthSnapshot?.rows_premetrics ?? null)),
    },
    { label: "Symbols In", value: formatNumber(parseNumber(healthSnapshot?.symbols_in ?? null)) },
    { label: "Last Run", value: lastRun },
    { label: "Freshness", value: freshness },
  ];

  const emptyState = !isLoading && !hasError && candidates.length === 0;
  const emptyLog: LogEntry = {
    time: "--:--:--",
    level: "INFO",
    message: "Pipeline log is quiet.",
  };
  const displayEntries: LogEntry[] = logEntries.length ? logEntries : [emptyLog];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-20 pb-12 sm:pt-24">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Screener Overview</h1>
            <p className="mt-2 text-sm text-slate-500">
              Pre-market candidates, coverage, and pipeline readiness.
            </p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Screener data could not be loaded. Showing placeholders.
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
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Candidate Shortlist</h2>
              <span className="text-xs text-slate-500">Top ranked opportunities</span>
            </div>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="grid grid-cols-[1fr_0.8fr_0.7fr_0.9fr_0.9fr_0.9fr_0.9fr_0.8fr] gap-3 border-b border-slate-100 px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <div>Symbol</div>
                <div>Score</div>
                <div>Exchange</div>
                <div>Close</div>
                <div>Entry</div>
                <div>ADV20</div>
                <div>ATRP</div>
                <div>Source</div>
              </div>
              <div className="max-h-80 overflow-y-auto px-5 py-2">
                {isLoading ? (
                  <div className="space-y-3 py-4">
                    {Array.from({ length: 6 }).map((_, index) => (
                      <div key={index} className="h-6 rounded-full bg-slate-100" />
                    ))}
                  </div>
                ) : candidates.length ? (
                  candidates.slice(0, 12).map((row, index) => (
                    <div
                      key={`${row.symbol}-${index}`}
                      className="grid grid-cols-[1fr_0.8fr_0.7fr_0.9fr_0.9fr_0.9fr_0.9fr_0.8fr] items-center gap-3 border-b border-slate-100 py-3 text-sm text-slate-700"
                    >
                      <div className="font-semibold text-slate-900 dark:text-slate-100">{row.symbol}</div>
                      <div>{formatNumber(row.score)}</div>
                      <div>{row.exchange}</div>
                      <div>{formatCurrency(row.close)}</div>
                      <div>{formatCurrency(row.entryPrice)}</div>
                      <div>{formatNumber(row.adv20)}</div>
                      <div>{formatPercent(row.atrp)}</div>
                      <div className="text-xs text-slate-500">{row.source}</div>
                    </div>
                  ))
                ) : (
                  <div className="py-6 text-sm text-slate-500">
                    {emptyState ? "No candidates available yet." : "Candidates unavailable."}
                  </div>
                )}
              </div>
            </div>
          </section>

          <section className="mt-10 grid grid-cols-[1.1fr_0.9fr] gap-6">
            <div>
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Pipeline Metadata</h2>
                <StatusBadge label={pipelineState.label} tone={pipelineState.tone} size="sm" />
              </div>
              <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <dl className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
                  {metadata.map((item) => (
                    <div key={item.label} className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <dt className="text-slate-500">{item.label}</dt>
                      <dd className="font-semibold text-slate-800 dark:text-slate-200">{item.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            <div>
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Pipeline Activity</h2>
              <div className="mt-4">
                <LogViewer
                  title="Pipeline Log"
                  statusLabel="Streaming"
                  entries={displayEntries}
                  actionLabel="Clear"
                />
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}



