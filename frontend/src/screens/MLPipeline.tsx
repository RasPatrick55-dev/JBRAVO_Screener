import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import KPICard from "../components/cards/KPICard";
import StatusBadge from "../components/badges/StatusBadge";
import LogViewer from "../components/panels/LogViewer";
import type { LogEntry, StatusTone } from "../types/ui";

type MLPipelineProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

type ApiHealthResponse = {
  data_ok?: boolean | null;
  feed?: string | null;
  last_run_utc?: string | null;
  pipeline_rc?: number | null;
  rows_final?: number | null;
  rows_premetrics?: number | null;
  symbols_in?: number | null;
  symbols_with_required_bars?: number | null;
  run_type?: string | null;
  latest_source?: string | null;
  freshness?: {
    age_seconds?: number | null;
    freshness_level?: string | null;
  };
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

export default function MLPipeline({ activeTab, onTabSelect }: MLPipelineProps) {
  const [healthSnapshot, setHealthSnapshot] = useState<ApiHealthResponse | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const [health, pipelineLog] = await Promise.all([
        fetchJson<ApiHealthResponse>("/api/health"),
        fetchText("/api/logs/pipeline.log"),
      ]);
      if (!isMounted) {
        return;
      }
      if (!health && !pipelineLog) {
        setHasError(true);
        setIsLoading(false);
        return;
      }
      setHealthSnapshot(health);
      setLogEntries(buildLogEntries(pipelineLog, 8));
      setHasError(false);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const currentTab = activeTab ?? "ML Pipeline";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  const pipelineState = pipelineStatus(healthSnapshot?.pipeline_rc ?? null);
  const lastRun = formatDateTime(healthSnapshot?.last_run_utc);
  const freshness = formatAge(healthSnapshot?.freshness?.age_seconds ?? null);
  const coveragePct = useMemo(() => {
    const covered = parseNumber(healthSnapshot?.symbols_with_required_bars ?? null);
    const total = parseNumber(healthSnapshot?.symbols_in ?? null);
    if (!covered || !total) {
      return null;
    }
    return (covered / total) * 100;
  }, [healthSnapshot?.symbols_in, healthSnapshot?.symbols_with_required_bars]);

  const kpiCards = [
    {
      title: "Pipeline Status",
      value: pipelineState.label,
      detail: healthSnapshot?.run_type ? `Run: ${healthSnapshot.run_type}` : "Run: n/a",
      detailTone: pipelineState.tone,
      footnote: `Updated ${freshness}`,
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-sky-600" fill="none">
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
      title: "Last Run",
      value: lastRun,
      detail: `Rows scored: ${formatNumber(parseNumber(healthSnapshot?.rows_final ?? null))}`,
      detailTone: "neutral" as StatusTone,
      footnote: `Source: ${healthSnapshot?.latest_source ?? "n/a"}`,
      icon: (
        <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
          <path
            d="M12 7v6l4 2"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
        </svg>
      ),
    },
    {
      title: "Coverage",
      value: formatPercent(coveragePct),
      detail: `Symbols: ${formatNumber(parseNumber(healthSnapshot?.symbols_in ?? null))}`,
      detailTone: "neutral" as StatusTone,
      footnote: `With bars: ${formatNumber(
        parseNumber(healthSnapshot?.symbols_with_required_bars ?? null)
      )}`,
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
      title: "Data Feed",
      value: healthSnapshot?.feed ? healthSnapshot.feed.toUpperCase() : "n/a",
      detail: healthSnapshot?.data_ok ? "Streaming" : "Delayed",
      detailTone: healthSnapshot?.data_ok ? ("success" as StatusTone) : ("warning" as StatusTone),
      footnote: "Market data status",
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

  const stages = [
    {
      label: "Feature Build",
      status: pipelineState.label,
      tone: pipelineState.tone,
      detail: "Bars normalized and enriched",
    },
    {
      label: "Model Training",
      status: pipelineState.label,
      tone: pipelineState.tone,
      detail: "Risk-adjusted swing model",
    },
    {
      label: "Signal Generation",
      status: pipelineState.label,
      tone: pipelineState.tone,
      detail: "Scoring and ranking",
    },
    {
      label: "Quality Gates",
      status: healthSnapshot?.data_ok ? "Passed" : "Review",
      tone: healthSnapshot?.data_ok ? ("success" as StatusTone) : ("warning" as StatusTone),
      detail: "Liquidity and coverage checks",
    },
  ];

  const artifacts = [
    { label: "Latest Model", value: "Swing-v1.3" },
    { label: "Feature Set", value: "FeatureSet-2026-01-20" },
    { label: "Scoring Cache", value: "candidate_scores_cache" },
    { label: "Snapshot Source", value: healthSnapshot?.latest_source ?? "n/a" },
  ];

  const emptyLog: LogEntry = {
    time: "--:--:--",
    level: "INFO",
    message: "Pipeline log is quiet.",
  };
  const displayEntries: LogEntry[] = logEntries.length ? logEntries : [emptyLog];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-32 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">ML Pipeline</h1>
            <p className="mt-2 text-sm text-slate-500">
              Model readiness, feature pipeline health, and scoring outputs.
            </p>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              Pipeline data could not be loaded. Showing placeholders.
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

          <section className="mt-10 grid grid-cols-[1.05fr_0.95fr] gap-6">
            <div>
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Pipeline Stages</h2>
                <StatusBadge label={pipelineState.label} tone={pipelineState.tone} size="sm" />
              </div>
              <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="space-y-3">
                  {stages.map((stage) => (
                    <div key={stage.label} className="flex items-center justify-between rounded-xl border border-slate-100 px-4 py-3">
                      <div>
                        <div className="text-xs uppercase tracking-wide text-slate-400">{stage.label}</div>
                        <div className="mt-1 text-sm font-semibold text-slate-800 dark:text-slate-200">{stage.detail}</div>
                      </div>
                      <StatusBadge label={stage.status} tone={stage.tone} size="sm" />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Model Artifacts</h2>
              <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <dl className="grid grid-cols-1 gap-y-4 text-sm">
                  {artifacts.map((item) => (
                    <div key={item.label} className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <dt className="text-slate-500">{item.label}</dt>
                      <dd className="font-semibold text-slate-800 dark:text-slate-200">{item.value}</dd>
                    </div>
                  ))}
                </dl>
                <div className="mt-4 rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  Rows pre-metrics: {formatNumber(parseNumber(healthSnapshot?.rows_premetrics ?? null))}
                </div>
              </div>
            </div>
          </section>

          <section className="mt-10">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Pipeline Activity</h2>
              <span className="text-xs text-slate-500">Latest pipeline events</span>
            </div>
            <div className="mt-4">
              <LogViewer title="ML Pipeline Log" statusLabel="Streaming" entries={displayEntries} actionLabel="Clear" />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}



