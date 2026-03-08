import { useMemo, type ReactNode } from "react";
import { buildNavbarBadges, type LiveDataSyncState, useLiveTradingStatus } from "../components/navbar/liveStatus";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import LatestLogsTable from "../components/dashboard/LatestLogsTable";
import StatusBadge from "../components/badges/StatusBadge";
import { useMlOverview, type MlStage } from "../components/ml/useMlOverview";
import type { StatusTone } from "../types/ui";

type MLPipelineProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
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

const decimalFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 3,
});

const summaryCardSurfaces = [
  "jbravo-panel-cyan",
  "jbravo-panel-violet",
  "jbravo-panel-amber",
  "jbravo-panel-emerald",
  "jbravo-panel-cyan",
];

const cardTone: Record<StatusTone, string> = {
  success: "text-emerald-300",
  warning: "text-amber-300",
  error: "text-rose-300",
  info: "text-sky-300",
  neutral: "text-slate-200",
};

const formatDateTime = (value: string | null | undefined) => {
  if (!value) {
    return "No data";
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

const formatShortDate = (value: string | null | undefined) => {
  if (!value) {
    return "No data";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
};

const formatNumber = (value: number | null | undefined) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "No data";
  }
  return numberFormatter.format(value);
};

const formatPercent = (value: number | null | undefined) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "No data";
  }
  return `${percentFormatter.format(value)}%`;
};

const formatDecimal = (value: number | null | undefined) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "No data";
  }
  return decimalFormatter.format(value);
};

const formatAction = (value: string | null | undefined) => {
  const text = String(value || "").trim();
  if (!text) {
    return "none";
  }
  return text.replace(/_/g, " ");
};

const truncateMiddle = (value: string | null | undefined, max = 42) => {
  const text = String(value || "").trim();
  if (!text) {
    return "No data";
  }
  if (text.length <= max) {
    return text;
  }
  const lead = Math.max(12, Math.floor((max - 3) / 2));
  const tail = Math.max(10, max - lead - 3);
  return `${text.slice(0, lead)}...${text.slice(-tail)}`;
};

const formatCoverage = (pct: number | null | undefined, nonNull: number | null | undefined, total: number | null | undefined) => {
  if (typeof nonNull === "number" && typeof total === "number" && total > 0) {
    return `${nonNull}/${total}`;
  }
  return formatPercent(pct);
};

function SummaryCard({
  title,
  value,
  detail,
  footnote,
  tone = "neutral",
  statusLabel,
  statusTone,
  surface,
}: {
  title: string;
  value: string;
  detail: string;
  footnote: string;
  tone?: StatusTone;
  statusLabel?: string;
  statusTone?: StatusTone;
  surface: string;
}) {
  return (
    <article className={`rounded-2xl p-4 shadow-card jbravo-panel ${surface}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
            {title}
          </p>
          <p className={`mt-3 font-cousine text-2xl font-bold tabular-nums ${cardTone[tone]}`}>
            {value}
          </p>
          <p className="mt-2 text-sm text-primary">{detail}</p>
        </div>
        {statusLabel ? <StatusBadge label={statusLabel} tone={statusTone ?? tone} size="sm" /> : null}
      </div>
      <p className="mt-3 text-xs text-secondary">{footnote}</p>
    </article>
  );
}

function SectionCard({
  title,
  subtitle,
  children,
  surface,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  surface: string;
}) {
  return (
    <section className={`rounded-2xl p-3 shadow-card jbravo-panel sm:p-md ${surface}`}>
      <header>
        <h2 className="font-arimo text-sm font-semibold uppercase tracking-[0.08em] text-primary">
          {title}
        </h2>
        <p className="mt-1 text-xs text-secondary">{subtitle}</p>
      </header>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function DetailRow({
  label,
  value,
  note,
  variant = "cyan",
}: {
  label: string;
  value: string;
  note?: string;
  variant?: "cyan" | "amber" | "emerald";
}) {
  const variantClass =
    variant === "amber"
      ? "jbravo-panel-inner-amber"
      : variant === "emerald"
        ? "jbravo-panel-inner-emerald"
        : "jbravo-panel-inner-cyan";
  return (
    <div className={`flex items-start justify-between gap-3 rounded-xl px-3 py-3 jbravo-panel-inner ${variantClass}`}>
      <div className="min-w-0">
        <div className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
          {label}
        </div>
        {note ? <div className="mt-1 text-[11px] text-secondary">{note}</div> : null}
      </div>
      <div className="min-w-0 text-right font-cousine text-sm font-bold text-primary">{value}</div>
    </div>
  );
}

function StageTile({ stage }: { stage: MlStage }) {
  return (
    <article className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-violet outline outline-1 outline-indigo-400/30">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
            {stage.label}
          </p>
          <p className="mt-2 text-xs text-secondary">{stage.detail ?? "No data"}</p>
        </div>
        <StatusBadge label={stage.status} tone={stage.tone} size="sm" />
      </div>
      <p className="mt-3 font-cousine text-[11px] text-primary">{formatDateTime(stage.timestamp)}</p>
    </article>
  );
}

export default function MLPipeline({ activeTab, onTabSelect }: MLPipelineProps) {
  const { data, isLoading, hasError } = useMlOverview();
  const liveTradingStatus = useLiveTradingStatus(null);
  const pageSyncState: LiveDataSyncState = isLoading ? "loading" : hasError ? "error" : "ready";
  const rightBadges = useMemo(
    () => buildNavbarBadges(liveTradingStatus, pageSyncState),
    [liveTradingStatus, pageSyncState]
  );

  const currentTab = activeTab ?? "ML Pipeline";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const coverageDetail = useMemo(() => {
    const nonNull = data?.coverage?.non_null ?? null;
    const total = data?.coverage?.total ?? null;
    if (typeof nonNull === "number" && typeof total === "number" && total > 0) {
      return `${formatPercent(data?.coverage?.pct ?? null)} scored`;
    }
    return "No model-score overlay";
  }, [data?.coverage?.non_null, data?.coverage?.pct, data?.coverage?.total]);

  const summaryCards = [
    {
      title: "Predictions freshness",
      value: data?.freshness?.label ?? "No data",
      detail: data?.freshness?.reason ?? "Waiting for freshness diagnostics",
      footnote: `Last predict ${formatDateTime(data?.timestamps?.last_predict)}`,
      tone: (data?.freshness?.stale ? "warning" : "success") as StatusTone,
      statusLabel: data?.status?.label ?? "No data",
      statusTone: (data?.status?.tone ?? "neutral") as StatusTone,
    },
    {
      title: "Score coverage",
      value: formatCoverage(
        data?.coverage?.pct ?? null,
        data?.coverage?.non_null ?? null,
        data?.coverage?.total ?? null
      ),
      detail: coverageDetail,
      footnote: `run_ts ${formatDateTime(data?.coverage?.run_ts_utc)}`,
      tone:
        typeof data?.coverage?.pct === "number" && data.coverage.pct > 0
          ? ("success" as StatusTone)
          : ("warning" as StatusTone),
    },
    {
      title: "Monitor action",
      value: formatAction(data?.monitor?.recommended_action ?? "none"),
      detail:
        data?.monitor?.guard_decision && data.monitor.guard_decision !== "allow"
          ? `Guard ${data.monitor.guard_decision} (${data.monitor.guard_mode ?? "n/a"})`
          : "Health guard currently allowing enrichment",
      footnote: `PSI ${formatDecimal(data?.monitor?.psi_score ?? null)} · Sharpe ${formatDecimal(
        data?.monitor?.recent_sharpe ?? null
      )}`,
      tone:
        data?.monitor?.recommended_action === "retrain"
          ? ("error" as StatusTone)
          : data?.monitor?.recommended_action === "recalibrate"
            ? ("warning" as StatusTone)
            : ("success" as StatusTone),
    },
    {
      title: "Champion",
      value: data?.champion?.present
        ? `${formatAction(data.champion.calibration ?? "none")}`
        : "No data",
      detail: data?.champion?.present
        ? `Run ${formatShortDate(data.champion.run_date)} · feature ${data.champion.feature_set ?? "n/a"}`
        : "Champion artifact unavailable",
      footnote: `min_model_score ${data?.champion?.execution?.min_model_score ?? "n/a"} · require ${String(
        data?.champion?.execution?.require_model_score ?? false
      )}`,
      tone: data?.champion?.present ? ("success" as StatusTone) : ("neutral" as StatusTone),
    },
    {
      title: "Recent ML activity",
      value: formatShortDate(data?.timestamps?.last_ml_run),
      detail: `Predict ${formatShortDate(data?.timestamps?.last_predict)} · Eval ${formatShortDate(
        data?.timestamps?.last_eval
      )}`,
      footnote: `Remediate ${formatShortDate(data?.timestamps?.last_autoremediate)} · Recal ${formatShortDate(
        data?.timestamps?.last_recalibrate
      )}`,
      tone: "info" as StatusTone,
    },
  ];

  const stageRows = data?.pipeline_stages?.length ? data.pipeline_stages : [];
  const candidateRefreshDone =
    data?.enrichment?.candidate_refresh_done && typeof data.enrichment.candidate_refresh_done === "object"
      ? data.enrichment.candidate_refresh_done
      : null;
  const recentEventRows = (data?.recent_events ?? []).map((event) => ({
    dateTime: formatDateTime(event.timestamp),
    script: event.token ?? "ML",
    text: event.message ?? "No message",
    level: event.level ?? "INFO",
  }));

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-36 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-indigo-500/16 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <header className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">ML Pipeline</h1>
              <p className="mt-2 text-sm text-slate-500">
                Freshness, monitoring, remediation, and scoring pipeline
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge
                label={data?.status?.label ?? (isLoading ? "Loading" : "No data")}
                tone={(data?.status?.tone ?? (isLoading ? "info" : "neutral")) as StatusTone}
                size="sm"
                showDot
              />
              <span className="rounded-full border border-slate-400/30 bg-slate-900/40 px-3 py-1 text-xs text-slate-300">
                Last ML run {formatDateTime(data?.status?.last_ml_run ?? data?.updated_at)}
              </span>
            </div>
          </header>

          {hasError ? (
            <div className="mt-6 rounded-xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              ML overview could not be loaded. The page is holding layout with empty-state values.
            </div>
          ) : null}

          <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            {summaryCards.map((card, index) => (
              <SummaryCard
                key={card.title}
                title={card.title}
                value={isLoading ? "--" : card.value}
                detail={isLoading ? "Loading..." : card.detail}
                footnote={isLoading ? "Loading..." : card.footnote}
                tone={card.tone}
                statusLabel={card.statusLabel}
                statusTone={card.statusTone}
                surface={summaryCardSurfaces[index % summaryCardSurfaces.length]}
              />
            ))}
          </section>

          <section className="mt-6 grid gap-4 xl:grid-cols-[1.35fr_0.95fr]">
            <SectionCard
              title="Pipeline stages"
              subtitle="Labels, features, train, recalibrate, predict, eval, monitor, remediation, and trade attribution"
              surface="jbravo-panel-violet"
            >
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {stageRows.length
                  ? stageRows.map((stage) => <StageTile key={stage.label} stage={stage} />)
                  : Array.from({ length: 9 }).map((_, index) => (
                      <div
                        key={`stage-skeleton-${index}`}
                        className="h-[118px] animate-pulse rounded-xl jbravo-panel-inner jbravo-panel-inner-violet"
                      />
                    ))}
              </div>
            </SectionCard>

            <SectionCard
              title="Health & monitoring"
              subtitle="Drift, calibration, guard recommendation, and remediation posture"
              surface="jbravo-panel-amber"
            >
              <div className="space-y-3">
                <DetailRow
                  label="PSI drift"
                  value={formatDecimal(data?.monitor?.psi_score ?? null)}
                  note="Latest max PSI from ranker_monitor"
                  variant="amber"
                />
                <DetailRow
                  label="Recent Sharpe"
                  value={formatDecimal(data?.monitor?.recent_sharpe ?? null)}
                  note="Recent window strategy quality"
                  variant="amber"
                />
                <DetailRow
                  label="Calibration ECE"
                  value={formatDecimal(data?.monitor?.ece ?? null)}
                  note={`Delta ${formatDecimal(data?.monitor?.delta_ece ?? null)}`}
                  variant="amber"
                />
                <DetailRow
                  label="Recommended action"
                  value={formatAction(data?.monitor?.recommended_action ?? "none")}
                  note={`Guard ${formatAction(data?.monitor?.guard_decision ?? "allow")} / ${
                    data?.monitor?.guard_mode ?? "n/a"
                  }`}
                  variant="amber"
                />
                <div className="rounded-xl px-3 py-3 jbravo-panel-inner jbravo-panel-inner-amber">
                  <div className="font-arimo text-[11px] font-semibold uppercase tracking-[0.08em] text-secondary">
                    Guard reasons
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(data?.monitor?.guard_reasons?.length ? data.monitor.guard_reasons : ["none"]).map(
                      (reason) => (
                        <span
                          key={reason}
                          className="rounded-full border border-amber-400/25 bg-amber-500/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-amber-100"
                        >
                          {reason}
                        </span>
                      )
                    )}
                  </div>
                </div>
              </div>
            </SectionCard>
          </section>

          <section className="mt-6 grid gap-4 xl:grid-cols-2">
            <SectionCard
              title="Freshness & compatibility"
              subtitle="Model identity, prediction freshness, schema compatibility, overlap diagnostics, and refresh state"
              surface="jbravo-panel-cyan"
            >
              <div className="space-y-3">
                <DetailRow
                  label="Latest model"
                  value={truncateMiddle(data?.freshness?.model_path)}
                  note={`Predictions model ${truncateMiddle(data?.freshness?.pred_model_path)}`}
                />
                <DetailRow
                  label="Feature set"
                  value={`${data?.freshness?.latest_features_set ?? "n/a"} -> ${data?.freshness?.pred_features_set ?? "n/a"}`}
                  note={`Compatibility ${String(data?.freshness?.pred_compatible ?? "unknown")} / ${
                    data?.freshness?.pred_compat_reason ?? "n/a"
                  }`}
                />
                <DetailRow
                  label="Signatures"
                  value={truncateMiddle(data?.freshness?.latest_features_signature, 34)}
                  note={truncateMiddle(data?.freshness?.pred_features_signature, 34)}
                />
                <DetailRow
                  label="Overlap"
                  value={`${formatNumber(data?.overlap?.overlap ?? null)} of ${formatNumber(
                    data?.overlap?.candidates ?? null
                  )}`}
                  note={`Pred symbols ${formatNumber(data?.overlap?.prediction_symbols ?? null)} · ${
                    data?.overlap?.score_col ?? "n/a"
                  }`}
                />
                <DetailRow
                  label="Refresh result"
                  value={
                    candidateRefreshDone && "rc" in candidateRefreshDone
                      ? String(candidateRefreshDone.rc)
                      : "No recent refresh"
                  }
                  note={
                    data?.overlap?.sample_reason
                      ? `Sample reason ${data.overlap.sample_reason} · missing ${
                          data.overlap.missing_symbols?.slice(0, 3).join(", ") || "none"
                        }`
                      : `Freshness reason ${data?.freshness?.reason ?? "n/a"}`
                  }
                />
              </div>
            </SectionCard>

            <SectionCard
              title="Remediation & champion"
              subtitle="Champion settings, last remediation outcome, repredict state, and attribution summary"
              surface="jbravo-panel-emerald"
            >
              <div className="space-y-3">
                <DetailRow
                  label="Champion run"
                  value={formatShortDate(data?.champion?.run_date)}
                  note={`Calibration ${formatAction(data?.champion?.calibration ?? "none")} · feature ${
                    data?.champion?.feature_set ?? "n/a"
                  }`}
                  variant="emerald"
                />
                <DetailRow
                  label="Execution settings"
                  value={`min ${data?.champion?.execution?.min_model_score ?? "n/a"}`}
                  note={`require_model_score ${String(data?.champion?.execution?.require_model_score ?? false)}`}
                  variant="emerald"
                />
                <DetailRow
                  label="Last remediation"
                  value={`${formatAction(data?.remediation?.last_kind ?? "none")} / ${String(
                    data?.remediation?.executed ?? false
                  )}`}
                  note={`Action ${formatAction(data?.remediation?.last_action ?? "none")} · run ${formatShortDate(
                    data?.remediation?.run_date
                  )}`}
                  variant="emerald"
                />
                <DetailRow
                  label="Repredict after remediation"
                  value={String(data?.remediation?.repredict_executed ?? false)}
                  note={`rc ${data?.remediation?.repredict_rc ?? "n/a"} · ${
                    data?.remediation?.repredict_reason ?? "no repredict metadata"
                  }`}
                  variant="emerald"
                />
                <DetailRow
                  label="Trade attribution"
                  value={`${formatNumber(data?.trade_attribution?.trades_scored ?? null)} / ${formatNumber(
                    data?.trade_attribution?.trades_total ?? null
                  )}`}
                  note={`Status ${data?.trade_attribution?.status ?? "n/a"} · brier ${formatDecimal(
                    data?.trade_attribution?.brier ?? null
                  )}`}
                  variant="emerald"
                />
              </div>
            </SectionCard>
          </section>

          <section className="mt-6">
            <LatestLogsTable
              title="Recent ML Events"
              rows={
                recentEventRows.length
                  ? recentEventRows
                  : [
                      {
                        dateTime: "No data",
                        script: "ML",
                        text: "No ML pipeline events available.",
                        level: "INFO" as const,
                      },
                    ]
              }
              enableFilter={false}
            />
          </section>
        </div>
      </main>
    </div>
  );
}
