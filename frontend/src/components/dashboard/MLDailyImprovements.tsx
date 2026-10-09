import CardShell from "./CardShell";
import StatusBadge from "../badges/StatusBadge";
import { useMlOverview } from "../ml/useMlOverview";
import type { StatusTone } from "../../types/ui";

interface MLDailyImprovementsProps {
  onOpenPipeline?: () => void;
}

const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const stageDotTone: Record<StatusTone, string> = {
  success: "jbravo-status-dot-success",
  warning: "bg-amber-400",
  error: "jbravo-status-dot-error",
  info: "bg-sky-400",
  neutral: "bg-slate-500",
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

const formatCoverage = (pct: number | null | undefined, nonNull: number | null | undefined, total: number | null | undefined) => {
  if (typeof nonNull === "number" && typeof total === "number" && total > 0) {
    return `${nonNull}/${total}`;
  }
  if (typeof pct === "number") {
    return `${percentFormatter.format(pct)}%`;
  }
  return "No data";
};

const formatAction = (value: string | null | undefined) => {
  const text = String(value || "").trim();
  if (!text) {
    return "None";
  }
  return text.replace(/_/g, " ");
};

export default function MLDailyImprovements({ onOpenPipeline }: MLDailyImprovementsProps) {
  const { data, isLoading, hasError } = useMlOverview();

  const statusLabel = isLoading ? "Loading" : data?.status?.label ?? "No data";
  const statusTone = isLoading ? "info" : data?.status?.tone ?? "neutral";
  const coverage = data?.coverage;
  const champion = data?.champion;
  const monitor = data?.monitor;
  const stages = (data?.pipeline_stages ?? []).filter((stage) =>
    ["Features", "Predict", "Eval", "Monitor", "Auto-remediate"].includes(stage.label)
  );

  const metricTiles = [
    {
      label: "Freshness",
      value: data?.freshness?.label ?? "No data",
      detail: data?.freshness?.reason ?? "Waiting for predictions data",
    },
    {
      label: "Coverage",
      value: formatCoverage(coverage?.pct ?? null, coverage?.non_null ?? null, coverage?.total ?? null),
      detail:
        typeof coverage?.pct === "number"
          ? `${percentFormatter.format(coverage.pct)}% scored`
          : "No model score overlay",
    },
    {
      label: "Champion",
      value:
        champion?.present
          ? `${formatAction(champion.calibration ?? "none")} / ${formatShortDate(champion.run_date)}`
          : "No data",
      detail: champion?.feature_set ? `Feature set ${champion.feature_set}` : "Champion not available",
    },
    {
      label: "Action",
      value: formatAction(monitor?.recommended_action ?? "none"),
      detail:
        data?.monitor?.guard_decision && data.monitor.guard_decision !== "allow"
          ? `Guard ${data.monitor.guard_decision}`
          : "No remediation requested",
    },
  ];

  const activityTiles = [
    { label: "Predict", value: formatShortDate(data?.timestamps?.last_predict) },
    { label: "Recal", value: formatShortDate(data?.timestamps?.last_recalibrate) },
    { label: "Fix", value: formatShortDate(data?.timestamps?.last_autoremediate) },
  ];

  return (
    <button type="button" onClick={onOpenPipeline} className="h-full w-full text-left">
      <CardShell className="border-indigo-400/30 bg-gradient-to-br from-slate-950/95 via-indigo-950/80 to-cyan-950/72 p-4 shadow-[0_18px_40px_-24px_rgba(99,102,241,0.45)] transition-transform duration-200 hover:-translate-y-0.5 hover:border-indigo-300/45 hover:shadow-[0_22px_46px_-24px_rgba(99,102,241,0.58)] sm:p-5">
        <div className="rounded-xl border border-indigo-400/40 bg-gradient-to-r from-slate-950/90 via-indigo-950/85 to-cyan-950/80 p-2.5 shadow-[0_0_24px_-12px_rgba(129,140,248,0.55)] sm:p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-100">
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 3v5" />
                  <path d="M12 16v5" />
                  <path d="M4.22 5.22l3.54 3.54" />
                  <path d="M16.24 15.24l3.54 3.54" />
                  <path d="M3 12h5" />
                  <path d="M16 12h5" />
                  <path d="M4.22 18.78l3.54-3.54" />
                  <path d="M16.24 8.76l3.54-3.54" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              </div>
              <div className="min-w-0">
                <h3 className="truncate text-[15px] font-bold leading-[22px] text-white">
                  Machine Learning
                </h3>
                <p className="text-[11px] text-indigo-100/80">
                  Model health, freshness, and remediation
                </p>
              </div>
            </div>
            <StatusBadge label={statusLabel} tone={statusTone} size="sm" showDot />
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2.5">
          {isLoading
            ? Array.from({ length: 4 }).map((_, index) => (
                <div
                  key={`ml-card-skeleton-${index}`}
                  className="rounded-xl border border-indigo-400/20 bg-slate-950/55 px-3 py-3"
                >
                  <div className="h-3 w-16 animate-pulse rounded bg-slate-700/70" />
                  <div className="mt-2 h-4 w-20 animate-pulse rounded bg-slate-600/70" />
                  <div className="mt-2 h-3 w-full animate-pulse rounded bg-slate-800/70" />
                </div>
              ))
            : metricTiles.map((tile) => (
                <div
                  key={tile.label}
                  className="rounded-xl border border-indigo-400/20 bg-slate-950/55 px-3 py-3 shadow-[0_0_18px_-14px_rgba(129,140,248,0.45)]"
                >
                  <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-indigo-100/60">
                    {tile.label}
                  </div>
                  <div className="mt-2 font-cousine text-[15px] font-semibold text-indigo-50">
                    {tile.value}
                  </div>
                  <div className="mt-1 text-[11px] leading-4 text-indigo-100/62">{tile.detail}</div>
                </div>
              ))}
        </div>

        <div className="mt-4 rounded-xl border border-indigo-400/25 bg-slate-950/55 px-3 py-3 shadow-[0_0_18px_-14px_rgba(56,189,248,0.32)]">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-indigo-100/60">
              Pipeline Stages
            </div>
            <div className="text-[11px] text-indigo-100/60">
              Last ML run {formatShortDate(data?.status?.last_ml_run ?? data?.updated_at)}
            </div>
          </div>
          <div className="mt-3 grid grid-cols-5 gap-2">
            {(stages.length ? stages : [
              { label: "Features", tone: "neutral" as const, status: "No data" },
              { label: "Predict", tone: "neutral" as const, status: "No data" },
              { label: "Eval", tone: "neutral" as const, status: "No data" },
              { label: "Monitor", tone: "neutral" as const, status: "No data" },
              { label: "Auto-remediate", tone: "neutral" as const, status: "No data" },
            ]).map((stage) => (
              <div key={stage.label} className="flex min-w-0 flex-col items-center gap-2 text-center">
                <span className={`h-2.5 w-2.5 rounded-full ${stageDotTone[stage.tone ?? "neutral"]}`} />
                <span className="text-[10px] font-semibold text-indigo-50/90">{stage.label}</span>
                <span className="text-[9px] uppercase tracking-[0.08em] text-indigo-100/52">
                  {stage.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-indigo-400/20 bg-slate-950/45 px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-2">
            {activityTiles.map((item) => (
              <span
                key={item.label}
                className="inline-flex items-center gap-1 rounded-full border border-indigo-400/20 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-indigo-100/72"
              >
                <span className="text-indigo-100/44">{item.label}</span>
                {item.value}
              </span>
            ))}
          </div>
          <div className="text-[11px] text-indigo-100/64">
            {hasError ? "ML data unavailable" : `Updated ${formatDateTime(data?.updated_at)}`}
          </div>
        </div>
      </CardShell>
    </button>
  );
}
