import CardShell from "./CardShell";
import StatusChip, { type StatusChipTone } from "./StatusChip";

export interface PipelineStatusCardProps {
  lastRun: { date: string; start: string; end: string; duration: string };
  subprocess: {
    screener: "OK" | "FAIL" | "UNKNOWN";
    backTester: "OK" | "FAIL" | "UNKNOWN";
    metrics: "OK" | "FAIL" | "UNKNOWN";
  };
  isLive: boolean;
}

const statusToneMap: Record<PipelineStatusCardProps["subprocess"]["screener"], StatusChipTone> = {
  OK: "success",
  FAIL: "error",
  UNKNOWN: "neutral",
};

const formatValue = (value: string | null | undefined) => {
  if (!value || !value.trim()) {
    return "--";
  }
  return value;
};

export default function PipelineStatusCard({ lastRun, subprocess, isLive }: PipelineStatusCardProps) {
  const statusLabel = isLive ? "COMPLETED" : "OVERDUE";
  const statusTone = isLive ? "success" : "neutral";
  const cycleIcon = (
    <svg
      className="h-3.5 w-3.5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M3.51 9a9 9 0 0 1 14.13-3.36L23 10" />
      <path d="M20.49 15a9 9 0 0 1-14.13 3.36L1 14" />
    </svg>
  );
  const statusIcons = {
    ok: (
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <polyline points="23 4 23 10 17 10" />
        <polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.13-3.36L23 10" />
        <path d="M20.49 15a9 9 0 0 1-14.13 3.36L1 14" />
      </svg>
    ),
    fail: (
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 8v5" />
        <path d="M12 16h.01" />
      </svg>
    ),
    pending: (
      <svg
        className="h-3.5 w-3.5"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 3" />
      </svg>
    ),
  } as const;

  const statusDisplay = (status: PipelineStatusCardProps["subprocess"]["screener"]) => {
    if (status === "OK") {
      return { label: "CYCLE", tone: statusToneMap[status], icon: statusIcons.ok };
    }
    if (status === "FAIL") {
      return { label: "FAILED", tone: statusToneMap[status], icon: statusIcons.fail };
    }
    return { label: "PENDING", tone: statusToneMap[status], icon: statusIcons.pending };
  };

  const valueStyles =
    "rounded-md border border-cyan-400/40 bg-slate-950/60 px-2 py-0.5 text-[13px] font-semibold tracking-[0.01em] text-cyan-100 tabular-nums";
  const lastRunStats = [
    { label: "Date", value: formatValue(lastRun?.date), valueClassName: valueStyles },
    { label: "Start Time", value: formatValue(lastRun?.start), valueClassName: valueStyles },
    { label: "End Time", value: formatValue(lastRun?.end), valueClassName: valueStyles },
    { label: "Duration", value: formatValue(lastRun?.duration), valueClassName: valueStyles },
  ];

  return (
    <CardShell className="border-cyan-400/30 bg-gradient-to-br from-slate-950/90 via-slate-900/80 to-cyan-950/70 p-4 shadow-[0_18px_40px_-24px_rgba(34,211,238,0.55)] sm:p-5">
      <div className="rounded-xl border border-cyan-400/40 bg-gradient-to-r from-slate-950/90 via-slate-900/90 to-cyan-950/80 p-2.5 shadow-[0_0_24px_-12px_rgba(34,211,238,0.6)] sm:p-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/20 text-cyan-200">
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
                <path d="M4 6h6v6H4z" />
                <path d="M14 6h6v6h-6z" />
                <path d="M9.5 9h5" />
                <path d="M12 12v6" />
                <path d="M9 18h6" />
              </svg>
            </div>
            <div className="flex flex-col">
              <h3 className="text-[15px] font-bold leading-[22px] text-white">
                Pipeline Daily Status
              </h3>
              <span className="text-[11px] text-cyan-100/80">
                Screener &gt; Backtest &gt; Metrics
              </span>
            </div>
          </div>
          <StatusChip label={statusLabel} tone={statusTone} icon={cycleIcon} />
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-sm" />
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-200/70">
            Last Task Run
          </span>
        </div>
        <div className="rounded-xl border border-cyan-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(34,211,238,0.45)] sm:p-3">
          <div className="flex flex-col gap-2">
            {lastRunStats.map((stat) => (
              <div key={stat.label} className="flex items-baseline gap-2">
                <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-100/70">
                  {stat.label}:
                </span>
                <span className={stat.valueClassName}>
                  {stat.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-sm" />
          <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-200/70">
            Subprocess Status
          </span>
        </div>
        <div className="flex flex-col gap-2 rounded-xl border border-cyan-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(34,211,238,0.45)] sm:p-3">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-100/70">
              Screener:
            </span>
            <StatusChip {...statusDisplay(subprocess.screener)} />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-100/70">
              Back Test:
            </span>
            <StatusChip {...statusDisplay(subprocess.backTester)} />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-100/70">
              Metrics:
            </span>
            <StatusChip {...statusDisplay(subprocess.metrics)} />
          </div>
        </div>
      </div>
    </CardShell>
  );
}
