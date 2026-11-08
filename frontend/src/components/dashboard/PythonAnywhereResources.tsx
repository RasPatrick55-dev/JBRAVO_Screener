import type { CSSProperties } from "react";

export interface PythonAnywhereResource {
  label: string;
  value: number | null;
  used?: number | null;
  limit?: number | null;
  quota?: number | null;
  unit?: string | null;
  source?: string | null;
  captured_at?: string | null;
  age_seconds?: number | null;
  max_age_seconds?: number | null;
  stale?: boolean | null;
  status?: string | null;
  over_quota?: boolean;
}

export const unknownResources: PythonAnywhereResource[] = ["CPU Usage", "File Storage", "Postgres Storage"]
  .map(label => ({ label, value: null, status: "unknown" }));

export function resourceRows(resources: PythonAnywhereResource[] | null): PythonAnywhereResource[] {
  return unknownResources.map(fallback => resources?.find(row => row.label.toLowerCase() === fallback.label.toLowerCase()) ?? fallback);
}

const formatAmount = (value: number | null | undefined) => typeof value === "number" && Number.isFinite(value)
  ? value.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "unknown";

interface PythonAnywhereResourcesProps {
  title?: string;
  resources: PythonAnywhereResource[];
}

export default function PythonAnywhereResources({
  title = "PythonAnywhere",
  resources,
}: PythonAnywhereResourcesProps) {
  return (
    <div
      className="flex h-full flex-col rounded-2xl border border-sky-400/30 bg-gradient-to-br from-slate-950/95 via-slate-900/85 to-sky-950/70 p-4 shadow-[0_18px_40px_-24px_rgba(56,189,248,0.5)] sm:p-5"
      style={
        {
          "--ds-progress-rail": "rgba(15, 23, 42, 0.7)",
          "--ds-progress-fill": "rgba(56, 189, 248, 0.85)",
        } as CSSProperties
      }
    >
      <div className="rounded-xl border border-sky-400/40 bg-gradient-to-r from-slate-950/90 via-slate-900/90 to-sky-950/80 p-2.5 shadow-[0_0_24px_-12px_rgba(56,189,248,0.6)] sm:p-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-500/20 text-sky-200">
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
              <rect x="3" y="4" width="18" height="7" rx="2" />
              <rect x="3" y="13" width="18" height="7" rx="2" />
              <path d="M7 8h.01" />
              <path d="M7 17h.01" />
            </svg>
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-bold leading-[22px] text-white">{title}</h3>
            <span className="text-[11px] text-sky-100/80">Resource utilization</span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        {resources.map((resource) => {
          const value = typeof resource.value === "number" && Number.isFinite(resource.value) ? resource.value : null;
          const safeValue = value === null ? 0 : Math.max(0, Math.min(100, value));
          const status = resource.stale === true || resource.status === "stale" ? "STALE"
            : resource.status === "fresh" && resource.captured_at && value !== null ? "FRESH" : "UNKNOWN";
          return (
            <div
              key={resource.label}
              className="rounded-xl border border-sky-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(56,189,248,0.35)] sm:p-3"
            >
              <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-[0.08em] text-sky-100/70">
                <span>{resource.label}</span>
                <span className="text-sky-100">{value === null ? "Unknown" : `${formatAmount(value)}%`}</span>
              </div>
              <div className="mt-2 h-2 w-full rounded-full bg-progress-rail">
                <div
                  className="h-2 rounded-full bg-progress-fill"
                  style={{ width: `${safeValue}%` }}
                />
              </div>
              <div className="mt-2 space-y-1 text-[10px] text-sky-100/80">
                <p>{status}{value !== null && value > 100 ? " / Over quota in this snapshot" : ""}</p>
                <p>Used / quota: {formatAmount(resource.used)} / {formatAmount(resource.quota ?? resource.limit)} {resource.unit ?? ""}</p>
                <p>Source: {resource.source || "unknown"}</p>
                <p>Captured: {resource.captured_at || "unknown"}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
