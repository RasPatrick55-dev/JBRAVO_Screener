import type { CSSProperties } from "react";

interface PythonAnywhereResource {
  label: string;
  value: number;
}

const widthClassFor = (value: number) => {
  if (value === 10) return "w-[10%]";
  if (value === 35) return "w-[35%]";
  if (value === 50) return "w-[50%]";
  if (value === 75) return "w-[75%]";
  if (value === 90) return "w-[90%]";
  if (value === 100) return "w-[100%]";
  return "w-[0%]";
};

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
          const safeValue = Math.max(0, Math.min(100, resource.value));
          const widthClass = widthClassFor(safeValue);
          return (
            <div
              key={resource.label}
              className="rounded-xl border border-sky-400/30 bg-slate-950/60 p-2.5 shadow-[0_0_18px_-12px_rgba(56,189,248,0.35)] sm:p-3"
            >
              <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-[0.08em] text-sky-100/70">
                <span>{resource.label}</span>
                <span className="text-sky-100">{safeValue}%</span>
              </div>
              <div className="mt-2 h-2 w-full rounded-full bg-progress-rail">
                <div
                  className={`h-2 rounded-full bg-progress-fill ${widthClass}`}
                  style={{ width: `${safeValue}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
