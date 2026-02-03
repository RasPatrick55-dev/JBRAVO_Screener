import type { CSSProperties } from "react";

interface MLDailyImprovementItem {
  label: string;
  active: boolean;
}

interface MLDailyImprovementsProps {
  title?: string;
  items: MLDailyImprovementItem[];
}

export default function MLDailyImprovements({
  title = "Machine Learning Daily Improvements",
  items,
}: MLDailyImprovementsProps) {
  return (
    <div
      className="flex h-full flex-col rounded-2xl border border-amber-400/30 bg-gradient-to-br from-slate-950/95 via-amber-950/70 to-amber-900/60 p-4 shadow-[0_18px_40px_-24px_rgba(251,191,36,0.45)] sm:p-5"
      style={
        {
          "--ds-ml-dot-active": "rgba(251, 191, 36, 1)",
          "--ds-ml-dot-inactive": "rgba(71, 85, 105, 0.9)",
        } as CSSProperties
      }
    >
      <div className="rounded-xl border border-amber-400/40 bg-gradient-to-r from-slate-950/90 via-amber-950/80 to-amber-900/80 p-2.5 shadow-[0_0_24px_-12px_rgba(251,191,36,0.55)] sm:p-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/20 text-amber-200">
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
              <path d="M9 18h6" />
              <path d="M10 22h4" />
              <path d="M12 2a7 7 0 0 0-4 12c1 .7 1.5 1.7 1.5 3h5c0-1.3.5-2.3 1.5-3a7 7 0 0 0-4-12Z" />
            </svg>
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-bold leading-[22px] text-white">{title}</h3>
            <span className="text-[11px] text-amber-100/80">Model deltas</span>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-amber-400/30 bg-slate-950/60 p-3 shadow-[0_0_18px_-12px_rgba(251,191,36,0.35)] sm:p-3.5">
        <div className="flex items-center justify-between gap-sm">
          {items.map((item) => (
            <div key={item.label} className="flex flex-col items-center gap-2">
              <span
                className={`h-3 w-3 rounded-full ${
                  item.active ? "bg-ml-dot-active" : "bg-ml-dot-inactive"
                }`}
              />
              <span className="text-[10px] font-semibold text-amber-100/70">
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
