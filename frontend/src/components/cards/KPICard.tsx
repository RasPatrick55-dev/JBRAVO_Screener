import type { KPICardProps, StatusTone } from "../../types/ui";

const detailStyles: Record<StatusTone, string> = {
  success: "text-emerald-600",
  warning: "text-amber-600",
  error: "text-rose-600",
  info: "text-sky-600",
  neutral: "text-slate-500",
};

export default function KPICard({
  title,
  value,
  footnote,
  detail,
  detailTone = "neutral",
  icon,
}: KPICardProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100">
            {icon}
          </div>
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            {title}
          </div>
        </div>
      </div>
      <div className="mt-4 text-2xl font-semibold text-slate-900">{value}</div>
      {detail ? (
        <div className={`mt-1 text-sm font-semibold ${detailStyles[detailTone]}`}>{detail}</div>
      ) : null}
      {footnote ? <div className="mt-2 text-xs text-slate-500">{footnote}</div> : null}
    </div>
  );
}
