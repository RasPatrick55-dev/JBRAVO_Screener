import type { StatusBadgeProps, StatusTone } from "../../types/ui";

const toneStyles: Record<StatusTone, string> = {
  success: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  warning: "bg-amber-50 text-amber-700 ring-amber-200",
  error: "bg-rose-50 text-rose-700 ring-rose-200",
  info: "bg-sky-50 text-sky-700 ring-sky-200",
  neutral: "bg-slate-100 text-slate-600 ring-slate-200",
};

const toneDots: Record<StatusTone, string> = {
  success: "bg-emerald-500",
  warning: "bg-amber-500",
  error: "bg-rose-500",
  info: "bg-sky-500",
  neutral: "bg-slate-400",
};

const sizeStyles = {
  sm: "px-2 py-0.5 text-[11px]",
  md: "px-2.5 py-1 text-xs",
};

export default function StatusBadge({
  label,
  tone,
  showDot = false,
  size = "md",
  className = "",
}: StatusBadgeProps) {
  return (
    <span
      className={
        "inline-flex items-center gap-1.5 rounded-full font-semibold ring-1 ring-inset " +
        toneStyles[tone] +
        " " +
        sizeStyles[size] +
        (className ? " " + className : "")
      }
    >
      {showDot ? <span className={`h-1.5 w-1.5 rounded-full ${toneDots[tone]}`} /> : null}
      {label}
    </span>
  );
}
