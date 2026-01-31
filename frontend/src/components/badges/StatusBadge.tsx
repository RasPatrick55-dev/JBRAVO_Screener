import type { StatusBadgeProps, StatusTone } from "../../types/ui";

const toneStyles: Record<StatusTone, string> = {
  success: "jbravo-status-success",
  warning:
    "bg-amber-50 text-amber-700 ring-amber-200 dark:bg-amber-500/15 dark:text-amber-200 dark:ring-amber-400/40",
  error: "jbravo-status-error",
  info: "bg-sky-50 text-sky-700 ring-sky-200 dark:bg-sky-500/15 dark:text-sky-200 dark:ring-sky-400/40",
  neutral:
    "bg-slate-100 text-slate-600 ring-slate-200 dark:bg-slate-700/30 dark:text-slate-200 dark:ring-slate-500/40",
};

const toneDots: Record<StatusTone, string> = {
  success: "jbravo-status-dot-success",
  warning: "bg-amber-500",
  error: "jbravo-status-dot-error",
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
