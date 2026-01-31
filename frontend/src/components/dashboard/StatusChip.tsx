import type { ReactNode } from "react";

export type StatusChipTone = "success" | "active" | "error" | "neutral";

const toneStyles: Record<StatusChipTone, string> = {
  success: "jbravo-chip-success",
  active: "text-cyan-200 outline-cyan-400 bg-cyan-500/20",
  error: "jbravo-chip-error",
  neutral: "text-slate-200 outline-slate-400 bg-slate-500/20",
};

interface StatusChipProps {
  label: string;
  tone: StatusChipTone;
  icon?: ReactNode;
  className?: string;
}

export default function StatusChip({ label, tone, icon, className = "" }: StatusChipProps) {
  return (
    <span
      className={
        "inline-flex items-center justify-center gap-1.5 rounded-lg outline outline-1 outline-offset-[-1px] px-2 py-1 text-[11px] font-bold uppercase tracking-[0.08em] " +
        toneStyles[tone] +
        (className ? " " + className : "")
      }
    >
      {icon}
      {label}
    </span>
  );
}
