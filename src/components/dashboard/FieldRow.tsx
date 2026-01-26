import type { ReactNode } from "react";

interface FieldRowProps {
  label: string;
  value: ReactNode;
  valueClassName?: string;
  className?: string;
}

export default function FieldRow({
  label,
  value,
  valueClassName = "",
  className = "",
}: FieldRowProps) {
  return (
    <div className={`flex items-center justify-between gap-2 ${className}`.trim()}>
      <span className="text-[12px] font-bold uppercase tracking-[0.08em] text-[rgb(107,114,128)]">
        {label}
      </span>
      <span
        className={`text-[14px] font-normal leading-[20px] text-[rgb(10,10,10)] ${valueClassName}`.trim()}
      >
        {value}
      </span>
    </div>
  );
}
