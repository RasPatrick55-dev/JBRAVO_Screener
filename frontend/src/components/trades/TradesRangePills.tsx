import type { RangeKey } from "./types";

const baseRanges: Array<{ key: Exclude<RangeKey, "all">; label: string }> = [
  { key: "d", label: "D" },
  { key: "w", label: "W" },
  { key: "m", label: "M" },
  { key: "y", label: "Y" },
];

interface TradesRangePillsProps {
  value: RangeKey;
  onChange: (next: RangeKey) => void;
  allLabel?: "A" | "ALL";
  className?: string;
}

export default function TradesRangePills({
  value,
  onChange,
  allLabel = "ALL",
  className = "",
}: TradesRangePillsProps) {
  const ranges: Array<{ key: RangeKey; label: string }> = [...baseRanges, { key: "all", label: allLabel }];

  return (
    <div className={`inline-flex items-center gap-2 ${className}`.trim()} role="group" aria-label="Range selection">
      {ranges.map((range) => {
        const isSelected = range.key === value;
        return (
          <button
            key={range.key}
            type="button"
            aria-pressed={isSelected}
            onClick={() => onChange(range.key)}
            className={
              "font-arimo inline-flex min-w-11 items-center justify-center rounded-lg px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] transition " +
              (isSelected
                ? "bg-amber-50 text-amber-700 outline outline-1 outline-offset-[-1px] outline-amber-300 dark:bg-amber-500/20 dark:text-amber-200 dark:outline-amber-300/45"
                : "bg-surface text-secondary outline outline-1 outline-offset-[-1px] outline-subtle hover:text-primary")
            }
          >
            {range.label}
          </button>
        );
      })}
    </div>
  );
}
