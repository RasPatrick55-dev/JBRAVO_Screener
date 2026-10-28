import type { CSSProperties } from "react";

export interface MarketClockGroupProps {
  serverUtc: string;
  market?: MarketClockRecord | null;
}

export type MarketClockRecord = { status?: string; is_open?: boolean | null; observed_at?: string | null; source?: string };
type MarketPhase = "Market Open" | "Closed" | "Unknown" | "Stale";
export function marketPhase(market: MarketClockRecord | null | undefined, now = Date.now()): MarketPhase {
  if (!market || market.status === "unavailable") return "Unknown";
  const observed = Date.parse(market.observed_at ?? "");
  if (market.status === "stale" || (Number.isFinite(observed) && now - observed > 60000)) return "Stale";
  if (market.status !== "ready" || !Number.isFinite(observed) || observed - now > 5000 || typeof market.is_open !== "boolean") return "Unknown";
  return market.is_open ? "Market Open" : "Closed";
}

const PLACEHOLDER_TIME = "--:--:--";
const PLACEHOLDER_DATE = "--- --, ----";

const TIME_FORMATTERS = {
  utc: new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }),
  ny: new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }),
  chicago: new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }),
};

const TIME_FORMATTERS_MOBILE = {
  utc: new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }),
  ny: new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }),
  chicago: new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }),
};

const NY_PARTS_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

const NY_DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  month: "short",
  day: "2-digit",
  year: "numeric",
});

const NY_DATE_FORMATTER_MOBILE = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  month: "short",
  day: "2-digit",
});

const PHASE_STYLES: Record<MarketPhase, { status: string; glow: string }> = {
  "Market Open": { status: "text-[#34D399]", glow: "rgba(52,211,153,0.65)" },
  Closed: { status: "text-[#B8C0CC]", glow: "rgba(184,192,204,0.45)" },
  Unknown: { status: "text-amber-300", glow: "rgba(245,158,11,0.4)" },
  Stale: { status: "text-amber-300", glow: "rgba(245,158,11,0.4)" },
};

const Divider = () => (
  <div className="h-[28px] w-px self-center bg-[#2B3447]" aria-hidden="true" />
);

const LABEL_STYLE: CSSProperties = {
  fontFamily: "Cousine, ui-monospace, SFMono-Regular, Menlo, monospace",
  fontSize: 9,
  fontWeight: 700,
  textTransform: "uppercase",
  lineHeight: "13.5px",
  letterSpacing: "0.9px",
  wordWrap: "break-word",
};

const TIME_STYLE: CSSProperties = {
  fontFamily: "Cousine, ui-monospace, SFMono-Regular, Menlo, monospace",
  fontSize: 14,
  fontWeight: 700,
  lineHeight: "14px",
  wordWrap: "break-word",
};

interface ClockChipProps {
  label: string;
  value: string;
  accentClass: string;
  labelClass: string;
  timeClass: string;
}

const ClockChip = ({ label, value, accentClass, labelClass, timeClass }: ClockChipProps) => (
  <div
    className={
      "flex h-[36px] min-w-[124px] flex-col justify-center gap-[2px] bg-transparent " +
      "pl-[8px] pr-[10px] " +
      "border-l-[5px] " +
      accentClass
    }
  >
    <span
      className={
        "break-words " + labelClass
      }
      style={LABEL_STYLE}
    >
      {label}
    </span>
    <span
      className={
        "break-words tabular-nums whitespace-nowrap " + timeClass
      }
      style={TIME_STYLE}
    >
      {value}
    </span>
  </div>
);

export default function MarketClockGroup({ serverUtc, market }: MarketClockGroupProps) {
  const timestamp = new Date(serverUtc);
  const isValid = !Number.isNaN(timestamp.getTime());

  const utcTime = isValid ? TIME_FORMATTERS.utc.format(timestamp) : PLACEHOLDER_TIME;
  const nyTime = isValid ? TIME_FORMATTERS.ny.format(timestamp) : PLACEHOLDER_TIME;
  const chicagoTime = isValid ? TIME_FORMATTERS.chicago.format(timestamp) : PLACEHOLDER_TIME;
  const utcTimeMobile = isValid ? TIME_FORMATTERS_MOBILE.utc.format(timestamp) : PLACEHOLDER_TIME;
  const nyTimeMobile = isValid ? TIME_FORMATTERS_MOBILE.ny.format(timestamp) : PLACEHOLDER_TIME;
  const chicagoTimeMobile = isValid
    ? TIME_FORMATTERS_MOBILE.chicago.format(timestamp)
    : PLACEHOLDER_TIME;

  const phase = marketPhase(market);
  const dateLabel = isValid
    ? NY_DATE_FORMATTER.format(timestamp).toUpperCase()
    : PLACEHOLDER_DATE;
  const dateLabelMobile = isValid
    ? NY_DATE_FORMATTER_MOBILE.format(timestamp).toUpperCase()
    : PLACEHOLDER_DATE;
  const phaseLabel = phase.toUpperCase();
  const phaseStyles = PHASE_STYLES[phase];
  const phaseGlowStyle = { "--market-glow": phaseStyles.glow } as CSSProperties;

  const clockChips = [
    {
      label: "UTC",
      value: utcTime,
      accentClass: "border-l-[#00C6E8]",
      labelClass: "text-[#00D3F2]",
      timeClass: "text-[#CEFAFE]",
    },
    {
      label: "ET",
      value: nyTime,
      accentClass: "border-l-[#3B4454]",
      labelClass: "text-[#9AA4B2]",
      timeClass: "text-[#F8FAFC]",
    },
    {
      label: "CT",
      value: chicagoTime,
      accentClass: "border-l-[#FF7A00]",
      labelClass: "text-[#FF9B45]",
      timeClass: "text-[#FFE4C2]",
    },
  ];

  return (
    <div className="flex items-center" title={`Market source: ${market?.source ?? "unknown"}; observed UTC: ${market?.observed_at ?? "unknown"}`}>
      <div className="hidden items-stretch overflow-hidden rounded-[12px] border border-[#2B3447] bg-[linear-gradient(135deg,#0B1220_0%,#111827_100%)] shadow-[0_0_0_1px_rgba(36,44,58,0.6)] sm:flex">
        <div className="flex min-w-[140px] flex-col justify-center gap-[2px] px-[12px]">
          <span className="font-cousine text-[9px] font-bold uppercase tracking-[0.3em] text-[#9AA4B2] whitespace-nowrap leading-[12px]">
            {dateLabel}
          </span>
          <span
            className={
              "font-cousine text-[16px] font-bold uppercase tracking-[0.18em] whitespace-nowrap leading-[16px] market-status " +
              phaseStyles.status
            }
            style={phaseGlowStyle}
          >
            {phaseLabel}
          </span>
        </div>
        <div className="flex items-stretch">
          {clockChips.map((clock, index) => (
            <div key={clock.label} className="inline-flex items-stretch">
              <ClockChip
                label={clock.label}
                value={clock.value}
                accentClass={clock.accentClass}
                labelClass={clock.labelClass}
                timeClass={clock.timeClass}
              />
              {index < clockChips.length - 1 ? <Divider /> : null}
            </div>
          ))}
        </div>
      </div>
      <div className="flex w-full flex-col gap-2 rounded-[12px] border border-[#2B3447] bg-[linear-gradient(135deg,#0B1220_0%,#111827_100%)] p-2 shadow-[0_0_0_1px_rgba(36,44,58,0.6)] sm:hidden">
        <div className="flex items-center justify-between gap-3">
          <span className="font-cousine text-[9px] font-bold uppercase tracking-[0.3em] text-[#9AA4B2] whitespace-nowrap leading-[12px]">
            {dateLabelMobile}
          </span>
          <span
            className={
              "font-cousine text-[12px] font-bold uppercase tracking-[0.22em] whitespace-nowrap leading-[14px] market-status " +
              phaseStyles.status
            }
            style={phaseGlowStyle}
          >
            {phaseLabel}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: "UTC", value: utcTimeMobile, accent: "border-[#00C6E8]", tone: "text-[#00D3F2]" },
            { label: "ET", value: nyTimeMobile, accent: "border-[#3B4454]", tone: "text-[#9AA4B2]" },
            { label: "CT", value: chicagoTimeMobile, accent: "border-[#FF7A00]", tone: "text-[#FF9B45]" },
          ].map((clock) => (
            <div
              key={clock.label}
              className={
                "rounded-[10px] border " +
                clock.accent +
                " bg-[#0F172A] px-2 py-1 text-[11px] font-semibold text-slate-100"
              }
            >
              <div className={"text-[9px] font-bold uppercase tracking-[0.2em] " + clock.tone}>
                {clock.label}
              </div>
              <div className="font-cousine text-[11px] font-bold text-slate-100">
                {clock.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
