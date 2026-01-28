import type { CSSProperties } from "react";

export interface MarketClockGroupProps {
  serverUtc: string;
}

type MarketPhase = "Pre-Market" | "Market Open" | "After-Hours" | "Closed";

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
  "Pre-Market": {
    status: "text-[#7DD3FC]",
    glow: "rgba(125, 211, 252, 0.65)",
  },
  "Market Open": {
    status: "text-[#34D399]",
    glow: "rgba(52, 211, 153, 0.65)",
  },
  "After-Hours": {
    status: "text-[#C084FC]",
    glow: "rgba(192, 132, 252, 0.6)",
  },
  Closed: {
    status: "text-[#B8C0CC]",
    glow: "rgba(184, 192, 204, 0.45)",
  },
};

const getNyTimeParts = (date: Date) => {
  const parts = NY_PARTS_FORMATTER.formatToParts(date);
  const readPart = (type: string) =>
    Number(parts.find((part) => part.type === type)?.value ?? "0");
  return {
    hour: readPart("hour"),
    minute: readPart("minute"),
    second: readPart("second"),
  };
};

const getMarketPhase = (hour: number, minute: number, second: number): MarketPhase => {
  const minutes = hour * 60 + minute + second / 60;
  if (minutes >= 4 * 60 && minutes < 9 * 60 + 30) {
    return "Pre-Market";
  }
  if (minutes >= 9 * 60 + 30 && minutes < 16 * 60) {
    return "Market Open";
  }
  if (minutes >= 16 * 60 && minutes < 20 * 60) {
    return "After-Hours";
  }
  return "Closed";
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

export default function MarketClockGroup({ serverUtc }: MarketClockGroupProps) {
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

  const nyParts = isValid ? getNyTimeParts(timestamp) : { hour: 0, minute: 0, second: 0 };
  const phase = isValid
    ? getMarketPhase(nyParts.hour, nyParts.minute, nyParts.second)
    : "Closed";
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
      label: "EST",
      value: nyTime,
      accentClass: "border-l-[#3B4454]",
      labelClass: "text-[#9AA4B2]",
      timeClass: "text-[#F8FAFC]",
    },
    {
      label: "CST",
      value: chicagoTime,
      accentClass: "border-l-[#FF7A00]",
      labelClass: "text-[#FF9B45]",
      timeClass: "text-[#FFE4C2]",
    },
  ];

  return (
    <div className="flex items-center">
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
            { label: "EST", value: nyTimeMobile, accent: "border-[#3B4454]", tone: "text-[#9AA4B2]" },
            { label: "CST", value: chicagoTimeMobile, accent: "border-[#FF7A00]", tone: "text-[#FF9B45]" },
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
