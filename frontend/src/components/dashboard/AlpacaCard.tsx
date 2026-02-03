import CardShell from "./CardShell";

interface AlpacaCardProps {
  value: string;
  title?: string;
  subtitle?: string;
}

export default function AlpacaCard({
  value,
  title = "Alpaca",
  subtitle = "Equity",
}: AlpacaCardProps) {
  return (
    <CardShell className="border-cyan-400/30 bg-gradient-to-br from-slate-950/95 via-slate-900/85 to-cyan-950/70 p-4 shadow-[0_18px_40px_-24px_rgba(34,211,238,0.5)] sm:p-5">
      <div className="rounded-xl border border-cyan-400/40 bg-gradient-to-r from-slate-950/90 via-slate-900/90 to-cyan-950/80 p-2.5 shadow-[0_0_24px_-12px_rgba(34,211,238,0.6)] sm:p-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/20 text-cyan-200">
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
              <path d="M4 10h16" />
              <path d="M6 10V7a6 6 0 0 1 12 0v3" />
              <rect x="5" y="10" width="14" height="10" rx="2" />
            </svg>
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-bold leading-[22px] text-white">{title}</h3>
            <span className="text-[11px] text-cyan-100/80">{subtitle}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-cyan-400/30 bg-slate-950/60 p-3 shadow-[0_0_18px_-12px_rgba(34,211,238,0.45)] sm:p-3.5">
        <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-cyan-100/70">
          Account Value
        </div>
        <div className="mt-2 font-cousine text-[20px] font-semibold text-cyan-100">
          {value}
        </div>
      </div>
    </CardShell>
  );
}
