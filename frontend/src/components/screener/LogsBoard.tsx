import LogsPanel from "./LogsPanel";

export default function LogsBoard() {
  return (
    <section className="space-y-2" aria-label="Live logs board">
      <header className="flex items-center justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">
          Live Logs Board
        </h2>
      </header>

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        <LogsPanel title="Screener Logs" stage="screener" />
        <LogsPanel title="Backtest Logs" stage="backtest" />
        <LogsPanel title="Metrics Logs" stage="metrics" />
      </div>
    </section>
  );
}
