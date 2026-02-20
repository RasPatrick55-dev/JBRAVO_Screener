import { useEffect, useMemo, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import LogsPanel from "./LogsPanel";

type LogsBoardProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

type LogPanelKey = "screener" | "backtest" | "metrics";

const initialPanelState: Record<LogPanelKey, LiveDataSyncState> = {
  screener: "loading",
  backtest: "loading",
  metrics: "loading",
};

export default function LogsBoard({ onSyncStateChange }: LogsBoardProps) {
  const [panelState, setPanelState] = useState<Record<LogPanelKey, LiveDataSyncState>>(
    initialPanelState
  );

  const updatePanelState = (key: LogPanelKey, state: LiveDataSyncState) => {
    setPanelState((previous) => (previous[key] === state ? previous : { ...previous, [key]: state }));
  };

  const boardState = useMemo<LiveDataSyncState>(() => {
    const values = Object.values(panelState);
    if (values.includes("error")) {
      return "error";
    }
    if (values.every((value) => value === "ready")) {
      return "ready";
    }
    return "loading";
  }, [panelState]);

  useEffect(() => {
    if (onSyncStateChange) {
      onSyncStateChange(boardState);
    }
  }, [boardState, onSyncStateChange]);

  return (
    <section className="space-y-2" aria-label="Live logs board">
      <header className="flex items-center justify-between gap-3">
        <h2 className="font-arimo text-xs font-semibold uppercase tracking-[0.08em] text-primary sm:text-sm">
          Live Logs Board
        </h2>
      </header>

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        <LogsPanel
          title="Screener Logs"
          stage="screener"
          onSyncStateChange={(state) => updatePanelState("screener", state)}
        />
        <LogsPanel
          title="Backtest Logs"
          stage="backtest"
          onSyncStateChange={(state) => updatePanelState("backtest", state)}
        />
        <LogsPanel
          title="Metrics Logs"
          stage="metrics"
          onSyncStateChange={(state) => updatePanelState("metrics", state)}
        />
      </div>
    </section>
  );
}
