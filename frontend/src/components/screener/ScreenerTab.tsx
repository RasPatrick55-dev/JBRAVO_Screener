import { useEffect, useMemo, useState } from "react";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import BacktestResultsCard from "./BacktestResultsCard";
import LogsBoard from "./LogsBoard";
import MetricsResultsCard from "./MetricsResultsCard";
import ScreenerPicksCard from "./ScreenerPicksCard";

type ScreenerTabProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

type ScreenerSectionKey = "picks" | "backtest" | "metrics" | "logs";

const initialSectionState: Record<ScreenerSectionKey, LiveDataSyncState> = {
  picks: "loading",
  backtest: "loading",
  metrics: "loading",
  logs: "loading",
};

export default function ScreenerTab({ onSyncStateChange }: ScreenerTabProps) {
  const [sectionState, setSectionState] = useState<Record<ScreenerSectionKey, LiveDataSyncState>>(
    initialSectionState
  );

  const updateSectionState = (key: ScreenerSectionKey, state: LiveDataSyncState) => {
    setSectionState((previous) => (previous[key] === state ? previous : { ...previous, [key]: state }));
  };

  const syncState = useMemo<LiveDataSyncState>(() => {
    const values = Object.values(sectionState);
    if (values.includes("error")) {
      return "error";
    }
    if (values.every((value) => value === "ready")) {
      return "ready";
    }
    return "loading";
  }, [sectionState]);

  useEffect(() => {
    if (onSyncStateChange) {
      onSyncStateChange(syncState);
    }
  }, [onSyncStateChange, syncState]);

  return (
    <section className="space-y-3 sm:space-y-4" aria-label="Screener tab">
      <ScreenerPicksCard onSyncStateChange={(state) => updateSectionState("picks", state)} />
      <BacktestResultsCard onSyncStateChange={(state) => updateSectionState("backtest", state)} />
      <MetricsResultsCard onSyncStateChange={(state) => updateSectionState("metrics", state)} />
      <LogsBoard onSyncStateChange={(state) => updateSectionState("logs", state)} />
    </section>
  );
}
