import BacktestResultsCard from "./BacktestResultsCard";
import LogsBoard from "./LogsBoard";
import MetricsResultsCard from "./MetricsResultsCard";
import ScreenerPicksCard from "./ScreenerPicksCard";

export default function ScreenerTab() {
  return (
    <section className="space-y-3 sm:space-y-4" aria-label="Screener tab">
      <ScreenerPicksCard />
      <BacktestResultsCard />
      <MetricsResultsCard />
      <LogsBoard />
    </section>
  );
}
