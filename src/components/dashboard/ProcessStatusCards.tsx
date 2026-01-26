import ExecuteTradesStatusCard, {
  type ExecuteTradesStatusCardProps,
} from "./ExecuteTradesStatusCard";
import MonitoringPositionsCard, { type MonitoringPositionsCardProps } from "./MonitoringPositionsCard";
import PipelineStatusCard, { type PipelineStatusCardProps } from "./PipelineStatusCard";

export interface ProcessStatusCardsProps {
  pipeline: PipelineStatusCardProps;
  executeTrades: ExecuteTradesStatusCardProps;
  monitoring: MonitoringPositionsCardProps;
}

export const processStatusCardsMock: ProcessStatusCardsProps = {
  pipeline: {
    lastRun: { date: "Jan 23, 2026", start: "08:45 UTC", end: "11:20 UTC", duration: "2h 35m" },
    subprocess: { screener: "OK", backTester: "OK", metrics: "OK" },
    isLive: true,
  },
  executeTrades: {
    lastRun: { date: "Jan 23, 2026", start: "09:02 UTC", end: "09:15 UTC", duration: "13m" },
    ordersPlaced: 4,
    totalValue: 667.46,
    successRate: 100,
    isCycleComplete: true,
    marketNote: "Market closed",
  },
  monitoring: {
    positions: [
      {
        symbol: "NGVC",
        currentPrice: 25.96,
        sparklineData: [24.7, 24.9, 25.1, 25.05, 25.3, 25.6, 25.96],
        percentPL: 9.2,
        dollarPL: 210.24,
      },
      {
        symbol: "AVB",
        currentPrice: 1.84,
        sparklineData: [1.62, 1.7, 1.68, 1.75, 1.78, 1.82, 1.84],
        percentPL: 10.5,
        dollarPL: 457.22,
      },
    ],
  },
};

export default function ProcessStatusCards({
  pipeline,
  executeTrades,
  monitoring,
}: ProcessStatusCardsProps) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row lg:gap-5">
      <div className="flex-1">
        <PipelineStatusCard {...pipeline} />
      </div>
      <div className="flex-1">
        <ExecuteTradesStatusCard {...executeTrades} />
      </div>
      <div className="flex-1">
        <MonitoringPositionsCard {...monitoring} />
      </div>
    </div>
  );
}
