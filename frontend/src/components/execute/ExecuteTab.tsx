import ExecutionLogsCard from "./ExecutionLogsCard";
import ExecutionSummaryCard from "./ExecutionSummaryCard";
import OrdersTableCard from "./OrdersTableCard";
import TrailingStopsCard from "./TrailingStopsCard";

export default function ExecuteTab() {
  return (
    <div className="space-y-4">
      <ExecutionSummaryCard />
      <OrdersTableCard />
      <TrailingStopsCard />
      <ExecutionLogsCard />
    </div>
  );
}
