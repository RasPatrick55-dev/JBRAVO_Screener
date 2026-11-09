import { useEffect } from "react";
import ExecutionLogsCard from "./ExecutionLogsCard";
import ExecutionSummaryCard from "./ExecutionSummaryCard";
import OrdersTableCard from "./OrdersTableCard";
import TrailingStopsCard from "./TrailingStopsCard";
import type { ExecuteStateResponse } from "./types";
import { usePollingRead, refreshNotice } from "../ui/usePollingRead";
import type { LiveDataSyncState } from "../navbar/liveStatus";
import { useSource } from "../ui/sourceState";
import type { ProcessReceipt } from "../dashboard/processReceipt";

const parseState = (value: unknown): ExecuteStateResponse => {
  const payload = value as ExecuteStateResponse | null;
  if (!payload || typeof payload !== "object" || !Boolean(payload.summary || payload.orders || payload.trailing_stops || payload.logs)) {
    throw new Error("Execution state unavailable.");
  }
  return payload;
};

type ExecuteTabProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function ExecuteTab({ onSyncStateChange }: ExecuteTabProps) {
  const executionTask = useSource("/api/execute/task");
  const source = usePollingRead("/api/execute/state", parseState);
  const { data: payload, isLoading } = source;
  const hasError = Boolean(source.error) || (!isLoading && payload?.ok === false);

  useEffect(() => {
    if (!onSyncStateChange) {
      return;
    }
    if (isLoading) {
      onSyncStateChange("loading");
      return;
    }
    if (hasError || !payload) {
      onSyncStateChange("error");
      return;
    }
    onSyncStateChange("ready");
  }, [hasError, isLoading, onSyncStateChange, payload]);

  const summary = payload?.summary ?? null;
  const ordersRows = payload?.orders?.rows ?? [];
  const trailingRows = payload?.trailing_stops?.rows ?? [];
  const logsByStage = {
    execute: payload?.logs?.execute?.rows ?? [],
    monitor: payload?.logs?.monitor?.rows ?? [],
    pipeline: payload?.logs?.pipeline?.rows ?? [],
  };

  return (
    <div className="space-y-4">
      {source.error ? <p role="alert" className="text-xs text-rose-700 dark:text-rose-300">{refreshNotice(source.error, source.receivedAt)}</p> : null}
      <ExecutionSummaryCard summary={summary} isLoading={isLoading} hasError={hasError || (!isLoading && summary?.ok !== true)}
        lastScheduledAttempt={(executionTask.payload as ProcessReceipt | null)?.last_scheduled_attempt}
        lastExecutionReceipt={(executionTask.payload as ProcessReceipt | null)?.last_execution_receipt}
        taskSource={executionTask} />
      <OrdersTableCard rows={ordersRows} isLoading={isLoading} hasError={hasError || (!isLoading && payload?.orders?.ok !== true)} />
      <TrailingStopsCard rows={trailingRows} isLoading={isLoading} hasError={hasError || (!isLoading && payload?.trailing_stops?.ok !== true)} />
      <ExecutionLogsCard logs={logsByStage} isLoading={isLoading} hasError={hasError} />
    </div>
  );
}
