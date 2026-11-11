// Test fixture: ExecuteTab lifecycle from canonical ab52f68 (pre-repair).
// Only import paths are adapted. Not imported by the production app.
import { useEffect, useRef, useState } from "react";
import ExecutionLogsCard from "../../src/components/execute/ExecutionLogsCard";
import ExecutionSummaryCard from "../../src/components/execute/ExecutionSummaryCard";
import OrdersTableCard from "../../src/components/execute/OrdersTableCard";
import TrailingStopsCard from "../../src/components/execute/TrailingStopsCard";
import type { ExecuteStateResponse } from "../../src/components/execute/types";
import { fetchJsonNoStore, parseSseJson } from "../../src/components/execute/utils";
import type { LiveDataSyncState } from "../../src/components/navbar/liveStatus";
import { useSource } from "../../src/components/ui/sourceState";
import type { ProcessReceipt } from "../../src/components/dashboard/processReceipt";

const REFRESH_INTERVAL_MS = 20_000;
const FIRST_LOAD_GRACE_MS = 2_500;
const REQUEST_TIMEOUT_MS = 15_000;

const hasUsableState = (value: ExecuteStateResponse | null): value is ExecuteStateResponse => {
  if (!value || typeof value !== "object") {
    return false;
  }
  return Boolean(value.summary || value.orders || value.trailing_stops || value.logs);
};

type ExecuteTabProps = {
  onSyncStateChange?: (state: LiveDataSyncState) => void;
};

export default function ExecuteTab({ onSyncStateChange }: ExecuteTabProps) {
  const executionTask = useSource("/api/execute/task");
  const [payload, setPayload] = useState<ExecuteStateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const hasLoadedOnceRef = useRef(false);
  const fallbackInFlightRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const applyPayload = (nextPayload: ExecuteStateResponse | null) => {
      if (!isMounted) {
        return;
      }

      if (hasUsableState(nextPayload)) {
        setPayload(nextPayload);
        setHasError(false);
        hasLoadedOnceRef.current = true;
      } else {
        if (!hasLoadedOnceRef.current) {
          setPayload(null);
        }
        setHasError(true);
      }
      setIsLoading(false);
    };

    const loadFallback = async () => {
      if (fallbackInFlightRef.current) {
        return;
      }
      fallbackInFlightRef.current = true;
      setIsLoading(!hasLoadedOnceRef.current);
      const fallback = await fetchJsonNoStore<ExecuteStateResponse>(
        `/api/execute/state?ts=${Date.now()}`,
        REQUEST_TIMEOUT_MS
      );
      applyPayload(fallback);
      fallbackInFlightRef.current = false;
    };

    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      void loadFallback();
      const intervalId = window.setInterval(() => {
        void loadFallback();
      }, REFRESH_INTERVAL_MS);
      return () => {
        isMounted = false;
        window.clearInterval(intervalId);
      };
    }

    setIsLoading(!hasLoadedOnceRef.current);
    const source = new EventSource("/api/execute/state/stream");

    const fallbackTimer = window.setTimeout(() => {
      if (!hasLoadedOnceRef.current) {
        void loadFallback();
      }
    }, FIRST_LOAD_GRACE_MS);
    const pollTimer = window.setInterval(() => {
      void loadFallback();
    }, REFRESH_INTERVAL_MS);

    source.onmessage = (event) => {
      const parsed = parseSseJson<ExecuteStateResponse>(event.data);
      if (!parsed) {
        return;
      }
      window.clearTimeout(fallbackTimer);
      applyPayload(parsed);
    };
    source.onerror = () => {
      if (!isMounted) {
        return;
      }
      window.clearTimeout(fallbackTimer);
      void loadFallback();
    };

    return () => {
      isMounted = false;
      window.clearTimeout(fallbackTimer);
      window.clearInterval(pollTimer);
      source.close();
    };
  }, []);

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
      <ExecutionSummaryCard summary={summary} isLoading={isLoading} hasError={hasError}
        lastScheduledAttempt={(executionTask.payload as ProcessReceipt | null)?.last_scheduled_attempt}
        lastExecutionReceipt={(executionTask.payload as ProcessReceipt | null)?.last_execution_receipt}
        taskSource={executionTask} />
      <OrdersTableCard rows={ordersRows} isLoading={isLoading} hasError={hasError} />
      <TrailingStopsCard rows={trailingRows} isLoading={isLoading} hasError={hasError} />
      <ExecutionLogsCard logs={logsByStage} isLoading={isLoading} hasError={hasError} />
    </div>
  );
}
