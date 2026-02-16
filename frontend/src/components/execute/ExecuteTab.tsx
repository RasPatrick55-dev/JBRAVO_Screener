import { useEffect, useRef, useState } from "react";
import ExecutionLogsCard from "./ExecutionLogsCard";
import ExecutionSummaryCard from "./ExecutionSummaryCard";
import OrdersTableCard from "./OrdersTableCard";
import TrailingStopsCard from "./TrailingStopsCard";
import type { ExecuteStateResponse } from "./types";
import { fetchJsonNoStore, parseSseJson } from "./utils";

export default function ExecuteTab() {
  const [payload, setPayload] = useState<ExecuteStateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const hasLoadedOnceRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const applyPayload = (nextPayload: ExecuteStateResponse | null) => {
      if (!isMounted) {
        return;
      }
      setPayload(nextPayload);
      setHasError(!nextPayload);
      setIsLoading(false);
      if (nextPayload) {
        hasLoadedOnceRef.current = true;
      }
    };

    const loadFallback = async () => {
      setIsLoading(!hasLoadedOnceRef.current);
      const fallback = await fetchJsonNoStore<ExecuteStateResponse>(`/api/execute/state?ts=${Date.now()}`);
      applyPayload(fallback);
    };

    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      void loadFallback();
      return () => {
        isMounted = false;
      };
    }

    setIsLoading(!hasLoadedOnceRef.current);
    const source = new EventSource("/api/execute/state/stream");

    const fallbackTimer = window.setTimeout(() => {
      if (!hasLoadedOnceRef.current) {
        void loadFallback();
      }
    }, 2500);

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
      if (!hasLoadedOnceRef.current) {
        window.clearTimeout(fallbackTimer);
        void loadFallback();
      }
    };

    return () => {
      isMounted = false;
      window.clearTimeout(fallbackTimer);
      source.close();
    };
  }, []);

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
      <ExecutionSummaryCard summary={summary} isLoading={isLoading} hasError={hasError} />
      <OrdersTableCard rows={ordersRows} isLoading={isLoading} hasError={hasError} />
      <TrailingStopsCard rows={trailingRows} isLoading={isLoading} hasError={hasError} />
      <ExecutionLogsCard logs={logsByStage} isLoading={isLoading} hasError={hasError} />
    </div>
  );
}
