import { useCallback, useEffect, useRef, useState } from "react";
import { pollRead, readJson } from "./readLifecycle";

type ReadState<T> = {
  data: T | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: string | null;
  receivedAt?: string;
};
const initial = <T,>(): ReadState<T> => ({ data: null, isLoading: true, isRefreshing: true, error: null });

// Parser identity must be stable (declare it at module scope). New query paths
// have no inherited rows/receipts; a failed same-query refresh keeps its sample.
export function usePollingRead<T>(path: string, parse: (value: unknown) => T) {
  const [state, setState] = useState<{ path: string; value: ReadState<T> }>(() => ({ path, value: initial<T>() }));
  const refreshRef = useRef<(() => void) | null>(null);
  const refresh = useCallback(() => refreshRef.current?.(), []);
  useEffect(() => {
    let current = initial<T>();
    const publish = (value: ReadState<T>) => { current = value; setState({ path, value }); };
    publish(current);
    const polling = pollRead({
      read: async signal => parse(await readJson(path, signal)),
      start: () => publish({ ...current, isRefreshing: true }),
      success: data => publish({ data, error: null, isLoading: false, isRefreshing: false, receivedAt: new Date().toISOString() }),
      failure: error => publish({ ...current, isLoading: false, isRefreshing: false,
        error: error instanceof Error ? error.message : "Source unavailable" }),
    });
    refreshRef.current = polling.refresh;
    return () => { refreshRef.current = null; polling.stop(); };
  }, [path, parse]);
  return { ...(state.path === path ? state.value : initial<T>()), refresh };
}

export function refreshNotice(error: string | null, receivedAt?: string) {
  return error && receivedAt ? `${error} Showing stale data from ${receivedAt}; current status is unverified.` : error;
}
