import { useEffect, useState } from "react";
import type { NavbarDesktopProps, StatusTone } from "../../types/ui";

type ApiHealthResponse = {
  trading_ok?: boolean | null;
};

type TradingStatus = boolean | null;
export type LiveDataSyncState = "loading" | "ready" | "error";

const POLL_INTERVAL_MS = 15_000;

const liveToneFromStatus = (status: TradingStatus, syncState: LiveDataSyncState | null): StatusTone => {
  if (status === false) {
    return "error";
  }
  if (status === true) {
    if (syncState === "error") {
      return "warning";
    }
    if (syncState === "loading") {
      return "neutral";
    }
    return "success";
  }
  return "neutral";
};

const parseTradingStatus = (payload: ApiHealthResponse | null): TradingStatus => {
  if (!payload) {
    return null;
  }
  return typeof payload.trading_ok === "boolean" ? payload.trading_ok : null;
};

const fetchTradingStatus = async (): Promise<TradingStatus> => {
  try {
    const response = await fetch("/api/health", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as ApiHealthResponse;
    return parseTradingStatus(payload);
  } catch {
    return null;
  }
};

export const buildNavbarBadges = (
  tradingStatus: TradingStatus,
  syncState: LiveDataSyncState | null = null
): NavbarDesktopProps["rightBadges"] => [
  { label: "Paper Trading", tone: "warning" as const, showDot: true },
  {
    label: "Live",
    tone: liveToneFromStatus(tradingStatus, syncState),
    showDot: tradingStatus === true && syncState !== "loading" && syncState !== "error",
  },
];

export const useLiveTradingStatus = (initialStatus: TradingStatus = null): TradingStatus => {
  const [tradingStatus, setTradingStatus] = useState<TradingStatus>(initialStatus);

  useEffect(() => {
    if (typeof initialStatus === "boolean") {
      setTradingStatus(initialStatus);
    }
  }, [initialStatus]);

  useEffect(() => {
    let isMounted = true;
    let inFlight = false;

    const load = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const next = await fetchTradingStatus();
        if (!isMounted) {
          return;
        }
        // Preserve the last known status on transient fetch failures.
        setTradingStatus((previous) => (next === null ? previous : next));
      } finally {
        inFlight = false;
      }
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  return tradingStatus;
};
