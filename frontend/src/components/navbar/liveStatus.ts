import { useEffect, useState } from "react";
import type { NavbarDesktopProps, StatusTone } from "../../types/ui";

type ApiHealthResponse = {
  trading_ok?: boolean | null;
};

type TradingStatus = boolean | null;

const POLL_INTERVAL_MS = 15_000;

const liveToneFromStatus = (status: TradingStatus): StatusTone => {
  if (status === true) {
    return "success";
  }
  if (status === false) {
    return "error";
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
  tradingStatus: TradingStatus
): NavbarDesktopProps["rightBadges"] => [
  { label: "Paper Trading", tone: "warning" as const, showDot: true },
  {
    label: "Live",
    tone: liveToneFromStatus(tradingStatus),
    showDot: tradingStatus === true,
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
