import { useEffect, useState } from "react";
import type { NavbarDesktopProps, StatusTone } from "../../types/ui";
import { deadlineRead } from "../ui/sourceState";

export type ApiHealthResponse = {
  freshness?: {age_seconds?: number | null; freshness_level?: string | null};
  trading_ok?: boolean | null;
  connection_check?: {
    source?: string | null; status?: string; reason?: string; checked_at?: string | null;
    trading_ok?: boolean | null;
    freshness?: {age_seconds?: number | null; freshness_level?: string | null};
  };
};

type TradingStatus = boolean | null;
export type LiveDataSyncState = "loading" | "ready" | "error";

const POLL_INTERVAL_MS = 15_000;
const REQUEST_TIMEOUT_MS = 5_000;
const MAX_CHECK_AGE_MS = 7_200_000;

const liveToneFromStatus = (status: TradingStatus): StatusTone => {
  if (status === false) {
    return "error";
  }
  if (status === true) {
    return "success";
  }
  return "neutral";
};

const checkedMilliseconds = (value: unknown): number | null => {
  if (typeof value !== "string" || value.length > 128 ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return null;
  const [year,month,day,hour,minute,second] = value.slice(0,19).split(/[-T:]/).map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (month < 1 || month > 12 || day < 1 || day > [31,leap?29:28,31,30,31,30,31,31,30,31,30,31][month-1] || hour > 23 || minute > 59 || second > 59) return null;
  const stamp = Date.parse(value);
  return Number.isFinite(stamp) ? stamp : null;
};

export const diagnosticsFresh = (payload: ApiHealthResponse | null, now=Date.now()): boolean => {
  const check = payload?.connection_check, age = check?.freshness?.age_seconds;
  const stamp = checkedMilliseconds(check?.checked_at);
  return !!check && ["connection_health.json","connectivity.json"].includes(check.source ?? "") && check.status === "fresh" &&
    check.freshness?.freshness_level === "green" && typeof age === "number" && Number.isFinite(age) && age >= 0 && age < 7200 &&
    stamp !== null && Number.isFinite(now) && stamp <= now && now - stamp < MAX_CHECK_AGE_MS;
};

export const parseTradingStatus = (payload: ApiHealthResponse | null, now=Date.now()): TradingStatus => {
  if (!diagnosticsFresh(payload,now)) {
    return null;
  }
  const result = payload?.connection_check?.trading_ok;
  return typeof result === "boolean" ? result : null;
};

export function watchTradingStatus(publish: (value: TradingStatus)=>void, transport: typeof fetch=fetch) {
  let active=true, inFlight=false;
  let payload: ApiHealthResponse | null=null;
  let last: TradingStatus | undefined;
  const emit=(value:TradingStatus)=>{if(active && value !== last){last=value;publish(value);}};
  emit(null);
  const load=async()=>{
    if(!active || inFlight)return;
    inFlight=true;
    try {
      const next=await deadlineRead(async signal=>{
        const response=await transport("/api/health",{cache:"no-store",headers:{Accept:"application/json"},signal});
        if(!response.ok)throw new Error("stored_check_unavailable");
        return await response.json() as ApiHealthResponse;
      },REQUEST_TIMEOUT_MS);
      if(active){payload=next;emit(parseTradingStatus(payload));}
    }catch{payload=null;emit(null);}
    finally{inFlight=false;}
  };
  void load();
  const poll=globalThis.setInterval(()=>void load(),POLL_INTERVAL_MS);
  const expiry=globalThis.setInterval(()=>emit(parseTradingStatus(payload)),1000);
  return ()=>{active=false;globalThis.clearInterval(poll);globalThis.clearInterval(expiry);};
};

export const buildNavbarBadges = (
  tradingStatus: TradingStatus,
  _syncState: LiveDataSyncState | null = null
): NavbarDesktopProps["rightBadges"] => [
  { label: "Paper Trading", tone: "warning" as const, showDot: true },
  {
    label: tradingStatus === null ? "API check unknown/stale" : "Stored API check",
    tone: liveToneFromStatus(tradingStatus),
    showDot: tradingStatus === true,
  },
];

export const useLiveTradingStatus = (_initialStatus: TradingStatus = null): TradingStatus => {
  // A legacy boolean seed has no timestamp. Wait for checked evidence instead.
  const [tradingStatus, setTradingStatus] = useState<TradingStatus>(null);
  useEffect(() => watchTradingStatus(setTradingStatus), []);

  return tradingStatus;
};
