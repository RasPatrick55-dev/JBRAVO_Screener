import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import ProcessStatusCards, {
  type ProcessStatusCardsProps,
} from "../components/dashboard/ProcessStatusCards";
import type { LogEntry, StatusTone, SystemStatusItem } from "../types/ui";

type HealthOverviewResponse = {
  ok?: boolean;
  metrics_summary_present?: boolean;
  trades_log_present?: boolean;
  trades_log_rows?: number | null;
  kpis?: Record<string, number | string | null>;
};

type ApiHealthResponse = {
  trading_ok?: boolean | null;
  trading_status?: number | null;
  data_ok?: boolean | null;
  data_status?: number | null;
  feed?: string | null;
  last_run_utc?: string | null;
  pipeline_rc?: number | null;
  rows_final?: number | null;
  rows_premetrics?: number | null;
  latest_source?: string | null;
  freshness?: {
    age_seconds?: number | null;
    freshness_level?: string | null;
  };
  run_type?: string | null;
  buying_power?: number | null;
};

type AccountOverviewResponse = {
  ok?: boolean;
  snapshot?: AccountSnapshot;
};

type AccountSnapshot = {
  account_id?: string;
  status?: string;
  equity?: number | null;
  cash?: number | null;
  buying_power?: number | null;
  portfolio_value?: number | null;
  taken_at?: string;
  source?: string;
};

type TradesMetrics = {
  total_trades?: number | null;
  win_rate?: number | null;
  net_pnl?: number | null;
  profit_factor?: number | null;
  last_run_utc?: string | null;
};

type TradeRecord = {
  trade_id?: string | number;
  symbol?: string;
  qty?: number | null;
  status?: string;
  entry_time?: string | null;
  entry_price?: number | null;
  exit_time?: string | null;
  exit_price?: number | null;
  realized_pnl?: number | null;
  updated_at?: string | null;
  created_at?: string | null;
};

type TradesOverviewResponse = {
  ok?: boolean;
  metrics?: TradesMetrics;
  trades?: TradeRecord[];
  open_positions?: {
    count?: number | null;
    realized_pnl?: number | null;
  };
};

type ExecutionSnapshot = {
  in_window?: boolean;
  buying_power?: number | null;
  open_positions?: number;
  orders_submitted?: number;
  orders_filled?: number;
  orders_rejected?: number;
  skip_counts?: Record<string, number>;
  last_execution?: string | null;
  ny_now?: string | null;
};

type MonitoringPositionsResponse = {
  ok?: boolean;
  positions?: Array<{
    symbol?: string | null;
    logoUrl?: string | null;
    qty?: number | null;
    entryPrice?: number | null;
    currentPrice?: number | null;
    sparklineData?: number[] | null;
    percentPL?: number | null;
    dollarPL?: number | null;
    costBasis?: number | null;
  }>;
};

type PipelineTaskRunResponse = {
  ok?: boolean;
  started_utc?: string | null;
  finished_utc?: string | null;
  duration_seconds?: number | null;
  rc?: number | null;
  source?: string | null;
};

type ExecuteTaskRunResponse = {
  ok?: boolean;
  started_utc?: string | null;
  finished_utc?: string | null;
  duration_seconds?: number | null;
  rc?: number | null;
  source?: string | null;
};

type ExecuteOrdersSummaryResponse = {
  ok?: boolean;
  orders_filled?: number | null;
  total_value?: number | null;
  since_utc?: string | null;
  until_utc?: string | null;
  source?: string | null;
};

type OpenPositionsSummary = {
  count: number | null;
  pnl: number | null;
  pnlPct: number | null;
  filledCount: number | null;
  totalCount: number;
};

const navLabels = [
  "Dashboard",
  "Account",
  "Trades",
  "Positions",
  "Execute",
  "Screener",
  "ML Pipeline",
];

const statusDotTone: Record<StatusTone, string> = {
  success: "bg-emerald-500",
  warning: "bg-amber-500",
  error: "bg-rose-500",
  info: "bg-sky-500",
  neutral: "bg-slate-400",
};

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

const percentFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

const parseNumber = (value: string | number | null | undefined) => {
  if (value === null || value === undefined) {
    return null;
  }
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : null;
};

const formatCurrency = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return currencyFormatter.format(value);
};

const formatSignedCurrency = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${currencyFormatter.format(value)}`;
};

const formatNumber = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return numberFormatter.format(value);
};

const formatPercent = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${percentFormatter.format(value)}%`;
};

const formatSignedPercent = (value: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${percentFormatter.format(value)}%`;
};

const logoUrlForSymbol = (symbol: string | null | undefined) => {
  if (!symbol) {
    return undefined;
  }
  const trimmed = symbol.trim();
  if (!trimmed || trimmed === "--") {
    return undefined;
  }
  const safeSymbol = trimmed.toUpperCase();
  return `/api/logos/${safeSymbol}.png`;
};

const formatDateTime = (value: string | null | undefined) => {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "n/a";
  }
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatTimeUtc = (value: string | null | undefined) => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
  }
  const time = parsed.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
  return `${time} UTC`;
};

const formatDateUtc = (value: string | null | undefined) => {
  if (!value) {
    return "--";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "--";
  }
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
};

const formatDuration = (start: string | null | undefined, end: string | null | undefined) => {
  if (!start || !end) {
    return "--";
  }
  const startDate = new Date(start);
  const endDate = new Date(end);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return "--";
  }
  const diffMs = endDate.getTime() - startDate.getTime();
  if (diffMs <= 0) {
    return "--";
  }
  const totalMinutes = Math.round(diffMs / 60000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours > 0 && minutes > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (hours > 0) {
    return `${hours}h`;
  }
  return `${minutes}m`;
};

const formatDurationSeconds = (seconds: number | null | undefined) => {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "--";
  }
  const totalSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;
  if (hours > 0 && minutes > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (hours > 0) {
    return `${hours}h`;
  }
  if (minutes > 0 && remainingSeconds > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m`;
  }
  return `${remainingSeconds}s`;
};

const formatAge = (seconds: number | null | undefined) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) {
    return "n/a";
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s ago`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h ago`;
  }
  return `${Math.round(seconds / 86400)}d ago`;
};

const normalizeWinRate = (value: number | null) => {
  if (value === null || value === undefined) {
    return null;
  }
  return value <= 1 ? value * 100 : value;
};

const statusFromPipelineRc = (value: number | null | undefined) => {
  if (value === 0) {
    return "OK" as const;
  }
  if (value === null || value === undefined) {
    return "UNKNOWN" as const;
  }
  return "FAIL" as const;
};

const parsePipelineLogRun = (logText: string | null | undefined) => {
  if (!logText) {
    return {
      start: null,
      end: null,
      durationSeconds: null,
      rc: null,
      stepRcs: { screener: null, backtest: null, metrics: null },
    };
  }

  const timestampPattern =
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,(\d+))?/;
  const normalizeTimestamp = (line: string) => {
    const match = line.match(timestampPattern);
    if (!match) {
      return null;
    }
    const [, date, time, millis] = match;
    const ms = (millis ?? "000").padEnd(3, "0").slice(0, 3);
    return `${date}T${time}.${ms}Z`;
  };

  let start: string | null = null;
  let end: string | null = null;
  let durationSeconds: number | null = null;
  let rc: number | null = null;
  let startIndex = -1;
  let endIndex = -1;
  type StepKey = "screener" | "backtest" | "metrics";
  const stepRcs: { screener: number | null; backtest: number | null; metrics: number | null } = {
    screener: null,
    backtest: null,
    metrics: null,
  };
  const stepPriorities: Record<StepKey, number> = {
    screener: 0,
    backtest: 0,
    metrics: 0,
  };

  const mapStepName = (raw: string): StepKey | null => {
    const text = raw.toLowerCase();
    if (text.includes("screener")) {
      return "screener";
    }
    if (text.includes("backtest")) {
      return "backtest";
    }
    if (text.includes("metrics")) {
      return "metrics";
    }
    return null;
  };

  const setStepRc = (step: StepKey | null, value: number | null, priority: number) => {
    if (!step) {
      return;
    }
    if (value === null || Number.isNaN(value)) {
      return;
    }
    if (priority >= stepPriorities[step]) {
      stepRcs[step] = value;
      stepPriorities[step] = priority;
    }
  };

  const lines = logText.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (line.includes("PIPELINE_END")) {
      endIndex = i;
      const parsed = normalizeTimestamp(line);
      if (parsed) {
        end = parsed;
      }
      const durationMatch = line.match(/duration=([0-9.]+)s/);
      if (durationMatch) {
        const seconds = Number(durationMatch[1]);
        durationSeconds = Number.isFinite(seconds) ? seconds : durationSeconds;
      }
      const rcMatch = line.match(/rc=([0-9-]+)/);
      if (rcMatch) {
        const parsedRc = Number(rcMatch[1]);
        rc = Number.isFinite(parsedRc) ? parsedRc : rc;
      }
      break;
    }
  }

  if (endIndex >= 0) {
    for (let i = endIndex - 1; i >= 0; i -= 1) {
      const line = lines[i];
      if (line.includes("PIPELINE_START")) {
        startIndex = i;
        const parsed = normalizeTimestamp(line);
        if (parsed) {
          start = parsed;
        }
        break;
      }
    }
  }

  if (startIndex >= 0 && endIndex >= 0) {
    for (let i = startIndex; i <= endIndex; i += 1) {
      const line = lines[i];

      const endMatch = line.match(/\bEND\s+(screener|backtest|metrics)\b.*\brc=([0-9-]+)/i);
      if (endMatch) {
        const step = mapStepName(endMatch[1]);
        const parsedRc = Number(endMatch[2]);
        setStepRc(step, Number.isFinite(parsedRc) ? parsedRc : null, 2);
        continue;
      }

      const timeoutMatch = line.match(/\bSTEP_TIMEOUT\b.*\bname=([a-zA-Z_]+)\b.*\brc=([0-9-]+)/i);
      if (timeoutMatch) {
        const step = mapStepName(timeoutMatch[1]);
        const parsedRc = Number(timeoutMatch[2]);
        setStepRc(step, Number.isFinite(parsedRc) ? parsedRc : null, 2);
        continue;
      }

      const completedMatch = line.match(/\bCompleted\s+(.+?)\s+successfully\b/i);
      if (completedMatch) {
        const step = mapStepName(completedMatch[1]);
        setStepRc(step, 0, 1);
        continue;
      }

      const failedMatch = line.match(/\b(.+?)\s+failed with exit\s+([0-9-]+)\b/i);
      if (failedMatch) {
        const step = mapStepName(failedMatch[1]);
        const parsedRc = Number(failedMatch[2]);
        setStepRc(step, Number.isFinite(parsedRc) ? parsedRc : null, 1);
        continue;
      }

      const stepFailedMatch = line.match(/\bStep\s+(.+?)\s+failed\b/i);
      if (stepFailedMatch) {
        const step = mapStepName(stepFailedMatch[1]);
        if (step) {
          const fallbackRc = stepRcs[step] ?? 1;
          setStepRc(step, fallbackRc, 1);
        }
      }
    }
  }

  if (rc === 0) {
    (["screener", "backtest", "metrics"] as const).forEach((step) => {
      if (stepRcs[step] === null) {
        stepRcs[step] = 0;
      }
    });
  }

  return { start, end, durationSeconds, rc, stepRcs };
};

const parseExecuteLogSummary = (logText: string | null | undefined) => {
  if (!logText) {
    return {
      ordersSubmitted: null as number | null,
      ordersFilled: null as number | null,
      filledCount24h: null as number | null,
      filledValue24h: null as number | null,
      skipCounts: {} as Record<string, number>,
      marketInWindow: null as boolean | null,
    };
  }

  let ordersSubmitted: number | null = null;
  let ordersFilled: number | null = null;
  let filledCount24h: number | null = null;
  let filledValue24h: number | null = null;
  let marketInWindow: boolean | null = null;
  let skipCounts: Record<string, number> = {};

  const lines = logText.split(/\r?\n/);
  const nowMs = Date.now();
  const windowMs = 24 * 60 * 60 * 1000;
  const seenOrders = new Set<string>();
  let fillCount = 0;
  let fillValue = 0;

  const parseTimestampMs = (line: string) => {
    const match = line.match(/(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,(\d+))?/);
    if (!match) {
      return null;
    }
    const [, date, time, millis] = match;
    const ms = (millis ?? "000").padEnd(3, "0").slice(0, 3);
    const parsed = Date.parse(`${date}T${time}.${ms}Z`);
    return Number.isNaN(parsed) ? null : parsed;
  };

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (!line.includes("BUY_FILL") || !line.includes("avg_price=")) {
      continue;
    }
    const ts = parseTimestampMs(line);
    if (ts === null || nowMs - ts > windowMs) {
      continue;
    }
    const orderMatch = line.match(/order_id=([a-f0-9-]+)/i);
    const qtyMatch = line.match(/filled_qty=([0-9.]+)/);
    const priceMatch = line.match(/avg_price=([0-9.]+)/);
    if (!qtyMatch || !priceMatch) {
      continue;
    }
    const qty = Number(qtyMatch[1]);
    const price = Number(priceMatch[1]);
    if (!Number.isFinite(qty) || !Number.isFinite(price)) {
      continue;
    }
    const key = orderMatch?.[1] ?? `${ts}-${qty}-${price}`;
    if (seenOrders.has(key)) {
      continue;
    }
    seenOrders.add(key);
    fillCount += 1;
    fillValue += qty * price;
  }
  filledCount24h = fillCount;
  filledValue24h = fillCount > 0 ? fillValue : null;

  let endIndex = -1;
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (line.includes("EXECUTE_SUMMARY") || line.includes("EXECUTE END")) {
      endIndex = i;
      break;
    }
  }
  const startMarkers = ["Starting pre-market trade execution", "Starting trade execution"];
  let startIndex = 0;
  if (endIndex >= 0) {
    for (let i = endIndex; i >= 0; i -= 1) {
      const line = lines[i];
      if (startMarkers.some((marker) => line.includes(marker))) {
        startIndex = i;
        break;
      }
    }
  } else {
    endIndex = lines.length - 1;
  }

  const parseSkipCounts = (line: string) => {
    const counts: Record<string, number> = {};
    const dotMatches = line.matchAll(/skips\.([A-Z_]+)=(\d+)/g);
    for (const match of dotMatches) {
      counts[match[1]] = Number(match[2]);
    }
    const dictMatches = line.matchAll(/'([A-Z_]+)'\s*:\s*(\d+)/g);
    for (const match of dictMatches) {
      counts[match[1]] = Number(match[2]);
    }
    return counts;
  };

  for (let i = endIndex; i >= startIndex; i -= 1) {
    const line = lines[i];
    if (line.includes("EXECUTE_SUMMARY") || line.includes("EXECUTE END")) {
      const submittedMatch = line.match(/orders_submitted=(\d+)/);
      const submittedAltMatch = line.match(/submitted=(\d+)/);
      if (submittedMatch) {
        const submitted = Number(submittedMatch[1]);
        ordersSubmitted = Number.isFinite(submitted) ? submitted : ordersSubmitted;
      } else if (submittedAltMatch) {
        const submitted = Number(submittedAltMatch[1]);
        ordersSubmitted = Number.isFinite(submitted) ? submitted : ordersSubmitted;
      }
      const filledMatch = line.match(/orders_filled=(\d+)/);
      if (filledMatch) {
        const filled = Number(filledMatch[1]);
        ordersFilled = Number.isFinite(filled) ? filled : ordersFilled;
      }
      skipCounts = { ...skipCounts, ...parseSkipCounts(line) };
      break;
    }
  }

  for (let i = endIndex; i >= startIndex; i -= 1) {
    const line = lines[i];
    if (line.includes("MARKET_TIME") && line.includes("in_window=")) {
      const match = line.match(/in_window=(True|False)/);
      if (match) {
        marketInWindow = match[1] === "True";
        break;
      }
    }
  }

  return {
    ordersSubmitted,
    ordersFilled,
    filledCount24h,
    filledValue24h,
    skipCounts,
    marketInWindow,
  };
};

const isFilledStatus = (status: string | undefined) => {
  if (!status) {
    return false;
  }
  const normalized = status.toLowerCase();
  return normalized.includes("fill") || normalized === "closed" || normalized === "filled";
};

const isOpenStatus = (status: string | undefined) => {
  if (!status) {
    return false;
  }
  return status.toLowerCase() === "open";
};

const fetchJson = async <T,>(path: string): Promise<T | null> => {
  try {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
};

const fetchText = async (path: string): Promise<string | null> => {
  try {
    const response = await fetch(path, {
      headers: { Accept: "text/plain" },
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }
    return await response.text();
  } catch {
    return null;
  }
};

type ParsedLogEntry = LogEntry & { timestampMs: number };

const normalizeLogLevel = (rawLevel: string | undefined): LogEntry["level"] => {
  if (!rawLevel) {
    return "INFO";
  }
  const level = rawLevel.toUpperCase();
  if (level.startsWith("WARN")) {
    return "WARN";
  }
  if (level === "ERROR") {
    return "ERROR";
  }
  if (level === "SUCCESS") {
    return "SUCCESS";
  }
  return "INFO";
};

const buildLogEntries = (sources: Array<{ text: string | null; source: string }>, limit = 8) => {
  const entries: ParsedLogEntry[] = [];
  const pattern =
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?(?:\s+-\s+[^-]+\s+-\s+)?\s*(?:\[(\w+)\])?\s*(.*)$/;

  sources.forEach(({ text }) => {
    if (!text) {
      return;
    }
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const sample = lines.slice(-Math.max(limit, 12));
    sample.forEach((line) => {
      const match = line.match(pattern);
      if (match) {
        const [, date, time, level, message] = match;
        const timestampMs = new Date(`${date}T${time}`).getTime() || 0;
        entries.push({
          time,
          level: normalizeLogLevel(level),
          message: message.trim() || "(no message)",
          timestampMs,
        });
      } else {
        entries.push({
          time: "--:--:--",
          level: "INFO",
          message: line.trim(),
          timestampMs: 0,
        });
      }
    });
  });

  entries.sort((a, b) => b.timestampMs - a.timestampMs);
  return entries.slice(0, limit).map(({ timestampMs, ...entry }) => entry);
};

const computePipelineScore = (health: ApiHealthResponse | null) => {
  if (!health) {
    return null;
  }
  const checks = [
    typeof health.pipeline_rc === "number" ? health.pipeline_rc === 0 : null,
    typeof health.trading_ok === "boolean" ? health.trading_ok : null,
    typeof health.data_ok === "boolean" ? health.data_ok : null,
  ].filter((value): value is boolean => typeof value === "boolean");

  if (!checks.length) {
    return null;
  }
  const passed = checks.filter(Boolean).length;
  const score = (passed / checks.length) * 100;
  return Math.round(score * 10) / 10;
};

const pipelineStatusFromScore = (score: number | null) => {
  if (score === null) {
    return { label: "Unknown", tone: "neutral" as StatusTone };
  }
  if (score >= 90) {
    return { label: "Healthy", tone: "success" as StatusTone };
  }
  if (score >= 70) {
    return { label: "Watch", tone: "warning" as StatusTone };
  }
  return { label: "Degraded", tone: "error" as StatusTone };
};

type DashboardHealthProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

export default function DashboardHealth({ activeTab, onTabSelect }: DashboardHealthProps) {
  const [healthSnapshot, setHealthSnapshot] = useState<ApiHealthResponse | null>(null);
  const [overviewSnapshot, setOverviewSnapshot] = useState<HealthOverviewResponse | null>(null);
  const [accountSnapshot, setAccountSnapshot] = useState<AccountSnapshot | null>(null);
  const [tradesOverview, setTradesOverview] = useState<TradesOverviewResponse | null>(null);
  const [executeSnapshot, setExecuteSnapshot] = useState<ExecutionSnapshot | null>(null);
  const [monitoringSnapshot, setMonitoringSnapshot] = useState<MonitoringPositionsResponse | null>(null);
  const [pipelineTaskRun, setPipelineTaskRun] = useState<PipelineTaskRunResponse | null>(null);
  const [executeTaskRun, setExecuteTaskRun] = useState<ExecuteTaskRunResponse | null>(null);
  const [executeOrdersSummary, setExecuteOrdersSummary] =
    useState<ExecuteOrdersSummaryResponse | null>(null);
  const [pipelineLogText, setPipelineLogText] = useState<string | null>(null);
  const [executeLogText, setExecuteLogText] = useState<string | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
        const [
          overview,
          health,
          accountPayload,
          tradesPayload,
          executePayload,
          monitoringPayload,
          taskRunPayload,
          executeTaskPayload,
          executeOrdersPayload,
          pipelineLog,
          executeLog,
        ] = await Promise.all([
          fetchJson<HealthOverviewResponse>("/health/overview"),
          fetchJson<ApiHealthResponse>("/api/health"),
          fetchJson<AccountOverviewResponse>("/api/account/overview"),
          fetchJson<TradesOverviewResponse>("/api/trades/overview"),
          fetchJson<ExecutionSnapshot>("/api/execute/overview"),
          fetchJson<MonitoringPositionsResponse>("/api/positions/monitoring"),
          fetchJson<PipelineTaskRunResponse>("/api/pipeline/task"),
          fetchJson<ExecuteTaskRunResponse>("/api/execute/task"),
          fetchJson<ExecuteOrdersSummaryResponse>("/api/execute/orders-summary"),
          fetchText(`/api/logs/pipeline.log?ts=${Date.now()}`),
          fetchText(`/api/logs/execute_trades.log?ts=${Date.now()}`),
        ]);

      if (!isMounted) {
        return;
      }

      setOverviewSnapshot(overview);
      setHealthSnapshot(health);
      setAccountSnapshot(accountPayload?.snapshot ?? null);
      setTradesOverview(tradesPayload);
        setExecuteSnapshot(executePayload);
        setMonitoringSnapshot(monitoringPayload);
        setPipelineTaskRun(taskRunPayload);
        setExecuteTaskRun(executeTaskPayload);
        setExecuteOrdersSummary(executeOrdersPayload);
        setPipelineLogText(pipelineLog);
        setExecuteLogText(executeLog);
        setLogEntries(
          buildLogEntries([
            { source: "pipeline", text: pipelineLog },
            { source: "execute", text: executeLog },
          ])
        );
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  const rightBadges = useMemo(() => {
    const liveTone: StatusTone =
      healthSnapshot?.trading_ok === true
        ? "success"
        : healthSnapshot?.trading_ok === false
          ? "error"
          : "neutral";
    return [
      { label: "Paper Trading", tone: "warning" as const },
      {
        label: "Live",
        tone: liveTone as StatusTone,
        showDot: healthSnapshot?.trading_ok === true,
      },
    ];
  }, [healthSnapshot?.trading_ok]);

  const pipelineScore = useMemo(
    () => computePipelineScore(healthSnapshot),
    [healthSnapshot]
  );

  const pipelineStatus = useMemo(
    () => pipelineStatusFromScore(pipelineScore),
    [pipelineScore]
  );

  const pipelineFootnote = useMemo(() => {
    const lastRun = formatDateTime(healthSnapshot?.last_run_utc);
    const source = healthSnapshot?.latest_source
      ? `Source: ${healthSnapshot.latest_source}`
      : null;
    return source ? `Last run: ${lastRun} / ${source}` : `Last run: ${lastRun}`;
  }, [healthSnapshot?.last_run_utc, healthSnapshot?.latest_source]);

  const tradesList = useMemo(() => tradesOverview?.trades ?? [], [tradesOverview]);

  const openTrades = useMemo(
    () => tradesList.filter((trade) => isOpenStatus(trade.status)),
    [tradesList]
  );

  const openTradeValue = useMemo(() => {
    let total = 0;
    let hasValue = false;
    openTrades.forEach((trade) => {
      const qty = parseNumber(trade.qty ?? null);
      const entryPrice = parseNumber(trade.entry_price ?? null);
      if (qty !== null && entryPrice !== null) {
        total += qty * entryPrice;
        hasValue = true;
      }
    });
    return hasValue ? total : Number.NaN;
  }, [openTrades]);

  const monitoringPositions = useMemo(
    () =>
      openTrades.map((trade) => {
        const symbol = trade.symbol ?? "--";
        const qty = parseNumber(trade.qty ?? null);
        const entryPrice = parseNumber(trade.entry_price ?? null);
        const exitPrice = parseNumber(trade.exit_price ?? null);
        const currentPrice = exitPrice ?? entryPrice ?? Number.NaN;
        const dollarPl = parseNumber(trade.realized_pnl ?? null);
        const costBasis = qty !== null && entryPrice !== null ? qty * entryPrice : null;
        const percentPl =
          costBasis !== null && dollarPl !== null ? (dollarPl / costBasis) * 100 : Number.NaN;
        const sparklineData = Number.isNaN(currentPrice)
          ? []
          : Array.from({ length: 6 }, () => currentPrice);

        return {
          symbol,
          logoUrl: logoUrlForSymbol(symbol),
          qty: qty ?? Number.NaN,
          entryPrice: entryPrice ?? Number.NaN,
          currentPrice,
          sparklineData,
          percentPL: percentPl,
          dollarPL: dollarPl ?? Number.NaN,
          costBasis: costBasis ?? Number.NaN,
        };
      }),
    [openTrades]
  );

  const monitoringPositionsFromApi = useMemo(() => {
    const positions = monitoringSnapshot?.positions;
    if (!positions || positions.length === 0) {
      return [];
    }
    return positions.map((position) => ({
      symbol: position.symbol ?? "--",
      logoUrl: position.logoUrl ?? logoUrlForSymbol(position.symbol ?? "--"),
      qty: typeof position.qty === "number" ? position.qty : Number.NaN,
      entryPrice: typeof position.entryPrice === "number" ? position.entryPrice : Number.NaN,
      currentPrice:
        typeof position.currentPrice === "number" ? position.currentPrice : Number.NaN,
      sparklineData: Array.isArray(position.sparklineData)
        ? position.sparklineData.filter((value) => Number.isFinite(value))
        : [],
      percentPL: typeof position.percentPL === "number" ? position.percentPL : Number.NaN,
      dollarPL: typeof position.dollarPL === "number" ? position.dollarPL : Number.NaN,
      costBasis: typeof position.costBasis === "number" ? position.costBasis : Number.NaN,
    }));
  }, [monitoringSnapshot?.positions]);
  const hasMonitoringApi = monitoringSnapshot?.ok === true;

  const pipelineLogRun = useMemo(
    () => parsePipelineLogRun(pipelineLogText),
    [pipelineLogText]
  );
  const executeLogSummary = useMemo(
    () => parseExecuteLogSummary(executeLogText),
    [executeLogText]
  );

  const tradeStats = useMemo(() => {
    if (!tradesList.length) {
      return {
        wins: 0,
        losses: 0,
        netPnl: null,
        winRate: null,
        filledCount: null,
      };
    }
    let wins = 0;
    let losses = 0;
    let netPnl = 0;
    let filledCount = 0;
    tradesList.forEach((trade) => {
      const pnl = parseNumber(trade.realized_pnl ?? null);
      if (pnl === null) {
        return;
      }
      netPnl += pnl;
      if (pnl > 0) {
        wins += 1;
      } else if (pnl < 0) {
        losses += 1;
      }
      if (isFilledStatus(trade.status)) {
        filledCount += 1;
      }
    });
    const total = wins + losses;
    const winRate = total > 0 ? (wins / total) * 100 : null;
    return { wins, losses, netPnl, winRate, filledCount };
  }, [tradesList]);

  const openPositions = useMemo<OpenPositionsSummary>(() => {
    const openCountFromApi = tradesOverview?.open_positions?.count;
    const openCountFallback = tradesList.length
      ? tradesList.filter((trade) => isOpenStatus(trade.status)).length
      : null;
    const count = typeof openCountFromApi === "number" ? openCountFromApi : openCountFallback;
    const pnl = parseNumber(tradesOverview?.open_positions?.realized_pnl ?? null);
    const filledCount = tradesList.length ? tradeStats.filledCount : null;
    return {
      count,
      pnl,
      pnlPct: null,
      filledCount,
      totalCount: tradesList.length,
    };
  }, [tradesOverview, tradesList, tradeStats.filledCount]);

  const tradesMetrics = useMemo(() => {
    const kpis = tradesOverview?.metrics ?? overviewSnapshot?.kpis ?? {};
    const totalTrades =
      parseNumber(kpis.total_trades) ??
      (tradesList.length ? tradesList.length : null) ??
      parseNumber(overviewSnapshot?.trades_log_rows ?? null);
    const winRateRaw = parseNumber(kpis.win_rate) ?? tradeStats.winRate ?? null;
    const winRate = normalizeWinRate(winRateRaw);
    const netPnl = parseNumber(kpis.net_pnl) ?? tradeStats.netPnl ?? null;
    return { totalTrades, winRate, netPnl };
  }, [overviewSnapshot, tradeStats.netPnl, tradeStats.winRate, tradesList.length, tradesOverview]);

  const buyingPowerValue = parseNumber(
    healthSnapshot?.buying_power ?? accountSnapshot?.buying_power ?? accountSnapshot?.equity ?? null
  );

  const openPositionsDetail = useMemo(() => {
    if (!openPositions || openPositions.pnl === null) {
      return "Open P/L: --";
    }
    const pctDetail =
      openPositions.pnlPct !== null
        ? ` (${formatSignedPercent(openPositions.pnlPct)})`
        : "";
    return `Open P/L: ${formatSignedCurrency(openPositions.pnl)}${pctDetail}`;
  }, [openPositions]);

  const openPositionsTone: StatusTone = useMemo(() => {
    if (!openPositions || openPositions.pnl === null) {
      return "neutral";
    }
    return openPositions.pnl < 0 ? "error" : "success";
  }, [openPositions]);

  const processStatusCards = useMemo<ProcessStatusCardsProps>(() => {
    const pipelineStartRaw = overviewSnapshot?.kpis?.last_run_utc;
    const pipelineStart =
      pipelineTaskRun?.started_utc ??
      pipelineLogRun.start ??
      (pipelineStartRaw ? String(pipelineStartRaw) : null);
    const pipelineEnd =
      pipelineTaskRun?.finished_utc ?? pipelineLogRun.end ?? healthSnapshot?.last_run_utc ?? null;
    const pipelineDuration =
      pipelineTaskRun?.duration_seconds !== null &&
      pipelineTaskRun?.duration_seconds !== undefined
        ? formatDurationSeconds(pipelineTaskRun.duration_seconds)
        : pipelineLogRun.durationSeconds
          ? formatDurationSeconds(pipelineLogRun.durationSeconds)
          : formatDuration(pipelineStart, pipelineEnd);
    const pipelineEndDate = pipelineEnd ? new Date(pipelineEnd) : null;
    const pipelineRecent =
      pipelineEndDate && !Number.isNaN(pipelineEndDate.getTime())
        ? Date.now() - pipelineEndDate.getTime() <= 24 * 60 * 60 * 1000
        : false;
    const pipelineRc = pipelineTaskRun?.rc ?? pipelineLogRun.rc;
    const pipelineLive = pipelineRecent && pipelineRc === 0;
    const executeStart = executeTaskRun?.started_utc ?? executeSnapshot?.last_execution ?? null;
    const executeEnd =
      executeTaskRun?.finished_utc ?? executeSnapshot?.ny_now ?? executeSnapshot?.last_execution ?? null;
    const executeDuration =
      executeTaskRun?.duration_seconds !== null && executeTaskRun?.duration_seconds !== undefined
        ? formatDurationSeconds(executeTaskRun.duration_seconds)
        : formatDuration(executeStart, executeEnd);
    const executeEndDate = executeEnd ? new Date(executeEnd) : null;
    const executeRecent =
      executeEndDate && !Number.isNaN(executeEndDate.getTime())
        ? Date.now() - executeEndDate.getTime() <= 24 * 60 * 60 * 1000
        : false;
    const executeCycleComplete = executeRecent && executeTaskRun?.rc === 0;

    const ordersSubmitted =
      executeLogSummary.ordersSubmitted ?? executeSnapshot?.orders_submitted;
    const ordersFilled =
      executeOrdersSummary?.orders_filled ?? executeLogSummary.ordersFilled ?? executeSnapshot?.orders_filled;
    const ordersPlaced =
      executeOrdersSummary?.orders_filled !== null &&
      executeOrdersSummary?.orders_filled !== undefined
        ? executeOrdersSummary.orders_filled
        : executeLogSummary.filledCount24h !== null
          ? executeLogSummary.filledCount24h
          : typeof ordersSubmitted === "number"
            ? ordersSubmitted
            : Number.NaN;
    const successRate =
      typeof ordersSubmitted === "number" &&
      ordersSubmitted > 0 &&
      typeof ordersFilled === "number"
        ? (ordersFilled / ordersSubmitted) * 100
        : Number.NaN;

    const executeMarketNote =
      (ordersPlaced === 0 || Number.isNaN(ordersPlaced)) &&
      (executeLogSummary.marketInWindow === false ||
        (executeLogSummary.skipCounts.TIME_WINDOW ?? 0) > 0)
        ? "Market closed"
        : null;

    return {
      pipeline: {
        lastRun: {
          date: formatDateUtc(pipelineStart ?? pipelineEnd),
          start: formatTimeUtc(pipelineStart),
          end: formatTimeUtc(pipelineEnd),
          duration: pipelineDuration,
        },
        subprocess: {
          screener: statusFromPipelineRc(pipelineLogRun.stepRcs.screener),
          backTester: statusFromPipelineRc(pipelineLogRun.stepRcs.backtest),
          metrics: statusFromPipelineRc(pipelineLogRun.stepRcs.metrics),
        },
        isLive: pipelineLive,
      },
      executeTrades: {
        lastRun: {
          date: formatDateUtc(executeStart ?? executeEnd),
          start: formatTimeUtc(executeStart),
          end: formatTimeUtc(executeEnd),
          duration: executeDuration,
        },
        ordersPlaced,
        totalValue:
          executeOrdersSummary?.total_value !== null &&
          executeOrdersSummary?.total_value !== undefined
            ? executeOrdersSummary.total_value
            : executeLogSummary.filledValue24h !== null
              ? executeLogSummary.filledValue24h
              : ordersPlaced === 0
                ? 0
                : openTradeValue,
        successRate,
        isCycleComplete: executeCycleComplete,
        marketNote: executeMarketNote,
      },
      monitoring: {
        positions: hasMonitoringApi ? monitoringPositionsFromApi : monitoringPositions,
      },
    };
  }, [
    executeSnapshot?.in_window,
    executeSnapshot?.last_execution,
    executeSnapshot?.ny_now,
    executeSnapshot?.orders_filled,
    executeSnapshot?.orders_submitted,
    executeLogSummary.filledCount24h,
    executeLogSummary.filledValue24h,
    executeLogSummary.marketInWindow,
    executeLogSummary.ordersFilled,
    executeLogSummary.ordersSubmitted,
    executeLogSummary.skipCounts,
    executeTaskRun?.duration_seconds,
    executeTaskRun?.finished_utc,
    executeTaskRun?.started_utc,
    executeTaskRun?.rc,
    executeOrdersSummary?.orders_filled,
    executeOrdersSummary?.total_value,
    healthSnapshot?.last_run_utc,
    healthSnapshot?.pipeline_rc,
    healthSnapshot?.trading_ok,
    monitoringPositionsFromApi,
    hasMonitoringApi,
    monitoringPositions,
    openTradeValue,
    overviewSnapshot?.kpis,
    pipelineTaskRun?.duration_seconds,
    pipelineTaskRun?.finished_utc,
    pipelineTaskRun?.rc,
    pipelineTaskRun?.started_utc,
    pipelineLogRun.durationSeconds,
    pipelineLogRun.end,
    pipelineLogRun.rc,
    pipelineLogRun.stepRcs.backtest,
    pipelineLogRun.stepRcs.metrics,
    pipelineLogRun.stepRcs.screener,
    pipelineLogRun.start,
  ]);

  const kpiCards = useMemo(
    () => [
      {
        title: "Pipeline Health",
        value: formatPercent(pipelineScore),
        detail: pipelineStatus.label,
        detailTone: pipelineStatus.tone,
        footnote: pipelineFootnote,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-emerald-600" fill="none">
            <path
              d="M4 12h3l2-5 4 10 2-5h5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.1s_ease-out_1]",
      },
      {
        title: "Buying Power",
        value: formatCurrency(buyingPowerValue),
        detail: `Cash: ${formatCurrency(parseNumber(accountSnapshot?.cash ?? null))}`,
        detailTone: "info" as const,
        footnote: `Equity: ${formatCurrency(parseNumber(accountSnapshot?.equity ?? null))}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-sky-600" fill="none">
            <path
              d="M12 3v18M8 7h6a3 3 0 0 1 0 6H10a3 3 0 0 0 0 6h6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.25s_ease-out_1]",
      },
      {
        title: "Open Positions",
        value: formatNumber(openPositions?.count ?? null),
        detail: openPositionsDetail,
        detailTone: openPositionsTone,
        footnote: `Filled orders: ${formatNumber(openPositions?.filledCount ?? null)}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-violet-600" fill="none">
            <path
              d="M4 16l5-5 4 4 7-7"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M16 8h4v4"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.4s_ease-out_1]",
      },
      {
        title: "Recent Trades Summary",
        value: formatNumber(tradesMetrics.totalTrades),
        detail: `Win rate: ${formatPercent(tradesMetrics.winRate)}`,
        detailTone: "neutral" as const,
        footnote: `Net P/L: ${formatSignedCurrency(tradesMetrics.netPnl)}`,
        icon: (
          <svg viewBox="0 0 24 24" className="h-5 w-5 text-orange-600" fill="none">
            <path
              d="M5 12h3v7H5zM10.5 5h3v14h-3zM16 9h3v10h-3z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ),
        animation: "motion-safe:animate-[pulse_1.55s_ease-out_1]",
      },
    ],
    [
      accountSnapshot?.cash,
      accountSnapshot?.equity,
      buyingPowerValue,
      openPositions,
      openPositionsDetail,
      openPositionsTone,
      pipelineFootnote,
      pipelineScore,
      pipelineStatus.label,
      pipelineStatus.tone,
      tradesMetrics.netPnl,
      tradesMetrics.totalTrades,
      tradesMetrics.winRate,
    ]
  );

  const systemStatus: SystemStatusItem[] = useMemo(() => {
    const tradingOk = healthSnapshot?.trading_ok ?? null;
    const dataOk = healthSnapshot?.data_ok ?? null;
    const pipelineRc = healthSnapshot?.pipeline_rc ?? null;
    const feed = healthSnapshot?.feed ? healthSnapshot.feed.toUpperCase() : null;

    const apiTone: StatusTone = tradingOk === false ? "error" : tradingOk === true ? "success" : "neutral";
    const dataTone: StatusTone = dataOk === false ? "warning" : dataOk === true ? "success" : "neutral";
    const pipelineTone: StatusTone =
      pipelineRc === 0 ? "success" : pipelineRc === null ? "neutral" : "warning";

    return [
      {
        title: "API Connection",
        status: tradingOk === true ? "Connected" : tradingOk === false ? "Offline" : "Unknown",
        tone: apiTone,
        description: tradingOk === true ? "Broker API responding" : "Broker API status unknown",
        meta: `Status: ${healthSnapshot?.trading_status ?? "n/a"}`,
      },
      {
        title: "Market Data Feed",
        status: dataOk === true ? "Streaming" : dataOk === false ? "Delayed" : "Unknown",
        tone: dataTone,
        description: feed ? `Feed: ${feed}` : "Market data feed",
        meta: `Status: ${healthSnapshot?.data_status ?? "n/a"}`,
      },
      {
        title: "ML Pipeline",
        status: pipelineRc === 0 ? "Ready" : pipelineRc === null ? "Unknown" : "Degraded",
        tone: pipelineTone,
        description: healthSnapshot?.run_type
          ? `Run type: ${healthSnapshot.run_type}`
          : "Run type: n/a",
        meta: `Rows: ${formatNumber(healthSnapshot?.rows_final ?? null)}`,
      },
    ];
  }, [
    healthSnapshot?.data_ok,
    healthSnapshot?.data_status,
    healthSnapshot?.feed,
    healthSnapshot?.pipeline_rc,
    healthSnapshot?.rows_final,
    healthSnapshot?.run_type,
    healthSnapshot?.trading_ok,
    healthSnapshot?.trading_status,
  ]);

  const updatedLabel = useMemo(() => {
    if (healthSnapshot?.freshness?.age_seconds !== undefined) {
      return `Updated ${formatAge(healthSnapshot.freshness.age_seconds)}`;
    }
    return "Updated n/a";
  }, [healthSnapshot?.freshness?.age_seconds]);

  const emptyEntries: LogEntry[] = [
    { time: "--:--:--", level: "INFO", message: "No recent activity logged." },
  ];

  const displayEntries = logEntries.length ? logEntries : emptyEntries;

  const currentTab = activeTab ?? "Dashboard";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-20 pb-12 sm:pt-24">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <section className="pt-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Process Status Cards</h2>
            </div>
            <div className="mt-4">
              <ProcessStatusCards {...processStatusCards} />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}



