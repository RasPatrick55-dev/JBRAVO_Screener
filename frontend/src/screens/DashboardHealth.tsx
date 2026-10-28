import { canDisplay, deadlineRead, useSource } from "../components/ui/sourceState";
import { useEffect, useMemo, useState } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import { buildNavbarBadges, useLiveTradingStatus, parseTradingStatus, diagnosticsFresh } from "../components/navbar/liveStatus";
import ProcessStatusCards, {
  type ProcessStatusCardsProps,
} from "../components/dashboard/ProcessStatusCards";
import AlpacaCard from "../components/dashboard/AlpacaCard";
import PythonAnywhereResources from "../components/dashboard/PythonAnywhereResources";
import MLDailyImprovements from "../components/dashboard/MLDailyImprovements";
import LatestLogsTable from "../components/dashboard/LatestLogsTable";
import PythonAnywhereTasksTable from "../components/dashboard/PythonAnywhereTasksTable";
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
  performance_status?: string;
  trades?: TradeRecord[];
  open_positions?: {
    count?: number | null;
    unrealized_pnl?: number | null;
    status?: string;
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
  summary?: {totalOpenPL?: number | null};
  status?: string;
  observed_at?: string | null;
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

type PythonAnywhereResource = {
  label: string;
  value: number;
  used?: number | null;
  limit?: number | null;
  unit?: string | null;
  source?: string | null;
};

type PythonAnywhereResourcesResponse = {
  ok?: boolean;
  resources?: PythonAnywhereResource[];
  source?: string | null;
};

type PythonAnywhereTask = {
  name: string;
  frequency: string;
  time: string;
};

type PythonAnywhereTasksResponse = {
  ok?: boolean;
  tasks?: PythonAnywhereTask[];
  source?: string | null;
};

type PythonAnywhereLogSource = {
  source: string;
  text: string;
};

type PythonAnywhereLogsResponse = {
  ok?: boolean;
  sources?: PythonAnywhereLogSource[];
  source?: string | null;
};

type OpenPositionsSummary = {
  count: number | null;
  pnl: number | null;
  pnlPct: number | null;
  filledCount: number | null;
  totalCount: number | null;
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

const pythonAnywhereResourceDefaults = [
  { label: "CPU Usage", value: 10 },
  { label: "File Storage", value: 35 },
  { label: "Postgres Storage", value: 10 },
];

const pythonAnywhereTasksDefaults = [
  { name: "Run Pipeline", frequency: "Daily", time: "03:45 UTC" },
  { name: "Screener Backtest", frequency: "Daily", time: "04:00 UTC" },
  { name: "Metrics Update", frequency: "Hourly", time: "**:15 UTC" },
];

const statusDotTone: Record<StatusTone, string> = {
  success: "jbravo-status-dot-success",
  warning: "bg-amber-500",
  error: "jbravo-status-dot-error",
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

const normalizeSymbol = (value: string | null | undefined) => {
  const trimmed = value?.trim() ?? "";
  return trimmed ? trimmed.toUpperCase() : "--";
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

const formatLogDateTime = (date: string | null | undefined, time: string | null | undefined) => {
  if (!date || !time) {
    return "--";
  }
  const parsed = Date.parse(`${date}T${time}Z`);
  if (Number.isNaN(parsed)) {
    return `${date} ${time}`;
  }
  const formatted = new Date(parsed).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
  return `${formatted} UTC`;
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

const labelLogSource = (source: string) => {
  const raw = source.trim();
  if (!raw) {
    return "task";
  }
  const normalized = raw.toLowerCase();

  const scriptMatches = Array.from(raw.matchAll(/([A-Za-z0-9_./\\-]+\.py)\b/g));
  if (scriptMatches.length) {
    const match = scriptMatches[scriptMatches.length - 1][1];
    const base = match.split(/[/\\]/).pop() ?? match;
    const aliasMap: Record<string, string> = {
      "pythonanywhere_usage.py": "PA Usage",
      "run_pipeline.py": "Run Pipeline",
      "execute_trades.py": "Execute Trades",
      "monitor_positions.py": "Monitor Positions",
      "update_dashboard_data.py": "Dash Update",
    };
    return aliasMap[base] ?? base;
  }

  if (normalized.includes("pipeline")) {
    return "Run Pipeline";
  }
  if (normalized.includes("execute")) {
    return "Execute Trades";
  }
  if (normalized.includes("monitor")) {
    return "Monitor Positions";
  }
  if (normalized.startsWith("bash") || normalized.startsWith("sh")) {
    return "Shell Task";
  }
  if (normalized.startsWith("python")) {
    return "Python Task";
  }
  if (normalized.endsWith(".log")) {
    return normalized.replace(/\.log$/i, ".py");
  }
  return raw.length > 24 ? `${raw.slice(0, 21)}...` : raw;
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

export const normalizeWinRate = (value: number | null) => {
  if (value === null || value === undefined) {
    return null;
  }
  return Number.isFinite(value) ? value : null; // scripts.metrics and API contract both use percentage points.
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





const fetchJson = async <T,>(path: string): Promise<T | null> => {
  try { return await deadlineRead(async signal => {
    const response = await fetch(path, {signal, cache: "no-store", headers: {Accept: "application/json"}});
    return response.ok ? await response.json() as T : null;
  }); } catch { return null; }
};

const fetchText = async (path: string): Promise<string | null> => {
  try {return await deadlineRead(async signal => {
    const response = await fetch(path, {signal, cache: "no-store", headers: {Accept: "text/plain"}});
    return response.ok ? await response.text() : null;
  });} catch {return null;}
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

const parseLogLine = (line: string) => {
  const patterns = [
    /^\[(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?\]\s*(?:\[(\w+)\])?\s*(.*)$/,
    /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?(?:\s+-\s+(\w+)\s+-\s+)?\s*(?:\[(\w+)\])?\s*(.*)$/,
  ];
  for (const pattern of patterns) {
    const match = line.match(pattern);
    if (!match) {
      continue;
    }
    if (pattern === patterns[0]) {
      const [, date, time, level, message] = match;
      return { date, time, level, message };
    }
    const [, date, time, levelA, levelB, message] = match;
    return { date, time, level: levelA || levelB, message };
  }
  return null;
};

const buildLogEntries = (
  sources: Array<{ text: string | null; source: string }>,
  maxLinesPerSource = 240,
  maxAgeHours = 48
) => {
  const entries: ParsedLogEntry[] = [];

  sources.forEach(({ text, source }) => {
    if (!text) {
      return;
    }
    const sourceLabel = labelLogSource(source);
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const sample = lines.slice(-Math.max(maxLinesPerSource, 12));
    sample.forEach((line) => {
      const parsed = parseLogLine(line);
      if (parsed) {
        const cleanedMessage = parsed.message.trim().replace(/^[\s:=-]+/, "").trim();
        const timestampMs = Date.parse(`${parsed.date}T${parsed.time}Z`) || 0;
        entries.push({
          time: formatLogDateTime(parsed.date, parsed.time),
          level: normalizeLogLevel(parsed.level),
          message: cleanedMessage || "(no message)",
          source: sourceLabel,
          timestampMs,
        });
      } else {
        entries.push({
          time: "--",
          level: "INFO",
          message: line.trim(),
          source: sourceLabel,
          timestampMs: 0,
        });
      }
    });
  });

  let filtered = entries;
  if (maxAgeHours && Number.isFinite(maxAgeHours)) {
    const cutoff = Date.now() - maxAgeHours * 60 * 60 * 1000;
    filtered = entries.filter((entry) => entry.timestampMs > 0 && entry.timestampMs >= cutoff);
  }

  filtered.sort((a, b) => b.timestampMs - a.timestampMs);
  return filtered.map(({ timestampMs, ...entry }) => entry);
};

export const computePipelineScore = (health: ApiHealthResponse | null) => {
  if (!health) {
    return null;
  }
  if (!health.freshness || health.freshness.freshness_level !== "green" || typeof health.freshness.age_seconds !== "number" || health.freshness.age_seconds < 0 || health.freshness.age_seconds >= 7200) return null;
  const checks = [
    typeof health.pipeline_rc === "number" ? health.pipeline_rc === 0 : null,
    typeof health.trading_ok === "boolean" ? health.trading_ok : null,
    typeof health.data_ok === "boolean" ? health.data_ok : null,
  ];

  if (checks.some(value => value === null)) {
    return null;
  }
  const passed = checks.filter(Boolean).length;
  const score = (passed / checks.length) * 100;
  return Math.round(score * 10) / 10;
};

export const pipelineStatusFromScore = (score: number | null) => {
  if (score === null) {
    return { label: "Unknown", tone: "neutral" as StatusTone };
  }
  if (score >= 90) {
    return { label: "Computational/API checks passed", tone: "success" as StatusTone };
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
  const accountPayloadSource = useSource("/api/account/overview");
  const accountSnapshot = canDisplay(accountPayloadSource.kind) ? (accountPayloadSource.payload as AccountOverviewResponse | null)?.snapshot ?? null : null;
  const tradesPayloadSource = useSource("/api/trades/overview");
  const tradesOverview = canDisplay(tradesPayloadSource.kind) ? tradesPayloadSource.payload as TradesOverviewResponse | null : null;
  const executePayloadSource = useSource("/api/execute/overview");
  const executeSnapshot = canDisplay(executePayloadSource.kind) ? executePayloadSource.payload as ExecutionSnapshot | null : null;
  const monitoringSource = useSource("/api/positions/monitoring");
  const monitoringSnapshot = canDisplay(monitoringSource.kind) ? monitoringSource.payload as MonitoringPositionsResponse | null : null;
  const [pipelineTaskRun, setPipelineTaskRun] = useState<PipelineTaskRunResponse | null>(null);
  const [executeTaskRun, setExecuteTaskRun] = useState<ExecuteTaskRunResponse | null>(null);
  const [executeOrdersSummary, setExecuteOrdersSummary] =
    useState<ExecuteOrdersSummaryResponse | null>(null);
  const [pipelineLogText, setPipelineLogText] = useState<string | null>(null);
  const [executeLogText, setExecuteLogText] = useState<string | null>(null);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [pythonAnywhereResources, setPythonAnywhereResources] =
    useState<PythonAnywhereResource[] | null>(null);
  const [pythonAnywhereTasks, setPythonAnywhereTasks] = useState<PythonAnywhereTask[] | null>(null);

  useEffect(() => {
    let isMounted = true, inFlight = false;
    const load = async () => {
      if (inFlight) return;
      inFlight = true;
      const [
        overview,
        health,
        taskRunPayload,
        executeTaskPayload,
        executeOrdersPayload,
        pythonAnywhereResourcesPayload,
        pythonAnywhereTasksPayload,
        pythonAnywhereLogsPayload,
        pipelineLog,
        executeLog,
        monitorLog,
      ] = await Promise.all([
        fetchJson<HealthOverviewResponse>("/api/health/overview"),
        fetchJson<ApiHealthResponse>("/api/health"),
        fetchJson<PipelineTaskRunResponse>("/api/pipeline/task"),
        fetchJson<ExecuteTaskRunResponse>("/api/execute/task"),
        fetchJson<ExecuteOrdersSummaryResponse>("/api/execute/orders-summary"),
        fetchJson<PythonAnywhereResourcesResponse>("/api/pythonanywhere/resources"),
        fetchJson<PythonAnywhereTasksResponse>("/api/pythonanywhere/tasks"),
        fetchJson<PythonAnywhereLogsResponse>("/api/pythonanywhere/logs"),
        fetchText(`/api/logs/pipeline.log?ts=${Date.now()}`),
        fetchText(`/api/logs/execute_trades.log?ts=${Date.now()}`),
        fetchText(`/api/logs/monitor.log?ts=${Date.now()}`),
      ]);

      if (!isMounted) {
        return;
      }

      setOverviewSnapshot(overview);
      setHealthSnapshot(health);
      setPipelineTaskRun(taskRunPayload);
      setExecuteTaskRun(executeTaskPayload);
      setExecuteOrdersSummary(executeOrdersPayload);
      setPythonAnywhereResources(pythonAnywhereResourcesPayload?.resources ?? null);
      setPythonAnywhereTasks(pythonAnywhereTasksPayload?.tasks ?? null);
      setPipelineLogText(pipelineLog);
      setExecuteLogText(executeLog);
      const pythonAnyLogSources =
        pythonAnywhereLogsPayload?.sources?.filter(
          (source) => source && typeof source.text === "string" && source.text.trim()
        ) ?? [];
      const fallbackSources = [
        { source: "pipeline", text: pipelineLog },
        { source: "execute", text: executeLog },
        { source: "monitor", text: monitorLog },
      ];
      const logSources = pythonAnyLogSources.length ? pythonAnyLogSources : fallbackSources;
      setLogEntries(buildLogEntries(logSources, 400, 48));
      inFlight = false;
    };

    void load();
    const timer = globalThis.setInterval(() => void load(), 20000);
    return () => {
      globalThis.clearInterval(timer);
      isMounted = false;
    };
  }, []);

  const liveTradingStatus = useLiveTradingStatus(
    parseTradingStatus(healthSnapshot)
  );
  const rightBadges = useMemo(
    () => buildNavbarBadges(liveTradingStatus),
    [liveTradingStatus]
  );

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
    return `Diagnostic artifacts only; accounting / entry readiness separate. Last run: ${lastRun}${source ? ` / ${source}` : ""}`;
  }, [healthSnapshot?.last_run_utc, healthSnapshot?.latest_source]);

  const monitoringPositionsFromApi = useMemo(() => {
    const positions = monitoringSnapshot?.positions;
    if (!positions || positions.length === 0) {
      return [];
    }
    return positions.map((position) => {
      const symbol = normalizeSymbol(position.symbol ?? null);
      const logoUrl = position.logoUrl ?? logoUrlForSymbol(symbol);
      return {
        symbol,
        logoUrl,
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
      };
    });
  }, [monitoringSnapshot?.positions]);
  const hasMonitoringApi = monitoringSnapshot?.ok === true && Array.isArray(monitoringSnapshot.positions);

  const pipelineLogRun = useMemo(
    () => parsePipelineLogRun(pipelineLogText),
    [pipelineLogText]
  );
  const executeLogSummary = useMemo(
    () => parseExecuteLogSummary(executeLogText),
    [executeLogText]
  );

  const openPositions = useMemo<OpenPositionsSummary>(() => ({
    count: hasMonitoringApi ? monitoringPositionsFromApi.length : null,
    pnl: hasMonitoringApi ? parseNumber(monitoringSnapshot?.summary?.totalOpenPL) : null,
    pnlPct: null, filledCount: null, totalCount: null,
  }), [hasMonitoringApi, monitoringPositionsFromApi.length, monitoringSnapshot?.summary?.totalOpenPL]);
  const tradesMetrics = useMemo(() => {
    const kpis = tradesOverview?.performance_status === "VALIDATED_PERFORMANCE" ? tradesOverview.metrics : null;
    return {totalTrades: parseNumber(kpis?.total_trades), winRate: normalizeWinRate(parseNumber(kpis?.win_rate)), netPnl: parseNumber(kpis?.net_pnl)};
  }, [tradesOverview]);

  const buyingPowerValue = parseNumber(
    accountSnapshot?.buying_power ?? null
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

    const ordersSubmitted = executeSnapshot?.orders_submitted;
    const ordersFilled = (executeOrdersSummary?.ok === true ? executeOrdersSummary.orders_filled : null) ?? executeSnapshot?.orders_filled;
    const ordersPlaced = typeof ordersSubmitted === "number" ? ordersSubmitted : Number.NaN;
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
        totalValue: executeOrdersSummary?.ok === true && executeOrdersSummary.source === "alpaca" && executeOrdersSummary.total_value !== null && executeOrdersSummary.total_value !== undefined
          ? executeOrdersSummary.total_value : Number.NaN,
        successRate,
        isCycleComplete: executeCycleComplete,
        marketNote: executeMarketNote,
      },
      monitoring: {
        positions: hasMonitoringApi ? monitoringPositionsFromApi : [],
        available: hasMonitoringApi, sourceStatus: monitoringSource.kind, observedAt: monitoringSnapshot?.observed_at,
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
    executeOrdersSummary?.ok,
    executeOrdersSummary?.source,
    executeOrdersSummary?.orders_filled,
    executeOrdersSummary?.total_value,
    healthSnapshot?.last_run_utc,
    healthSnapshot?.pipeline_rc,
    healthSnapshot?.trading_ok,
    monitoringPositionsFromApi,
    hasMonitoringApi,
    monitoringSource.kind,
    monitoringSnapshot?.observed_at,
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
        title: "Pipeline / API diagnostics",
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
        footnote: `Source: Alpaca | ${monitoringSource.kind} | As of ${formatDateTime(monitoringSnapshot?.observed_at)}`,
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
      monitoringSource.kind,
      monitoringSnapshot?.observed_at,
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
    const tradingOk = parseTradingStatus(healthSnapshot);
    const dataOk = diagnosticsFresh(healthSnapshot) ? healthSnapshot?.data_ok ?? null : null;
    const pipelineRc = healthSnapshot?.pipeline_rc ?? null;
    const feed = healthSnapshot?.feed ? healthSnapshot.feed.toUpperCase() : null;

    const apiTone: StatusTone = tradingOk === false ? "error" : tradingOk === true ? "success" : "neutral";
    const dataTone: StatusTone = dataOk === false ? "warning" : dataOk === true ? "success" : "neutral";
    const pipelineTone: StatusTone =
      pipelineRc === 0 ? "success" : pipelineRc === null ? "neutral" : "warning";

    return [
      {
        title: "Saved API diagnostic",
        status: tradingOk === true ? "Stored check passed" : tradingOk === false ? "Stored check failed" : "Unknown / stale",
        tone: apiTone,
        description: "Saved operational check; live broker readiness is separate",
        meta: `Status: ${healthSnapshot?.trading_status ?? "n/a"}`,
      },
      {
        title: "Market Data Feed",
        status: dataOk === true ? "Stored check passed" : dataOk === false ? "Stored check failed" : "Unknown / stale",
        tone: dataTone,
        description: feed ? `Feed: ${feed}` : "Market data feed",
        meta: `Status: ${healthSnapshot?.data_status ?? "n/a"}`,
      },
      {
        title: "Recorded Pipeline computation",
        status: pipelineRc === 0 ? "Recorded rc=0" : pipelineRc === null ? "Unknown" : "Recorded failure",
        tone: pipelineTone,
        description: healthSnapshot?.run_type
          ? `Run type: ${healthSnapshot.run_type}`
          : "Run type: n/a",
        meta: `Rows: ${formatNumber(healthSnapshot?.rows_final ?? null)}`,
      },
    ];
  }, [
    healthSnapshot?.freshness,
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
    { time: "--", level: "INFO", message: "No recent activity logged." },
  ];

  const displayEntries = logEntries.length ? logEntries : emptyEntries;
  const alpacaEquity = formatCurrency(
    parseNumber(accountSnapshot?.equity ?? accountSnapshot?.portfolio_value ?? null)
  );
  const latestLogsRows = displayEntries.map((entry) => ({
    dateTime: entry.time,
    script: entry.source ?? "--",
    text: entry.message,
    level: entry.level,
  }));
  const pythonAnywhereResourceRows = useMemo(() => {
    if (!pythonAnywhereResources?.length) {
      return pythonAnywhereResourceDefaults;
    }
    return pythonAnywhereResourceDefaults.map((fallback) => {
      const match = pythonAnywhereResources.find(
        (resource) => resource.label.toLowerCase() === fallback.label.toLowerCase()
      );
      return match ? { label: fallback.label, value: match.value } : fallback;
    });
  }, [pythonAnywhereResources]);
  const pythonAnywhereTaskRows = useMemo(() => {
    const source = pythonAnywhereTasks?.length ? pythonAnywhereTasks : pythonAnywhereTasksDefaults;
    return source;
  }, [pythonAnywhereTasks]);

  const currentTab = activeTab ?? "Dashboard";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-36 pb-12 sm:pt-28">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl dark:from-cyan-500/15 dark:via-slate-950/40 dark:to-amber-500/20 dark:opacity-70" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl dark:from-emerald-500/15 dark:via-slate-950/40 dark:to-cyan-500/15 dark:opacity-70" />

        <div className="relative mx-auto max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <p role="status" className="pt-4 text-xs text-secondary">
            Alpaca account: {accountPayloadSource.kind} (as of {String(accountPayloadSource.payload?.taken_at_utc ?? "unknown")}) |
            PostgreSQL performance: {tradesPayloadSource.kind} | Broker positions: {monitoringSource.kind} |
            Execution snapshot: {executePayloadSource.kind}. Stale values are last known observations.
          </p>
          <section className="pt-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">Process Status Cards</h2>
            </div>
            <div className="mt-4">
              <ProcessStatusCards {...processStatusCards} />
            </div>
          </section>

          <section className="mt-6">
            <div className="grid gap-4 lg:grid-cols-3">
              <AlpacaCard value={alpacaEquity} subtitle={`Broker equity: ${accountPayloadSource.kind}`} />
              <PythonAnywhereResources resources={pythonAnywhereResourceRows} />
              <MLDailyImprovements onOpenPipeline={() => onTabSelect?.("ML Pipeline")} />
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <LatestLogsTable rows={latestLogsRows} />
              <PythonAnywhereTasksTable rows={pythonAnywhereTaskRows} />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}



