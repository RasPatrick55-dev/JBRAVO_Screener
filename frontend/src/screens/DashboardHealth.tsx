import { canDisplay, useSource } from "../components/ui/sourceState";
import { useMemo } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import { buildNavbarBadges, useLiveTradingStatus, parseTradingStatus, diagnosticsFresh } from "../components/navbar/liveStatus";
import ProcessStatusCards, {
  type ProcessStatusCardsProps,
} from "../components/dashboard/ProcessStatusCards";
import AlpacaCard from "../components/dashboard/AlpacaCard";
import { buyFillSummary, type ExecuteOrdersSummaryResponse } from "../components/dashboard/buyFillSummary";
import PythonAnywhereResources, { resourceRows, type PythonAnywhereResource } from "../components/dashboard/PythonAnywhereResources";
import MLDailyImprovements from "../components/dashboard/MLDailyImprovements";
import LatestLogsTable from "../components/dashboard/LatestLogsTable";
import { useProviderLogs, providerLogNotice, PROVIDER_READ_TIMEOUT_MS, type ProviderLogSource } from "../components/dashboard/providerLogs";
import PythonAnywhereTasksTable, { scheduledTaskRows } from "../components/dashboard/PythonAnywhereTasksTable";
import type { LogEntry, StatusTone, SystemStatusItem } from "../types/ui";
import { isCurrentReceipt, receiptStatus, type ProcessReceipt } from "../components/dashboard/processReceipt";

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

type PipelineTaskRunResponse = ProcessReceipt & {
  ok?: boolean;
  started_utc?: string | null;
  finished_utc?: string | null;
  duration_seconds?: number | null;
  rc?: number | null;
  source?: string | null;
};

type ExecuteTaskRunResponse = ProcessReceipt & {
  ok?: boolean;
  started_utc?: string | null;
  finished_utc?: string | null;
  duration_seconds?: number | null;
  rc?: number | null;
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
  observed_at?: string | null;
  tasks?: PythonAnywhereTask[];
  source?: string | null;
};

type PythonAnywhereLogSource = ProviderLogSource;

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

export const buildLogEntries = (
  sources: Array<{ text: string | null; source: string; entries?: PythonAnywhereLogSource["entries"] }>,
  maxLinesPerSource = 240,
  _maxAgeHours = 48
) => {
  const entries: ParsedLogEntry[] = [];

  sources.forEach(({ text, source, entries: evidence }) => {
    if (!text) {
      return;
    }
    const sourceLabel = labelLogSource(source);
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const sample = lines.slice(-Math.max(maxLinesPerSource, 12));
    sample.forEach((line) => {
      const parsed = parseLogLine(line);
      const metadata = evidence?.find((entry) => entry.raw === line);
      const provenanceLabel = metadata?.provenance_label ?? "Unclassified · not broker evidence";
      const fallbackTimestamp = parsed ? `${parsed.date}T${parsed.time}Z` : null;
      const eventTimestamp = metadata ? metadata.ts_utc ?? null : fallbackTimestamp;
      const timestampMs = eventTimestamp ? Date.parse(eventTimestamp) || 0 : 0;
      const canonicalTime = timestampMs ? new Date(timestampMs).toISOString() : null;
      const displayTime = canonicalTime ? formatLogDateTime(canonicalTime.slice(0, 10), canonicalTime.slice(11, 19)) : "--";
      if (parsed) {
        const cleanedMessage = parsed.message.trim().replace(/^[\s:=-]+/, "").trim();
        entries.push({
          time: displayTime,
          level: normalizeLogLevel(parsed.level),
          message: cleanedMessage || "(no message)",
          source: sourceLabel,
          provenanceLabel,
          eventTimestamp,
          timestampMs,
        });
      } else {
        entries.push({
          time: displayTime,
          level: "INFO",
          message: line.trim(),
          source: sourceLabel,
          provenanceLabel,
          eventTimestamp,
          timestampMs,
        });
      }
    });
  });

  let filtered = entries;
  // Retain historical and untimed evidence; label age rather than silently dropping it.

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
  const healthSource = useSource("/api/health");
  const healthSnapshot = healthSource.freshness === "current" ? healthSource.payload as ApiHealthResponse | null : null;
  const accountPayloadSource = useSource("/api/account/overview");
  const accountSnapshot = canDisplay(accountPayloadSource.kind) ? (accountPayloadSource.payload as AccountOverviewResponse | null)?.snapshot ?? null : null;
  const tradesPayloadSource = useSource("/api/trades/overview");
  const tradesOverview = canDisplay(tradesPayloadSource.kind) ? tradesPayloadSource.payload as TradesOverviewResponse | null : null;
  const executePayloadSource = useSource("/api/execute/overview");
  const executeSnapshot = canDisplay(executePayloadSource.kind) ? executePayloadSource.payload as ExecutionSnapshot | null : null;
  const monitoringSource = useSource("/api/positions/monitoring");
  const monitoringSnapshot = canDisplay(monitoringSource.kind) ? monitoringSource.payload as MonitoringPositionsResponse | null : null;
  const pipelineTaskSource = useSource("/api/pipeline/task");
  const pipelineTaskRun = pipelineTaskSource.payload as PipelineTaskRunResponse | null;
  const executeTaskSource = useSource("/api/execute/task");
  const executeTaskRun = executeTaskSource.payload as ExecuteTaskRunResponse | null;
  const executeOrdersSource = useSource("/api/execute/orders-summary");
  const executeOrdersSummary = executeOrdersSource.freshness === "current"
    ? executeOrdersSource.payload as ExecuteOrdersSummaryResponse | null : null;
  const resourcesSource = useSource("/api/pythonanywhere/resources", 20000, PROVIDER_READ_TIMEOUT_MS);
  const resourcesPayload = resourcesSource.payload as PythonAnywhereResourcesResponse | null;
  const pythonAnywhereResources = resourcesPayload?.ok === true ? resourcesPayload.resources ?? null : null;
  const tasksSource = useSource("/api/pythonanywhere/tasks", 20000, PROVIDER_READ_TIMEOUT_MS);
  const pythonAnywhereTasksPayload = tasksSource.payload as PythonAnywhereTasksResponse | null;
  const pythonAnywhereTasks = scheduledTaskRows(pythonAnywhereTasksPayload);
  const pythonAnywhereTasksObservedAt = pythonAnywhereTasksPayload?.ok === true ? pythonAnywhereTasksPayload.observed_at ?? null : null;
  const providerLogs = useProviderLogs();
  const logEntries = useMemo(() => buildLogEntries(providerLogs.sources, 400, 48), [providerLogs.sources]);
  const logReceipts = useMemo(() => providerLogs.sources.map(source => ({ ...source, source: labelLogSource(source.source) })), [providerLogs.sources]);

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
    const currentPipeline = isCurrentReceipt(pipelineTaskRun);
    const currentExecutor = isCurrentReceipt(executeTaskRun);
    const pipelineState = receiptStatus(pipelineTaskRun) ?? "UNAVAILABLE";
    const executorState = receiptStatus(executeTaskRun) ?? "UNAVAILABLE";
    const pipelineStart = currentPipeline ? pipelineTaskRun?.started_utc ?? null : null;
    const pipelineEnd = currentPipeline ? pipelineTaskRun?.finished_utc ?? null : null;
    const pipelineDuration = currentPipeline && pipelineTaskRun?.duration_seconds != null
      ? formatDurationSeconds(pipelineTaskRun.duration_seconds)
      : formatDuration(pipelineStart, pipelineEnd);
    const pipelineLive = pipelineState === "COMPLETED";
    const executeStart = currentExecutor ? executeTaskRun?.started_utc ?? null : null;
    const executeEnd = currentExecutor ? executeTaskRun?.finished_utc ?? null : null;
    const executeDuration =
      currentExecutor && executeTaskRun?.duration_seconds !== null && executeTaskRun?.duration_seconds !== undefined
        ? formatDurationSeconds(executeTaskRun.duration_seconds)
        : formatDuration(executeStart, executeEnd);
    const executeCycleComplete = executorState === "COMPLETED";

    const ordersSubmitted = executeSnapshot?.orders_submitted;
    const ordersFilled = executeSnapshot?.orders_filled;
    const fills24h = buyFillSummary(executeOrdersSummary);
    const ordersPlaced = fills24h.count;
    const successRate =
      typeof ordersSubmitted === "number" &&
      ordersSubmitted > 0 &&
      typeof ordersFilled === "number"
        ? (ordersFilled / ordersSubmitted) * 100
        : Number.NaN;

    const executeMarketNote = null; // A historical log cannot describe the current market window.

    return {
      pipeline: {
        lastRun: {
          date: formatDateUtc(pipelineStart ?? pipelineEnd),
          start: formatTimeUtc(pipelineStart),
          end: formatTimeUtc(pipelineEnd),
          duration: pipelineDuration,
        },
        subprocess: {
          screener: statusFromPipelineRc(currentPipeline && pipelineTaskRun?.ok === true ? pipelineTaskRun.stage_rcs?.screener ?? null : null),
          backTester: statusFromPipelineRc(currentPipeline && pipelineTaskRun?.ok === true ? pipelineTaskRun.stage_rcs?.backtest ?? null : null),
          metrics: statusFromPipelineRc(currentPipeline && pipelineTaskRun?.ok === true ? pipelineTaskRun.stage_rcs?.metrics ?? null : null),
        },
        isLive: pipelineLive,
        status: pipelineState,
        taskSource: pipelineTaskSource,
      },
      executeTrades: {
        lastRun: {
          date: formatDateUtc(executeStart ?? executeEnd),
          start: formatTimeUtc(executeStart),
          end: formatTimeUtc(executeEnd),
          duration: executeDuration,
        },
        ordersPlaced,
        totalValue: fills24h.value,
        successRate,
        isCycleComplete: executeCycleComplete,
        status: executorState,
        marketNote: executeMarketNote,
        lastScheduledAttempt: executeTaskRun?.last_scheduled_attempt,
        taskSource: executeTaskSource,
      },
      monitoring: {
        positions: hasMonitoringApi ? monitoringPositionsFromApi : [],
        available: hasMonitoringApi, sourceStatus: monitoringSource.kind, observedAt: monitoringSnapshot?.observed_at,
      },
    };
  }, [
    pipelineTaskRun,
    pipelineTaskSource,
    executeTaskRun,
    executeSnapshot?.in_window,
    executeSnapshot?.last_execution,
    executeSnapshot?.ny_now,
    executeSnapshot?.orders_filled,
    executeSnapshot?.orders_submitted,
    executeTaskRun?.duration_seconds,
    executeTaskRun?.finished_utc,
    executeTaskRun?.started_utc,
    executeTaskRun?.rc,
    executeTaskRun?.last_scheduled_attempt,
    executeTaskSource.kind,
    executeTaskSource.freshness,
    executeTaskSource.receivedAt,
    executeOrdersSummary,
    executeOrdersSummary?.source,
    executeOrdersSummary?.fill_count,
    executeOrdersSummary?.total_value,
    healthSnapshot?.last_run_utc,
    healthSnapshot?.pipeline_rc,
    healthSnapshot?.trading_ok,
    monitoringPositionsFromApi,
    hasMonitoringApi,
    monitoringSource.kind,
    monitoringSnapshot?.observed_at,
    pipelineTaskRun?.duration_seconds,
    pipelineTaskRun?.finished_utc,
    pipelineTaskRun?.rc,
    pipelineTaskRun?.started_utc,
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

  const alpacaEquity = formatCurrency(
    parseNumber(accountSnapshot?.equity ?? accountSnapshot?.portfolio_value ?? null)
  );
  const latestLogsRows = logEntries.map((entry) => ({
    dateTime: entry.time,
    script: entry.source ?? "--",
    text: entry.message,
    provenanceLabel: entry.provenanceLabel,
    eventTimestamp: entry.eventTimestamp,
    level: entry.level,
  }));
  const pythonAnywhereResourceRows = useMemo(() => resourceRows(pythonAnywhereResources), [pythonAnywhereResources]);
  const pythonAnywhereTaskRows = pythonAnywhereTasks ?? [];

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
              <PythonAnywhereResources resources={pythonAnywhereResourceRows} sourceStatus={resourcesSource.kind} />
              <MLDailyImprovements onOpenPipeline={() => onTabSelect?.("ML Pipeline")} />
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <LatestLogsTable rows={latestLogsRows} receipts={logReceipts} notice={providerLogNotice(providerLogs)} emptyMessage={providerLogNotice(providerLogs)} />
              <PythonAnywhereTasksTable rows={pythonAnywhereTaskRows} observedAt={pythonAnywhereTasksObservedAt} sourceStatus={tasksSource.kind} />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}



