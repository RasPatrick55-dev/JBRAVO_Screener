export type ExecuteStatusScope = "all" | "open" | "closed";
export type ExecuteLsxFilter = "all" | "l" | "s" | "e" | "x";
export type ExecuteLogStage = "execute" | "monitor" | "pipeline";
export type ExecuteLogLevelFilter = "all" | "errors" | "warnings";

export interface ExecuteSummaryResponse {
  ok?: boolean;
  last_run_utc?: string | null;
  in_window?: boolean;
  candidates?: number | null;
  submitted?: number | null;
  filled?: number | null;
  rejected?: number | null;
  result_pl_usd?: number | null;
  source?: string;
  source_detail?: string;
}

export interface ExecuteOrderRow {
  ts_utc?: string | null;
  symbol?: string | null;
  side?: string | null;
  type?: string | null;
  qty?: number | string | null;
  limit_stop_trail?: string | null;
  status?: string | null;
  filled_avg?: number | string | null;
  order_id?: string | null;
  notes?: string | null;
}

export interface ExecuteOrdersResponse {
  ok?: boolean;
  rows?: ExecuteOrderRow[];
  source?: string;
  source_detail?: string;
}

export interface ExecuteTrailingStopRow {
  ts_utc?: string | null;
  symbol?: string | null;
  qty?: number | string | null;
  trail?: string | null;
  stop_price?: number | string | null;
  status?: string | null;
  parent_leg?: string | null;
}

export interface ExecuteTrailingStopsResponse {
  ok?: boolean;
  rows?: ExecuteTrailingStopRow[];
  source?: string;
  source_detail?: string;
}

export interface ExecuteLogRow {
  ts_utc?: string | null;
  level?: string | null;
  message?: string | null;
}

export interface ExecuteLogsResponse {
  ok?: boolean;
  stage?: ExecuteLogStage;
  rows?: ExecuteLogRow[];
  source?: string;
  source_detail?: string;
}

export interface ExecuteStateResponse {
  ok?: boolean;
  ts_utc?: string | null;
  summary?: ExecuteSummaryResponse;
  orders?: ExecuteOrdersResponse;
  trailing_stops?: ExecuteTrailingStopsResponse;
  logs?: {
    execute?: ExecuteLogsResponse;
    monitor?: ExecuteLogsResponse;
    pipeline?: ExecuteLogsResponse;
  };
}

export interface ExecuteAuditFinding {
  severity?: "high" | "warning" | "info" | string;
  code?: string;
  message?: string;
  details?: string;
  value?: number | string | null;
}

export interface ExecuteAuditResponse {
  ok?: boolean;
  fetched_at_utc?: string | null;
  limit?: number;
  row_counts?: Record<string, number>;
  summary?: {
    last_run_utc?: string | null;
    candidates?: number | null;
    submitted?: number | null;
    filled?: number | null;
    rejected?: number | null;
    source?: string;
    source_detail?: string;
  };
  quality?: Record<string, number>;
  cross_checks?: {
    orders_subset?: {
      all_scope_limited?: boolean;
      open_subset_of_all?: boolean | null;
      closed_subset_of_all?: boolean | null;
      open_closed_intersection_count?: number;
      open_not_in_all?: number;
      closed_not_in_all?: number;
    };
    freshness?: {
      orders_newest_age_seconds?: number | null;
      execute_logs_newest_age_seconds?: number | null;
    };
  };
  severity_counts?: {
    high?: number;
    warning?: number;
    info?: number;
  };
  findings?: ExecuteAuditFinding[];
}
