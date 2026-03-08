import { useEffect, useState } from "react";
import type { LogEntry, StatusTone } from "../../types/ui";

export type MlStage = {
  label: string;
  status: string;
  tone: StatusTone;
  timestamp?: string | null;
  detail?: string | null;
};

export type MlOverviewResponse = {
  ok?: boolean;
  updated_at?: string | null;
  status?: {
    label?: string | null;
    tone?: StatusTone | null;
    detail?: string | null;
    last_ml_run?: string | null;
  };
  freshness?: {
    label?: string | null;
    stale?: boolean | null;
    reason?: string | null;
    model_path?: string | null;
    pred_model_path?: string | null;
    latest_features_set?: string | null;
    latest_features_signature?: string | null;
    pred_features_set?: string | null;
    pred_features_signature?: string | null;
    pred_compatible?: boolean | null;
    pred_missing_frac?: number | null;
    pred_compat_reason?: string | null;
    generated_at_utc?: string | null;
    snapshot_date?: string | null;
    latest_auto_refresh?: Record<string, unknown> | null;
  };
  coverage?: {
    total?: number | null;
    non_null?: number | null;
    pct?: number | null;
    source?: string | null;
    run_ts_utc?: string | null;
  };
  champion?: {
    present?: boolean;
    source?: string | null;
    run_date?: string | null;
    status?: string | null;
    calibration?: string | null;
    feature_set?: string | null;
    bars_adjustment?: string | null;
    split_adjust?: string | null;
    top_k?: number | null;
    execution?: {
      min_model_score?: number | null;
      require_model_score?: boolean | null;
    };
  };
  monitor?: {
    present?: boolean;
    run_date?: string | null;
    recommended_action?: string | null;
    psi_score?: number | null;
    recent_sharpe?: number | null;
    ece?: number | null;
    delta_ece?: number | null;
    guard_decision?: string | null;
    guard_mode?: string | null;
    guard_reasons?: string[];
  };
  eval?: {
    present?: boolean;
    run_date?: string | null;
    signal_quality?: string | null;
    sample_size?: number | null;
    decile_lift?: number | null;
    top_avg_label?: number | null;
    bottom_avg_label?: number | null;
  };
  remediation?: {
    present?: boolean;
    last_action?: string | null;
    last_kind?: string | null;
    executed?: boolean | null;
    run_date?: string | null;
    repredict_executed?: boolean | null;
    repredict_rc?: number | null;
    repredict_reason?: string | null;
    features_refresh_attempted?: boolean | null;
  };
  timestamps?: {
    last_predict?: string | null;
    last_eval?: string | null;
    last_monitor?: string | null;
    last_recalibrate?: string | null;
    last_autoremediate?: string | null;
    last_trade_attribution?: string | null;
    last_model?: string | null;
    last_features?: string | null;
    last_labels?: string | null;
    last_ml_run?: string | null;
  };
  overlap?: {
    candidates?: number | null;
    prediction_symbols?: number | null;
    overlap?: number | null;
    run_ts_utc?: string | null;
    run_date?: string | null;
    score_col?: string | null;
    pred_ts_min?: string | null;
    pred_ts_max?: string | null;
    sample_reason?: string | null;
    missing_symbols?: string[];
  };
  enrichment?: {
    latest_result?: Record<string, unknown> | null;
    latest_skip?: Record<string, unknown> | null;
    latest_decision?: Record<string, unknown> | null;
    candidate_refresh?: Record<string, unknown> | null;
    candidate_refresh_done?: Record<string, unknown> | null;
  };
  trade_attribution?: {
    present?: boolean;
    status?: string | null;
    trades_scored?: number | null;
    trades_total?: number | null;
    win_rate_scored?: number | null;
    brier?: number | null;
  };
  pipeline_stages?: MlStage[];
  recent_events?: Array<{
    timestamp?: string | null;
    level?: LogEntry["level"] | null;
    token?: string | null;
    message?: string | null;
  }>;
};

type MlOverviewState = {
  data: MlOverviewResponse | null;
  isLoading: boolean;
  hasError: boolean;
};

const fetchMlOverview = async (): Promise<MlOverviewResponse | null> => {
  try {
    const response = await fetch(`/api/ml/overview?ts=${Date.now()}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as MlOverviewResponse;
  } catch {
    return null;
  }
};

export function useMlOverview(): MlOverviewState {
  const [data, setData] = useState<MlOverviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setIsLoading(true);
      const payload = await fetchMlOverview();
      if (!isMounted) {
        return;
      }
      setData(payload);
      setHasError(payload === null);
      setIsLoading(false);
    };

    load();
    return () => {
      isMounted = false;
    };
  }, []);

  return { data, isLoading, hasError };
}
