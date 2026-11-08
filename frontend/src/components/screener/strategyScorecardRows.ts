import type { SourceResult } from "../ui/sourceState";
import { evidenceCount, evidenceNumber, evidenceText, isDigest, isRecord, isStrategyId, isVersion } from "./strategyEvidence";

export const STRATEGY_SCORECARD_PATH = "/api/strategies/scorecard";
export type StrategyScorecardRow = ReturnType<typeof normalizeScorecardRow>;
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((x): x is string => typeof x === "string") : [];
const record = (value: unknown) => isRecord(value) ? value : {};

function normalizeScorecardRow(value: unknown) {
  if (!isRecord(value) || !isStrategyId(value.strategy_id)) {
    throw new Error("Strategy scorecard identities are unavailable.");
  }
  const row = value;
  const evidence = evidenceText(row.evidence_status);
  const usable = evidence === "VERIFIED";
  if (usable && (!isVersion(row.strategy_version) || !isDigest(row.definition_sha256) || !isDigest(row.episode_id))) {
    throw new Error("Strategy scorecard evidence is missing its definition or episode identity.");
  }
  const costs = record(row.costs), uncertainty = record(row.uncertainty), proof = record(row.evidence), episode = record(row.episode), freshness = record(row.freshness);
  const number = (field: string) => usable ? evidenceNumber(row[field]) : null;
  const count = (field: string) => usable ? evidenceCount(row[field]) : null;
  const closed = count("closed_signals");
  const sampleMetric = (field: string) => closed !== null && closed > 0 ? number(field) : null;
  const intervalLow = evidenceNumber(uncertainty.low), intervalHigh = evidenceNumber(uncertainty.high);
  const level = evidenceNumber(uncertainty.level);
  const intervalValid = usable && closed !== null && closed > 0 && uncertainty.status === "DESCRIPTIVE_NOT_VALIDATION"
    && intervalLow !== null && intervalHigh !== null && intervalLow <= intervalHigh && level !== null && level > 0 && level < 1;
  return {
    key: JSON.stringify([row.strategy_id, row.strategy_version, row.definition_sha256, row.episode_id]),
    strategy_id: evidenceText(row.strategy_id), strategy_name: evidenceText(row.strategy_name), strategy_version: evidenceText(row.strategy_version),
    definition_sha256: evidenceText(row.definition_sha256), episode_id: evidenceText(row.episode_id),
    as_of: evidenceText(row.as_of), scope: evidenceText(row.scope), evidence_status: evidence || "UNAVAILABLE",
    sample_status: evidenceText(row.sample_status), closed_signals: closed, open_signals: count("open_signals"), unfilled_signals: count("unfilled_signals"),
    wins: count("wins"), losses: count("losses"), breakeven: count("breakeven"), win_rate_pct: sampleMetric("win_rate_pct"),
    net_pnl: number("net_pnl"), expectancy: sampleMetric("expectancy_net_per_signal"),
    average_win: sampleMetric("average_win"), average_loss: sampleMetric("average_loss"),
    payoff_ratio: sampleMetric("payoff_ratio"), profit_factor: row.profit_factor_status === "DEFINED" ? sampleMetric("profit_factor") : null,
    profit_factor_status: usable ? evidenceText(row.profit_factor_status) : "UNKNOWN",
    costs: { basis: evidenceText(costs.basis), fees: usable ? evidenceNumber(costs.fees) : null,
      slippage: usable ? evidenceNumber(costs.slippage) : null, total: usable ? evidenceNumber(costs.total) : null },
    uncertainty: { method: evidenceText(uncertainty.method), status: usable ? evidenceText(uncertainty.status) : "UNAVAILABLE",
      level: intervalValid ? level : null, low: intervalValid ? intervalLow : null, high: intervalValid ? intervalHigh : null,
      signal_weeks: usable ? evidenceCount(uncertainty.signal_weeks) : null, samples: usable ? evidenceCount(uncertainty.samples) : null },
    evidence: { archive_count: evidenceCount(proof.archive_count), distinct_signals: evidenceCount(proof.distinct_signals),
      duplicate_observations: evidenceCount(proof.duplicate_observations), archive_id: evidenceText(proof.archive_id),
      result_sha256: evidenceText(proof.result_sha256), scan_run_id: evidenceText(proof.scan_run_id), manifest_sha256: evidenceText(proof.manifest_sha256) },
    episode: { test_start: evidenceText(episode.test_start), as_of: evidenceText(episode.as_of),
      universe_kind: evidenceText(episode.universe_kind), cost_case: evidenceText(record(episode.cost_case).id),
      declared_symbols: evidenceCount(episode.declared_symbols) },
    freshness: { evidence_as_of: evidenceText(freshness.evidence_as_of), source_age_calendar_days: evidenceCount(freshness.source_age_calendar_days),
      latest_attempt_status: evidenceText(freshness.latest_attempt_status) },
    limitations: strings(row.limitations), reason: evidenceText(row.reason) || evidenceText(row.failure_code),
  };
}

export type ScorecardSource = Pick<SourceResult, "kind" | "payload"> & Partial<Pick<SourceResult, "freshness">>;
export function strategyScorecardPresentation(result: ScorecardSource) {
  let rows: StrategyScorecardRow[] = [], error: string | null = null;
  const payload = result.payload;
  const metadataOnly = result.kind === "unavailable" && result.freshness !== "stale" && payload?.status === "UNAVAILABLE"
    && payload.source === "postgres" && payload.metric_basis === "STRATEGY_RESEARCH_SIMULATION" && Array.isArray(payload.rows);
  if (result.kind !== "loading") {
    if (!["ready", "empty", "partial"].includes(result.kind) && !metadataOnly) error = `Strategy scorecard ${result.kind}; current evidence is not established.`;
    else if (!payload || (!metadataOnly && payload.ok !== true) || payload.metric_basis !== "STRATEGY_RESEARCH_SIMULATION" || !Array.isArray(payload.rows)
      || (!metadataOnly
      && !["COMPLETE", "PARTIAL", "EMPTY"].includes(evidenceText(payload.status)))) error = "Strategy-wide evidence is unavailable.";
    else {
      try {
        rows = payload.rows.map(raw => normalizeScorecardRow(metadataOnly && isRecord(raw)
          ? { ...raw, evidence_status: raw.evidence_status === "CONFLICT" ? "CONFLICT" : "UNAVAILABLE" } : raw));
        if (new Set(rows.map(row => row.key)).size !== rows.length) throw new Error("Duplicate strategy episodes cannot be presented as independent evidence.");
        if (metadataOnly) error = "Strategy-wide performance is unavailable. Strategy identities and missing-evidence reasons are shown below.";
      } catch (cause) { rows = []; error = cause instanceof Error ? cause.message : "Strategy-wide evidence is unavailable."; }
    }
  }
  const available = (!error || metadataOnly) && result.kind !== "loading";
  const coverage = available ? record(payload?.coverage) : {};
  return { rows, error, status: available ? evidenceText(payload?.status) : result.kind.toUpperCase(),
    limitations: available ? strings(payload?.limitations) : [],
    actualPaperReason: available ? evidenceText(record(payload?.actual_paper).reason) : "",
    coverage: { selection: evidenceText(coverage.selection), truncated: coverage.truncated === true,
      max_groups: evidenceCount(coverage.max_groups), rejected_archives: evidenceCount(coverage.rejected_archives),
      verified_groups: evidenceCount(coverage.verified_groups), unavailable_groups: evidenceCount(coverage.unavailable_groups),
      reasons: strings(coverage.reasons) },
    sync: result.kind === "loading" ? "loading" : error ? "error" : "ready" } as const;
}
