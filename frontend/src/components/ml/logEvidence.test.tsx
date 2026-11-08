import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { formatMlReturnCode } from "./displayValues";
import { useMlOverview, type MlOverviewResponse } from "./useMlOverview";
import MLPipeline from "../../screens/MLPipeline";
import MLDailyImprovements from "../dashboard/MLDailyImprovements";
import { predictionFreshness, mlStagePresentation, successfulPredictionExecution } from "./freshness";

vi.mock("./useMlOverview", () => ({ useMlOverview: vi.fn() }));
vi.mock("../navbar/liveStatus", () => ({ useLiveTradingStatus: () => null, buildNavbarBadges: () => [] }));
vi.mock("../navbar/NavbarDesktop", () => ({ default: () => null }));

function render(data: MlOverviewResponse) {
  vi.mocked(useMlOverview).mockReturnValue({ data, isLoading: false, hasError: false });
  return renderToStaticMarkup(<MLPipeline />);
}
afterEach(() => vi.clearAllMocks());

describe("ML log diagnostic evidence", () => {
  it.each([0, 1, 2, -1])("renders numeric return code %s", value => {
    expect(formatMlReturnCode(value)).toBe(String(value));
    expect(render({ enrichment: { candidate_refresh_done: { rc: value } } }))
      .toMatch(new RegExp(`Historical refresh return code[\\s\\S]*?>${value}</div>`));
  });
  it.each([false, true, null, undefined, "0", "false", NaN, Infinity, 0.5])("keeps malformed return code %s unknown", value => {
    expect(formatMlReturnCode(value)).toBe("Unknown");
    expect(render({ enrichment: { candidate_refresh_done: { rc: value } } }))
      .toMatch(/Historical refresh return code[\s\S]*?>Unknown<\/div>/);
  });
  it("renders independent diagnostic dates instead of attaching old missing symbols to refresh", () => {
    const html = render({
      enrichment: { candidate_refresh_done: { rc: 0, predictions_source: "db:present",
        evidence: { timestamp: "2026-10-24T03:58:31Z", source: "logs/pipeline.log" } } },
      overlap: { overlap: 7, candidates: 7, prediction_symbols: 9,
        evidence: { timestamp: "2026-10-25T04:04:58Z", source: "logs/pipeline.log" },
        latest_sample: { reason: "no_prediction_for_symbol", missing_symbols: ["AAA", "BBB"],
          evidence: { timestamp: "2026-10-24T03:57:41Z", source: "logs/pipeline.log" } } },
    });
    expect(html).toContain("Oct 25, 2026, 04:04 UTC");
    expect(html).toContain("Oct 24, 2026, 03:58 UTC");
    expect(html).toContain("Oct 24, 2026, 03:57 UTC");
    expect(html).toContain("Historical missing-symbol sample");
    expect(html).toContain("This sample is not bound to the overlap or refresh above.");
    expect(html).toContain("not current readiness");
    expect(html).toContain("Reported predictions source db:present");
    expect(html).toMatch(/Historical overlap[\s\S]*?>7 of 7<\/div>/);
    expect(html).toMatch(/Historical refresh return code[\s\S]*?>0<\/div>/);
    expect(html).not.toContain("Sample reason");
  });
  it("does not display unbound legacy sample fields as the refresh result", () => {
    const html = render({ overlap: { sample_reason: "legacy_warning", missing_symbols: ["OLD"] } });
    expect(html).not.toContain("legacy_warning");
    expect(html).not.toContain("Historical missing-symbol sample");
    expect(html).toContain("Source unknown");
  });
  it("does not trust legacy log-only execution or filesystem freshness claims", () => {
    const data: MlOverviewResponse = {
      ok: true, last_successful_execution: "2026-11-06T23:00:01Z",
      execution_evidence_scope: "PREDICTION_STAGE_ONLY",
      freshness: { source: "fs", status: "fresh", stale: false },
      pipeline_stages: [{ label: "Predict", status: "success", tone: "success",
        artifact_present: true, source: "fs", execution_status: "verified", freshness_status: "fresh" }],
    };
    expect(predictionFreshness(data.freshness)).toBe("Unknown");
    expect(successfulPredictionExecution(data)).toBeNull();
    expect(mlStagePresentation(data.pipeline_stages![0])).toMatchObject({ status: "Diagnostic only", execution_status: "unverified", tone: "neutral" });
    const pipeline = render(data);
    const dashboard = renderToStaticMarkup(<MLDailyImprovements />);
    for (const html of [pipeline, dashboard]) {
      expect(html).toContain("Diagnostic only");
      expect(html).not.toContain(">success<");
    }
    expect(pipeline).toMatch(/Last verified prediction execution[\s\S]*?>Unknown<\/div>/);
  });
  it("keeps database artifact metadata freshness separate from current execution", () => {
    const data: MlOverviewResponse = { freshness: { source: "db", source_authority: "postgresql", status: "fresh", stale: false },
      status: { label: "Unknown", tone: "neutral" },
      coverage: { pct: 100, non_null: 1, total: 1, diagnostic_only: true,
        evidence: { timestamp: "2026-10-01T10:00:00Z", source: "logs/pipeline.log" } } };
    expect(predictionFreshness(data.freshness)).toBe("Fresh");
    expect(successfulPredictionExecution(data)).toBeNull();
    const pipeline = render(data);
    const dashboard = renderToStaticMarkup(<MLDailyImprovements />);
    expect(pipeline).toContain("Prediction artifact freshness");
    expect(pipeline).toContain("Historical score coverage");
    expect(dashboard).toContain("Historical coverage");
    expect(dashboard).toContain("current coverage unverified");
    for (const html of [pipeline, dashboard]) {
      expect(html).toContain("Oct 1, 2026, 10:00 UTC");
      expect(html).not.toContain("Predictions fresh");
    }
  });
  it("keeps a missing filesystem fallback missing rather than inventing available diagnostics", () => {
    expect(predictionFreshness({ source: "fs", source_authority: "diagnostic_only", status: "missing" })).toBe("Missing");
    const stage = { label: "Predict", source: "fs", artifact_present: false, status: "blocked", tone: "neutral" as const };
    expect(mlStagePresentation(stage)).toBe(stage);
  });

});
