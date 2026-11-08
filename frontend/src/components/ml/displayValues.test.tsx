import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { formatRunDate, formatMlTimestamp, formatMlInstantDate, formatMlBoolean } from "./displayValues";
import { useMlOverview, type MlOverviewResponse } from "./useMlOverview";
import MLPipeline from "../../screens/MLPipeline";
import MLDailyImprovements from "../dashboard/MLDailyImprovements";

vi.mock("./useMlOverview", () => ({ useMlOverview: vi.fn() }));
vi.mock("../navbar/liveStatus", () => ({ useLiveTradingStatus: () => null, buildNavbarBadges: () => [] }));
vi.mock("../navbar/NavbarDesktop", () => ({ default: () => null }));

const render = (data: MlOverviewResponse | null) => {
  vi.mocked(useMlOverview).mockReturnValue({ data, isLoading: false, hasError: data === null });
  return { pipeline: renderToStaticMarkup(<MLPipeline />), dashboard: renderToStaticMarkup(<MLDailyImprovements />) };
};
afterEach(() => vi.clearAllMocks());

describe("ML display contracts", () => {
  it.each(["2026-02-28", "2024-02-29", "2000-02-29", "2026-12-31"])("preserves calendar identifier %s exactly", value => {
    expect(formatRunDate(value)).toBe(value);
  });
  it.each([null, undefined, "", "not-a-date", "2026-02-29", "1900-02-29", "2026-04-31", "2026-00-01", "2026-13-01", "2026-01-00", "2026-2-8", "2026-02-28T00:00:00Z", 20260228])("rejects invalid calendar identifier %s", value => {
    expect(formatRunDate(value)).toBe("Unknown");
  });
  it.each([
    ["2026-02-28T00:30:00+14:00", "2026-02-27 UTC", "Feb 27, 2026, 10:30 UTC"],
    ["2026-02-28T23:30:00-12:00", "2026-03-01 UTC", "Mar 1, 2026, 11:30 UTC"],
    ["2026-02-28T00:00:00Z", "2026-02-28 UTC", "Feb 28, 2026, 00:00 UTC"],
    ["2026-02-28T12:34:56.123456+00:00", "2026-02-28 UTC", "Feb 28, 2026, 12:34 UTC"],
  ])("formats an explicit-offset instant %s in UTC", (value, day, stamp) => {
    expect(formatMlInstantDate(value)).toBe(day);
    expect(formatMlTimestamp(value)).toBe(stamp);
  });
  it.each([null, undefined, "", "invalid", "2026-02-28", "2026-02-28T10:00:00", "2026-02-30T00:00:00Z", "2026-02-28T24:00:00Z", "2026-02-28T10:60:00Z", "2026-02-28T10:00:60Z", "2026-02-28T10:00:00+25:00"])("does not invent an instant for %s", value => {
    expect(formatMlTimestamp(value)).toBe("Unknown");
    expect(formatMlInstantDate(value)).toBe("Unknown");
  });
  it.each([true, false, null, undefined, "", "false", "true", 0, 1, {}, []])("renders boolean %s without truthiness coercion", value => {
    expect(formatMlBoolean(value)).toBe(value === true ? "true" : value === false ? "false" : "Unknown");
  });
  it("keeps run dates separate from instants on both ML surfaces and retains stale qualification", () => {
    const { pipeline, dashboard } = render({
      champion: { present: true, run_date: "2026-02-28", calibration: "none" },
      remediation: { run_date: "2026-02-28" },
      timestamps: { last_predict: "2026-02-28T00:30:00+14:00" },
      updated_at: "2026-02-28T00:30:00+14:00",
      freshness: { stale: true, status: "stale", expected_session: "2026-02-28" },
      pipeline_stages: [{ label: "Predict", tone: "success", status: "success", freshness_status: "stale" }],
    });
    for (const html of [pipeline, dashboard]) {
      expect(html).toContain("2026-02-28");
      expect(html).toContain("2026-02-27 UTC");
      expect(html).toContain("Stale");
      expect(html).not.toContain("Feb 27 · feature");
    }
    expect(pipeline).toContain("Expected session 2026-02-28");
    expect(dashboard).toContain("Feb 27, 2026, 10:30 UTC");
  });
  it.each([true, false, null, undefined, "false", 0])("renders booleans in actual pipeline rows for %s", raw => {
    const value = raw as boolean | null | undefined;
    const text = formatMlBoolean(raw);
    const { pipeline } = render({
      champion: { execution: { require_model_score: value } },
      remediation: { executed: value, repredict_executed: value, last_kind: "recalibrate" },
      freshness: { pred_compatible: value },
    });
    expect(pipeline).toContain(`require_model_score ${text}`);
    expect(pipeline).toContain(`· require ${text}`);
    expect(pipeline).toContain(`recalibrate / ${text}`);
    expect(pipeline).toMatch(new RegExp(`Repredict after remediation[\\s\\S]*?>${text}</div>`));
    expect(pipeline).toContain(`Compatibility ${text}`);
  });
  it.each([null, {}])("preserves unknown with missing payload %s", data => {
    const { pipeline, dashboard } = render(data);
    expect(pipeline).toContain("require_model_score Unknown");
    expect(pipeline).not.toContain("require_model_score false");
    expect(pipeline).toContain("Unknown / Unknown");
    expect(dashboard).toContain("Guard decision unknown");
    expect(dashboard).not.toContain("No remediation requested");
  });
  it("preserves a known allow guard while remediation is recommended", () => {
    const { dashboard } = render({ monitor: { guard_decision: "allow", recommended_action: "retrain" } });
    expect(dashboard).toContain("Guard allow");
    expect(dashboard).not.toContain("Guard decision unknown");
  });
  it("preserves explicit no-remediation evidence", () => {
    expect(render({ monitor: { recommended_action: "none" } }).dashboard).toContain("No remediation requested");
  });
});
