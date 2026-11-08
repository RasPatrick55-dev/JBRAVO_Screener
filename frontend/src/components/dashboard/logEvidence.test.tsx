import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { eventFreshness, LogSourceReceipt } from "./logEvidence";
import LatestLogsTable from "./LatestLogsTable";
import { buildLogEntries } from "../../screens/DashboardHealth";

const now = Date.parse("2026-11-08T14:00:00Z");
describe("log provenance and freshness", () => {
  it("separates event age from fetch age and keeps unknown age honest", () => {
    expect(eventFreshness("2026-10-25T12:00:00Z", now)).toContain("Historical / stale");
    expect(eventFreshness("2026-11-08T13:00:00Z", now)).toContain("Recent");
    for (const stamp of [null, "invalid", "2099-01-01T00:00:00Z"]) expect(eventFreshness(stamp, now)).toBe("Age unknown");
    const html = renderToStaticMarkup(<LogSourceReceipt receipt={{source:"screener.log", observed_at: "2026-11-08T14:00:00Z", latest_event_at: "2026-10-25T12:00:00Z"}} />);
    expect(html).toContain("Historical / stale");
    expect(html).toContain("Fetched:");
    expect(html).toContain("Latest event:");
  });
  it("honors explicit missing metadata and canonical offset times", () => {
    const invalid = "2026-02-30 12:00:00 invalid date";
    const offset = "2026-11-08T08:00:00-06:00 INFO offset timestamp";
    const rows = buildLogEntries([{source:"pipeline", text:`${invalid}\n${offset}`, entries:[{raw: invalid, ts_utc:null}, {raw:offset, ts_utc:"2026-11-08T14:00:00+00:00"}]}]);
    expect(rows[0].eventTimestamp).toBe("2026-11-08T14:00:00+00:00");
    expect(rows[1].eventTimestamp).toBeNull();
    expect(rows[1].time).toBe("--");
  });
  it("retains untimed, old, unknown and mixed provenance without inventing broker activity", () => {
    const raw = "[2026-11-08 03:47:47,034] [INFO]: Opened PTCT @ 69.68 (63 shares)";
    const rows = buildLogEntries([{source:"pipeline", text: `${raw}\nunknown untimed\n2020-01-01 00:00:00 -- Started task`, entries:[{raw, ts_utc:"2026-11-08T03:47:47Z", provenance_label:"Simulation-format match · runtime unverified"}]}]);
    expect(rows).toHaveLength(3);
    expect(rows[0].provenanceLabel).toContain("runtime unverified");
    expect(rows[2].provenanceLabel).toContain("Unclassified");
    expect(rows[2].eventTimestamp).toBeNull();
    const html = renderToStaticMarkup(<LatestLogsTable rows={rows.map(row => ({dateTime:row.time, script:row.source || "", text:row.message, provenanceLabel:row.provenanceLabel, eventTimestamp:row.eventTimestamp}))} />);
    expect(html).toContain("not broker fills");
    expect(html).toContain("Age unknown");
    expect(html).toContain("Historical / stale");
    expect(html).toContain("runtime unverified");
  });
});
