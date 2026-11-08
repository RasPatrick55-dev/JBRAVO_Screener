import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { filterOrdersClient, filterTrailingClient } from "../src/components/execute/utils";

// Read only this pure backend function's declared sets so adding a supported
// lifecycle state there also exercises the client filters without live APIs.
const backend = readFileSync(new URL("../../dashboards/dashboard_app.py", import.meta.url), "utf8");
const start = backend.indexOf("def _execute_status_scope_matches(");
const functionSource = backend.slice(start, backend.indexOf("def _execute_lsx_matches(", start));
const sets = [...functionSource.matchAll(/return normalized in\s*\{([^}]+)\}/g)]
  .map(match => [...match[1].matchAll(/"([A-Z_]+)"/g)].map(status => status[1]));
const [openStatuses = [], closedStatuses = []] = sets;

describe("Execute status scope parity", () => {
  it("loads the complete backend open/closed status contract", () => {
    expect(openStatuses).toEqual([
      "PENDING", "PARTIAL", "ACTIVE", "PENDING_REPLACE", "PENDING_CANCEL",
      "DONE_FOR_DAY", "HELD", "STOPPED", "SUSPENDED", "CALCULATED",
    ]);
    expect(closedStatuses).toEqual(["FILLED", "REJECTED", "CANCELED", "EXPIRED", "REPLACED", "TRIGGERED"]);
  });

  it.each(openStatuses)("retains backend open state %s in orders and trailing stops", status => {
    for (const filter of [filterOrdersClient, filterTrailingClient]) {
      // Covers both raw broker status and the normalized legacy response shape.
      const rows = [{ raw_status: status.toLowerCase(), status: "PENDING" }, { status }];
      expect(filter(rows, "", "open", "all")).toEqual(rows);
      expect(filter(rows, "", "all", "all")).toEqual(rows);
      expect(filter(rows, "", "closed", "all")).toEqual([]);
    }
  });

  it.each(closedStatuses)("keeps backend closed state %s out of Open in both tables", status => {
    for (const filter of [filterOrdersClient, filterTrailingClient]) {
      const rows = [{ raw_status: status.toLowerCase(), status: "PENDING" }, { status }];
      expect(filter(rows, "", "closed", "all")).toEqual(rows);
      expect(filter(rows, "", "open", "all")).toEqual([]);
    }
  });

  it.each(["new", "accepted", "pending_new", "accepted_for_bidding", "open", "partially_filled"])("retains raw broker alias %s before backend bucketing", raw_status => {
    for (const filter of [filterOrdersClient, filterTrailingClient]) {
      const rows = [{ raw_status, status: "PENDING" }];
      expect(filter(rows, "", "open", "all")).toEqual(rows);
      expect(filter(rows, "", "closed", "all")).toEqual([]);
    }
  });

  it("does not guess an unknown status into either lifecycle scope", () => {
    for (const filter of [filterOrdersClient, filterTrailingClient]) {
      const rows = [{ raw_status: "unrecognized", status: "PENDING" }];
      expect(filter(rows, "", "all", "all")).toEqual(rows);
      expect(filter(rows, "", "open", "all")).toEqual([]);
      expect(filter(rows, "", "closed", "all")).toEqual([]);
    }
  });
});
