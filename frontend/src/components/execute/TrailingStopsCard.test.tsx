import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import TrailingStopsCard from "./TrailingStopsCard";
import { filterTrailingClient } from "./utils";

describe("trailing observation provenance", () => {
  it("uses reconciled status, qualifies history and retains replacement identity", () => {
    const row = { status: "CANCELED", raw_status: "OrderStatus.NEW", order_id: "CRBG-old", replaced_by: "CRBG-new", observation_source: "postgres_history" as const };
    const html = renderToStaticMarkup(<TrailingStopsCard rows={[row]} isLoading={false} hasError={false} />);
    expect(html).toContain("CANCELED");
    expect(html).toContain("Historical / unverified");
    expect(html).toContain("current protection is unverified");
    expect(html).toContain("CRBG-new");
    expect(html).not.toContain("Broker status:");
    expect(filterTrailingClient([row], "", "closed", "all")).toHaveLength(1);
    expect(filterTrailingClient([row], "", "open", "all")).toHaveLength(0);
  });
  it("qualifies unavailable, empty and stale retained broker rows", () => {
    const html = renderToStaticMarkup(<TrailingStopsCard rows={[{observation_source:"broker_response", status:"PENDING_REPLACE"}]} isLoading={false} hasError={true} />);
    expect(html).toContain("Historical / unverified");
    expect(html).toContain("last known observations");
    expect(html).not.toContain(">Broker response<");
    const empty = renderToStaticMarkup(<TrailingStopsCard rows={[]} isLoading={false} hasError={false} />);
    expect(empty).toContain("No trailing-stop observations in this response");
    expect(empty).toContain("not a complete protection inventory");
  });
});
