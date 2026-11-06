import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { receiptStatus, type ProcessReceipt } from "./processReceipt";
import PipelineStatusCard from "./PipelineStatusCard";
import ExecuteTradesStatusCard from "./ExecuteTradesStatusCard";

const NOW = Date.parse("2026-11-06T20:00:00Z");
const receipt: ProcessReceipt = {source:"primary_receipt",ok:true,status:"COMPLETED",rc:0,finished_utc:"2026-11-06T03:47:00Z"};
describe("current process receipts", () => {
  it("shows today's real completed run rather than an older legacy task", () => {
    expect(receiptStatus(receipt,NOW)).toBe("COMPLETED");
    const html=renderToStaticMarkup(<PipelineStatusCard lastRun={{date:"Nov 6",start:"03:45",end:"03:47",duration:"2m"}} subprocess={{screener:"OK",backTester:"OK",metrics:"OK"}} isLive={false} status="COMPLETED"/>);
    expect(html).toContain("COMPLETED");expect(html).not.toContain("OVERDUE");expect(html).not.toContain("CYCLE");
  });
  it.each([{ok:false},{rc:1},{status:"PENDING"},{finished_utc:"2026-11-05T01:00:00Z"},{finished_utc:"2026-11-07T01:00:00Z"},{finished_utc:"2026-11-06T03:47:00"}])("never promotes unavailable, failed, pending, stale or future receipts", change => {
    expect(receiptStatus({...receipt,...change},NOW)).not.toBe("COMPLETED");
  });
  it("renders an execution failure explicitly instead of calling it overdue",()=>{
    const html=renderToStaticMarkup(<ExecuteTradesStatusCard lastRun={{date:"Nov 6",start:"09:00",end:"09:35",duration:"35m"}} ordersPlaced={0} totalValue={Number.NaN} successRate={Number.NaN} isCycleComplete={false} status="FAILED"/>);
    expect(html).toContain("FAILED");expect(html).not.toContain("OVERDUE");
  });
});
