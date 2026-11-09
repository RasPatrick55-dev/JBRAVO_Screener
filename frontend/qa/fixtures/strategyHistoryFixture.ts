import wireFixture from "./strategyHistoryWire.synthetic.json";

// Entirely fictional contract examples, not broker or production evidence.
export const completeStrategySignal = structuredClone(wireFixture.rows[0]);
export function strategyHistoryFixture() {
  return { ...structuredClone(wireFixture), observed_at: "2026-11-09T16:00:00Z" };
}
