export interface ComputationContract {
  ok?: boolean;
  status?: string | null;
  completion_status?: string | null;
  computation_complete?: boolean;
  completion_scope?: string | null;
  research_qualified?: boolean;
  order_submission_authorized?: boolean;
  entry_readiness_scope?: string | null;
  broker_preflight_required?: boolean;
}

export function computationComplete(payload: ComputationContract | null) {
  return payload?.ok === true && payload.completion_status === "verified"
    && payload.computation_complete === true && ["COMPLETE", "EMPTY"].includes(payload.status ?? "");
}

export function computationStatus(payload: ComputationContract | null): string {
  if (payload?.ok !== true) return "UNAVAILABLE";
  if (payload.status === "ERROR") return "ERROR";
  return computationComplete(payload) ? "COMPUTE COMPLETE" : "UNVERIFIED";
}

export function ComputationScope({ payload }: { payload: ComputationContract | null }) {
  return <p className="mt-2 text-[11px] text-secondary">
    Computation: {computationComplete(payload) ? "verified complete" : "unverified"}.
    {" "}Research qualification: {payload?.research_qualified === true ? "qualified" : payload?.research_qualified === false ? "not qualified" : "unknown"}.
    {" "}Order submission: {payload?.order_submission_authorized === true ? "authorized by source" : payload?.order_submission_authorized === false ? "not authorized" : "unknown"}.
    {payload?.entry_readiness_scope ? ` Scope: ${payload.entry_readiness_scope}.` : ""}
    {payload?.broker_preflight_required ? " Broker preflight required." : ""}
  </p>;
}
