export interface OrderProvenanceFields {
  raw_status?: string | null;
  status?: string | null;
  order_id?: string | null;
  replaces?: string | null;
  replaced_by?: string | null;
}

export default function OrderProvenance({ order, showStatus = true }: { order: OrderProvenanceFields; showStatus?: boolean }) {
  return <div className="mt-1 space-y-0.5 break-all text-[10px] font-normal normal-case text-secondary">
    {showStatus ? <div>Broker status: {order.raw_status || order.status || "unknown"}</div> : null}
    <div>Order: {order.order_id || "unknown"}</div>
    {order.replaces || order.replaced_by ? <div>
      <span>Replacement chain</span>
      {order.replaces ? <div>Replaces: {order.replaces}</div> : null}
      {order.replaced_by ? <div>Replaced by: {order.replaced_by}</div> : null}
    </div> : null}
  </div>;
}
