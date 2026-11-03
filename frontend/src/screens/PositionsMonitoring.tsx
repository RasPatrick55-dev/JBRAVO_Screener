import { buildNavbarBadges, useLiveTradingStatus } from "../components/navbar/liveStatus";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import PositionsTab from "../components/positions/PositionsTab";
import SourceNotice from "../components/ui/SourceNotice";
import { canDisplay, finiteNumber, useSource } from "../components/ui/sourceState";
import type { PositionRowProps } from "../components/positions/PositionRow";

export function toPositionRow(position: Record<string, unknown>): PositionRowProps {
  return { symbol: String(position.symbol ?? "--"), logoUrl: String(position.logoUrl ?? ""),
    shares: finiteNumber(position.qty), entryPrice: finiteNumber(position.entryPrice),
    currentPrice: finiteNumber(position.currentPrice), openPL: finiteNumber(position.dollarPL),
    daysHeld: finiteNumber(position.daysHeld), trailingStop: finiteNumber(position.trailingStop),
    capturedPL: finiteNumber(position.capturedPL) };
}
export default function PositionsMonitoring({ activeTab, onTabSelect }: { activeTab?: string; onTabSelect?: (label: string) => void }) {
  const positions = useSource("/api/positions/monitoring");
  const logs = useSource("/api/positions/logs?limit=200");
  const live = useLiveTradingStatus();
  const available = canDisplay(positions.kind) && positions.payload?.ok === true && Array.isArray(positions.payload.positions);
  const rows = available ? (positions.payload?.positions as Record<string, unknown>[]).map(toPositionRow) : [];
  const rawSummary = positions.payload?.summary as Record<string, unknown> | undefined;
  const summary = { totalShares: finiteNumber(rawSummary?.totalShares), totalOpenPL: finiteNumber(rawSummary?.totalOpenPL),
    avgDaysHeld: finiteNumber(rawSummary?.avgDaysHeld), totalCapturedPL: finiteNumber(rawSummary?.totalCapturedPL) };
  const logRows = canDisplay(logs.kind) && Array.isArray(logs.payload?.logs) ? logs.payload.logs as Record<string, unknown>[] : [];
  const currentTab = activeTab ?? "Positions";
  const navTabs = ["Dashboard", "Account", "Trades", "Positions", "Execute", "Screener", "ML Pipeline"].map(label => ({ label, isActive: label === currentTab }));
  const badges = buildNavbarBadges(live, positions.kind === "loading" ? "loading" : available && positions.kind !== "stale" ? "ready" : "error");
  // Positions and logs are intentionally independent: a logs timeout cannot zero broker exposure.
  const monitoringLogs = logRows.map(row => ({ timestamp: String(row.timestamp ?? "Unknown"),
    type: (["warning", "success"].includes(String(row.type)) ? row.type : "info") as "warning" | "success" | "info", message: String(row.message ?? "") }));
  return <div className="dark min-h-screen bg-slate-950 text-slate-100">
    <NavbarDesktop tabs={navTabs} rightBadges={badges} onTabSelect={onTabSelect} />
    <main className="mx-auto max-w-7xl space-y-4 px-4 pb-12 pt-[calc(var(--app-nav-height,208px)+16px)]">
      <SourceNotice label="Broker positions" {...positions} />
      <p className="text-sm text-secondary">Alpaca broker marks only. PostgreSQL OPEN trades are not substituted. Optional trailing-stop, days-held and captured-P/L enrichment is deferred and remains unknown.</p>
      <SourceNotice label="Operational monitoring logs" {...logs} />
      <PositionsTab positions={rows} summary={summary} logs={monitoringLogs} isLoading={positions.kind === "loading"}
        hasError={!available && positions.kind !== "loading"} available={available} logsAvailable={canDisplay(logs.kind)} />
    </main></div>;
}
