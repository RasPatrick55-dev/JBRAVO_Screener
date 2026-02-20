import { useMemo, useState } from "react";
import { buildNavbarBadges, type LiveDataSyncState, useLiveTradingStatus } from "../components/navbar/liveStatus";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import TradesTab from "../components/trades/TradesTab";

type TradesOverviewProps = {
  activeTab?: string;
  onTabSelect?: (label: string) => void;
};

const navLabels = [
  "Dashboard",
  "Account",
  "Trades",
  "Positions",
  "Execute",
  "Screener",
  "ML Pipeline",
];

export default function TradesOverview({ activeTab, onTabSelect }: TradesOverviewProps) {
  const currentTab = activeTab ?? "Trades";
  const liveTradingStatus = useLiveTradingStatus();
  const [pageSyncState, setPageSyncState] = useState<LiveDataSyncState>("loading");
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = useMemo(
    () => buildNavbarBadges(liveTradingStatus, pageSyncState),
    [liveTradingStatus, pageSyncState]
  );

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />
      <main className="relative pb-12 pt-[calc(var(--app-nav-height,208px)+16px)]">
        <div className="pointer-events-none absolute -top-28 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-cyan-300/20 via-sky-200/10 to-transparent blur-3xl dark:from-cyan-500/18 dark:via-blue-500/12 dark:to-transparent" />
        <div className="pointer-events-none absolute left-0 top-56 h-72 w-72 rounded-full bg-gradient-to-br from-violet-300/16 via-emerald-200/12 to-transparent blur-3xl dark:from-violet-500/12 dark:via-emerald-500/12 dark:to-transparent" />
        <div className="relative mx-auto w-full max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <TradesTab onSyncStateChange={setPageSyncState} />
        </div>
      </main>
    </div>
  );
}
