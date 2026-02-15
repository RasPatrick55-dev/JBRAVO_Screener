import { useMemo } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";
import ScreenerTab from "../components/screener/ScreenerTab";

type ScreenerOverviewProps = {
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

export default function ScreenerOverview({ activeTab, onTabSelect }: ScreenerOverviewProps) {
  const currentTab = activeTab ?? "Screener";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  return (
    <div className="dark min-h-screen bg-slate-50 font-['Manrope'] text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />
      <main className="relative pb-12 pt-[calc(var(--app-nav-height,208px)+16px)]">
        <div className="pointer-events-none absolute -top-28 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-cyan-300/20 via-sky-200/10 to-transparent blur-3xl dark:from-cyan-500/18 dark:via-blue-500/12 dark:to-transparent" />
        <div className="pointer-events-none absolute left-0 top-56 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-300/16 via-amber-200/10 to-transparent blur-3xl dark:from-emerald-500/14 dark:via-amber-500/10 dark:to-transparent" />
        <div className="relative mx-auto w-full max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <ScreenerTab />
        </div>
      </main>
    </div>
  );
}
