import { useMemo } from "react";
import ExecuteTab from "../components/execute/ExecuteTab";
import NavbarDesktop from "../components/navbar/NavbarDesktop";

type ExecuteTradesProps = {
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

export default function ExecuteTrades({ activeTab, onTabSelect }: ExecuteTradesProps) {
  const currentTab = activeTab ?? "Execute";
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "neutral" as const },
  ];

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />
      <main className="relative pb-12 pt-[calc(var(--app-nav-height,208px)+16px)]">
        <div className="pointer-events-none absolute -top-28 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-cyan-300/16 via-sky-200/8 to-transparent blur-3xl dark:from-cyan-500/15 dark:via-blue-500/12 dark:to-transparent" />
        <div className="pointer-events-none absolute left-0 top-64 h-72 w-72 rounded-full bg-gradient-to-br from-blue-300/14 via-indigo-200/10 to-transparent blur-3xl dark:from-indigo-500/12 dark:via-sky-500/10 dark:to-transparent" />
        <div className="relative mx-auto w-full max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <ExecuteTab />
        </div>
      </main>
    </div>
  );
}
