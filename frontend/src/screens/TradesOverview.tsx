import { useMemo } from "react";
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
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );

  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: true },
    { label: "Live", tone: "success" as const, showDot: true },
  ];

  return (
    <div className="dark min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900 dark:bg-[radial-gradient(circle_at_top,_#0B1220,_#0F172A_55%,_#020617_100%)] dark:text-slate-100">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />
      <main className="pt-36 pb-12 sm:pt-32">
        <div className="mx-auto w-full max-w-[1240px] px-4 sm:px-6 lg:px-8">
          <TradesTab />
        </div>
      </main>
    </div>
  );
}
