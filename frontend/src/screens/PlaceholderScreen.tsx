import { useMemo } from "react";
import NavbarDesktop from "../components/navbar/NavbarDesktop";

type PlaceholderScreenProps = {
  title: string;
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

export default function PlaceholderScreen({ title, activeTab, onTabSelect }: PlaceholderScreenProps) {
  const currentTab = activeTab ?? title;
  const navTabs = useMemo(
    () => navLabels.map((label) => ({ label, isActive: label === currentTab })),
    [currentTab]
  );
  const rightBadges = [
    { label: "Paper Trading", tone: "warning" as const, showDot: false },
    { label: "Live", tone: "success" as const, showDot: true },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_#f1f5f9,_#f8fafc_55%,_#ffffff_100%)] font-['Manrope'] text-slate-900">
      <NavbarDesktop tabs={navTabs} rightBadges={rightBadges} onTabSelect={onTabSelect} />

      <main className="relative pt-24 pb-12">
        <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-gradient-to-br from-sky-100 via-white to-amber-100 opacity-70 blur-3xl" />
        <div className="pointer-events-none absolute left-0 top-40 h-72 w-72 rounded-full bg-gradient-to-br from-emerald-100 via-white to-slate-100 opacity-60 blur-3xl" />

        <div className="relative mx-auto max-w-[1240px] px-8">
          <header className="max-w-xl">
            <h1 className="text-2xl font-semibold text-slate-900">{title}</h1>
            <p className="mt-2 text-sm text-slate-500">
              This view is staged for the next UI build-out.
            </p>
          </header>

          <section className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="text-sm text-slate-600">
              The {title} screen is not available yet. Navigation is wired and ready for the next
              implementation pass.
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
