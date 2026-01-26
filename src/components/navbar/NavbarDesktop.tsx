import type { NavbarDesktopProps } from "../../types/ui";
import StatusBadge from "../badges/StatusBadge";
import ThemeToggle from "../ui/ThemeToggle";

export default function NavbarDesktop({ tabs, rightBadges, onTabSelect }: NavbarDesktopProps) {
  return (
    <div className="fixed inset-x-0 top-0 z-50 border-b border-slate-200/80 bg-white/80 backdrop-blur dark:border-slate-800/80 dark:bg-slate-950/80">
      <div className="mx-auto w-full max-w-[1240px] px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 py-3 lg:h-16 lg:flex-row lg:items-center lg:justify-between lg:py-0">
          <div className="flex items-center justify-between lg:justify-start">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">
                JB
              </div>
              <div className="text-sm font-semibold tracking-wide text-slate-900 dark:text-slate-100">
                JBRAVO Trading
              </div>
            </div>
            <div className="flex items-center gap-2 lg:hidden">
              <ThemeToggle />
            </div>
          </div>
          <nav
            aria-label="Primary"
            className="flex items-center gap-2 overflow-x-auto text-sm font-medium text-slate-500 dark:text-slate-400 lg:overflow-visible"
          >
            {tabs.map((tab) => (
              <button
                key={tab.label}
                type="button"
                aria-current={tab.isActive ? "page" : undefined}
                onClick={() => onTabSelect?.(tab.label)}
                className={
                  "whitespace-nowrap rounded-full px-3 py-1 transition-colors " +
                  (tab.isActive
                    ? "bg-blue-50 text-blue-700 dark:bg-cyan-500/15 dark:text-cyan-200"
                    : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100")
                }
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2 overflow-x-auto lg:overflow-visible">
            {rightBadges.map((badge) => (
              <StatusBadge
                key={badge.label}
                label={badge.label}
                tone={badge.tone}
                showDot={badge.showDot}
              />
            ))}
            <div className="hidden lg:block">
              <ThemeToggle />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
