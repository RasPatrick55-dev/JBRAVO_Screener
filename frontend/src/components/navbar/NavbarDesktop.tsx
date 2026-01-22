import type { NavbarDesktopProps } from "../../types/ui";
import StatusBadge from "../badges/StatusBadge";

export default function NavbarDesktop({ tabs, rightBadges, onTabSelect }: NavbarDesktopProps) {
  return (
    <div className="fixed inset-x-0 top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-[1240px] items-center justify-between px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">
            JB
          </div>
          <div className="text-sm font-semibold tracking-wide text-slate-900">JBRAVO Trading</div>
        </div>
        <nav aria-label="Primary" className="flex items-center gap-2 text-sm font-medium text-slate-500">
          {tabs.map((tab) => (
            <button
              key={tab.label}
              type="button"
              aria-current={tab.isActive ? "page" : undefined}
              onClick={() => onTabSelect?.(tab.label)}
              className={
                "rounded-full px-3 py-1 transition-colors " +
                (tab.isActive
                  ? "bg-blue-50 text-blue-700"
                  : "text-slate-500 hover:text-slate-800")
              }
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          {rightBadges.map((badge) => (
            <StatusBadge
              key={badge.label}
              label={badge.label}
              tone={badge.tone}
              showDot={badge.showDot}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
