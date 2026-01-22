import type { ReactNode } from "react";

export type StatusTone = "success" | "warning" | "error" | "info" | "neutral";

export interface NavTab {
  label: string;
  isActive?: boolean;
}

export interface NavbarDesktopProps {
  tabs: NavTab[];
  rightBadges: Array<{
    label: string;
    tone: StatusTone;
    showDot?: boolean;
  }>;
  onTabSelect?: (label: string) => void;
}

export interface KPICardProps {
  title: string;
  value: string;
  footnote?: string;
  detail?: string;
  detailTone?: StatusTone;
  icon: ReactNode;
}

export interface StatusBadgeProps {
  label: string;
  tone: StatusTone;
  showDot?: boolean;
  size?: "sm" | "md";
  className?: string;
}

export interface SystemStatusItem {
  title: string;
  status: string;
  tone: StatusTone;
  description: string;
  meta: string;
}

export interface LogEntry {
  time: string;
  level: "INFO" | "WARN" | "ERROR" | "SUCCESS";
  message: string;
}

export interface LogViewerProps {
  title: string;
  entries: LogEntry[];
  statusLabel?: string;
  actionLabel?: string;
}
