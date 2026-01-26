import type { CSSProperties, ReactNode } from "react";

interface CardShellProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}

const baseClasses =
  "flex h-full flex-col rounded-2xl border border-black/5 p-6 shadow-[0_4px_6px_-4px_rgba(0,0,0,0.10)] dark:border-white/10";

export default function CardShell({ children, className = "", style }: CardShellProps) {
  return (
    <div className={`${baseClasses}${className ? " " + className : ""}`} style={style}>
      {children}
    </div>
  );
}
