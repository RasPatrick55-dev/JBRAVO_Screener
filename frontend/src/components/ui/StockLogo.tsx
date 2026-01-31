import { useEffect, useState } from "react";

interface StockLogoProps {
  symbol: string;
  logoUrl?: string;
}

const logoToken =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.REACT_APP_LOGO_DEV_API_KEY ??
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_LOGO_DEV_API_KEY ??
  "";

/**
 * Set REACT_APP_LOGO_DEV_API_KEY (and VITE_LOGO_DEV_API_KEY for Vite builds)
 * in the repo .env to enable Logo.dev stock ticker logos.
 */
export default function StockLogo({ symbol, logoUrl }: StockLogoProps) {
  const trimmed = symbol.trim().toUpperCase();
  const resolvedLogoUrl = logoUrl?.trim() || "";
  const useLogoUrl = resolvedLogoUrl.toLowerCase().includes("logo.dev");
  const logoSrc = useLogoUrl
    ? resolvedLogoUrl
    : logoToken
      ? `https://img.logo.dev/ticker/${trimmed}?token=${logoToken}&size=64&retina=true`
      : "/images/placeholder-stock-logo.png";
  const [status, setStatus] = useState<"loading" | "loaded" | "fallback">(
    useLogoUrl || logoToken ? "loading" : "fallback"
  );

  useEffect(() => {
    setStatus(useLogoUrl || logoToken ? "loading" : "fallback");
  }, [useLogoUrl, trimmed]);

  return (
    <div className="relative h-8 w-8 flex-shrink-0">
      <img
        src={logoSrc}
        alt={`${trimmed} logo`}
        className="h-8 w-8 rounded-md border jbravo-border-success bg-slate-950/80 object-contain p-1 shadow-[0_0_10px_rgba(34,197,94,0.25)]"
        onLoad={(event) => {
          if (event.currentTarget.naturalWidth < 2 || event.currentTarget.naturalHeight < 2) {
            event.currentTarget.src = "/images/placeholder-stock-logo.png";
            setStatus("fallback");
            return;
          }
          setStatus("loaded");
        }}
        onError={(event) => {
          if (!useLogoUrl && logoToken) {
            event.currentTarget.src = `https://img.logo.dev/ticker/${trimmed}?token=${logoToken}&size=64&retina=true`;
            return;
          }
          event.currentTarget.src = "/images/placeholder-stock-logo.png";
          setStatus("fallback");
        }}
        loading="lazy"
      />
      <span
        className={
          "absolute -right-1 -top-1 h-1.5 w-1.5 rounded-full " +
          (status === "loaded"
            ? "jbravo-status-dot-success"
            : status === "fallback"
              ? "bg-amber-300"
              : "bg-cyan-300")
        }
        title={status === "loaded" ? "Logo loaded" : status === "fallback" ? "Fallback" : "Loading"}
      />
    </div>
  );
}
