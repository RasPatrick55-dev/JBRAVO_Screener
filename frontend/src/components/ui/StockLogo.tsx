import { useEffect, useState } from "react";
import verifiedLogoAssets from "./stockLogoAssets.json";

interface StockLogoProps {
  symbol: string;
  logoUrl?: string;
}

/** CDN keys are publishable pk_ values; secret sk_ API keys must never enter an image URL. */
export function logoPublishableToken(...values: (string | undefined)[]): string {
  return values.map(value => value?.trim() ?? "").find(value => /^pk_[A-Za-z0-9_-]+$/.test(value)) ?? "";
}

const logoToken = logoPublishableToken(
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_LOGO_DEV_API_KEY,
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.REACT_APP_LOGO_DEV_API_KEY
);

export const LOGO_LOAD_TIMEOUT_MS = 8000;
export type LogoStatus = "loading" | "loaded" | "fallback";

export function normalizeLogoSymbol(symbol: string): string {
  return typeof symbol === "string" ? symbol.trim().toUpperCase() : "";
}

type StoredLogo = {
  ticker: string; provider: string; src: string; sha256: string;
  providerEtag: string; resolution: { type: string; value: string };
};

/** Source-pinned API exports only; absent tickers do not inherit another company's logo. */
export function stockLogoAssetSource(symbol: string, assets: Record<string, StoredLogo> = verifiedLogoAssets): string | null {
  const ticker = normalizeLogoSymbol(symbol);
  if (!Object.prototype.hasOwnProperty.call(assets, ticker)) return null;
  const asset = assets[ticker];
  if (
    !asset || asset.provider !== "Logo.dev" || asset.ticker !== ticker ||
    asset.resolution?.type !== "ticker" || asset.resolution?.value !== ticker ||
    !/^[a-f0-9]{64}$/.test(asset.sha256) || !/^[a-f0-9]{64}$/.test(asset.providerEtag) ||
    asset.src !== `/assets/company-logos/${ticker}-${asset.sha256}.png`
  ) return null;
  return asset.src;
}

/** Only Logo.dev ticker lookups for this exact symbol; never guess company domains. */
export function stockLogoSource(symbol: string, logoUrl = "", publishableToken = logoToken): string | null {
  const ticker = normalizeLogoSymbol(symbol);
  if (!/^[A-Z0-9]+(?:[.-][A-Z0-9]+)*$/.test(ticker) || ticker.length > 20 || ticker === "UNKNOWN") return null;
  const token = logoPublishableToken(publishableToken);
  try {
    const supplied = new URL(logoUrl.trim());
    if (
      supplied.protocol === "https:" &&
      supplied.hostname === "img.logo.dev" &&
      !supplied.username && !supplied.password && !supplied.port &&
      decodeURIComponent(supplied.pathname).replace(/^\/ticker\//, "").toUpperCase() === ticker &&
      supplied.pathname.startsWith("/ticker/")
    ) {
      const suppliedToken = logoPublishableToken(supplied.searchParams.get("token") ?? undefined);
      const selectedToken = suppliedToken || token;
      if (!selectedToken) return null;
      // Only public CDN token and supported image parameters cross the browser boundary.
      const target = new URL(`https://img.logo.dev/ticker/${encodeURIComponent(ticker)}`);
      target.searchParams.set("token", selectedToken);
      target.searchParams.set("fallback", "404");
      const size = supplied.searchParams.get("size");
      if (size && /^\d+$/.test(size) && Number(size) >= 1 && Number(size) <= 800) target.searchParams.set("size", size);
      const retina = supplied.searchParams.get("retina");
      if (retina === "true" || retina === "false") target.searchParams.set("retina", retina);
      return target.toString();
    }
  } catch {
    // Missing/malformed URLs do not change provider or company identity.
  }
  if (!token) return null;
  const target = new URL(`https://img.logo.dev/ticker/${encodeURIComponent(ticker)}`);
  target.searchParams.set("token", token);
  target.searchParams.set("size", "64");
  target.searchParams.set("retina", "true");
  target.searchParams.set("fallback", "404");
  return target.toString();
}

/** A new symbol or source mounts a new image lifecycle, isolating late old events. */
export function StockLogoImage({ symbol, src }: { symbol: string; src: string | null }) {
  const [status, setStatus] = useState<LogoStatus>(src ? "loading" : "fallback");
  useEffect(() => {
    if (!src || status !== "loading") return;
    const timeout = setTimeout(() => setStatus(current => current === "loading" ? "fallback" : current), LOGO_LOAD_TIMEOUT_MS);
    return () => clearTimeout(timeout);
  }, [src, status]);

  const ticker = symbol || "?";
  const fallbackText = ticker.length > 5 ? `${ticker.slice(0, 4)}…` : ticker;
  const description = status === "loaded"
    ? `${ticker} company logo`
    : `${ticker} ticker; logo ${status === "loading" ? "loading" : "unavailable"}`;

  return (
    <span className="relative h-8 w-8 flex-shrink-0" role="img" aria-label={description}>
      <span
        aria-hidden="true"
        className="flex h-8 w-8 items-center justify-center rounded-md border jbravo-border-success bg-slate-950/80 p-1 font-arimo text-[10px] font-semibold text-slate-100 shadow-[0_0_10px_rgba(34,197,94,0.25)]"
        title={ticker}
        style={{ fontSize: ticker.length > 4 ? 8 : 10 }}
      >
        {status === "loaded" ? null : fallbackText}
      </span>
      {src && status !== "fallback" ? (
        <img
          src={src}
          alt=""
          aria-hidden="true"
          className={"absolute inset-0 h-8 w-8 rounded-md border jbravo-border-success bg-slate-50 object-contain p-1 " + (status === "loaded" ? "opacity-100" : "opacity-0")}
          onLoad={(event) => {
            const valid = event.currentTarget.naturalWidth >= 2 && event.currentTarget.naturalHeight >= 2;
            setStatus(current => current === "loading" ? (valid ? "loaded" : "fallback") : current);
          }}
          onError={() => setStatus("fallback")}
          referrerPolicy="origin"
          loading="eager"
        />
      ) : null}
      <span
        aria-hidden="true"
        className={"absolute -right-1 -top-1 h-1.5 w-1.5 rounded-full " + (
          status === "loaded" ? "jbravo-status-dot-success" : status === "fallback" ? "bg-amber-300" : "bg-cyan-300"
        )}
        title={status === "loaded" ? "Logo loaded" : status === "fallback" ? "Logo unavailable" : "Logo loading"}
      />
    </span>
  );
}

/** Use an existing Logo.dev publishable key at UI build time; no server secret is needed. */
export default function StockLogo({ symbol, logoUrl }: StockLogoProps) {
  const ticker = normalizeLogoSymbol(symbol);
  const src = stockLogoAssetSource(ticker) ?? stockLogoSource(ticker, logoUrl);
  return <StockLogoImage key={`${ticker}|${src ?? ""}`} symbol={ticker} src={src} />;
}
