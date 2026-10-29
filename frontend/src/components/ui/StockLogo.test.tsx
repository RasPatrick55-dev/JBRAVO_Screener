import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import assets from "./stockLogoAssets.json";
import LogoAttribution from "./LogoAttribution";
import TradesLeaderboard from "../trades/TradesLeaderboard";

describe("source-bound OAuth API logo assets", () => {
  it.each(Object.keys(assets))("uses the matching local asset for %s without credentials", ticker => {
    expect(StockLogo({symbol:ticker}).props.src).toBe(assets[ticker as keyof typeof assets].src);
  });
  it("contains nonsquare marks on a light backing without stretching or layout changes", () => {
    for (const ticker of ["RTX","SGI"]) {
      const node=frame(ticker,stockLogoAssetSource(ticker));
      expect(image(node)!.props.className).toContain("object-contain");
      expect(image(node)!.props.className).toContain("bg-slate-50");
      expect(image(node)!.props.className).toContain("h-8 w-8");
    }
  });
  it("uses the exact STRL export without credentials or an invented company/domain", () => {
    const local=stockLogoAssetSource(" strl ");
    expect(local).toBe(assets.STRL.src);
    expect(StockLogo({symbol:"STRL"}).props.src).toBe(local);
    expect(assets.STRL.resolution).toEqual({type:"ticker",value:"STRL",method:"security"});
    expect(assets.STRL).not.toHaveProperty("name");
    expect(assets.STRL).not.toHaveProperty("domain");
    expect(JSON.stringify(assets)).not.toMatch(/token=|https:|SEALED_TOKEN|sk_/);
  });
  it.each(["SYN","SYNX","STRL.L","STRL-OTHER","constructor","__proto__",""])("does not reuse STRL for unmapped ticker %s", symbol => {
    expect(stockLogoAssetSource(symbol)).toBeNull();
  });
  it.each(["ticker","provider","resolution","sha256","providerEtag","src"])("rejects a changed %s binding instead of requesting an unverified image", field => {
    const changed=structuredClone(assets) as unknown as Record<string, any>;
    changed.STRL[field]=field==="resolution"?{type:"ticker",value:"OTHER"}:"invalid";
    expect(stockLogoAssetSource("STRL",changed)).toBeNull();
  });
  it("rejects a missing/null map row safely and keeps unrelated CDN configuration support", () => {
    expect(stockLogoAssetSource("STRL",{STRL:null} as any)).toBeNull();
    expect(stockLogoAssetSource("STRL",{})).toBeNull();
    expect(stockLogoSource("ETSY","","pk_fictional_test")).toContain("/ticker/ETSY?");
  });
  it("retains ticker fallback on a missing/failed local asset, never another company's image", () => {
    const node=frame("STRL",assets.STRL.src);
    image(node)!.props.onError();
    expect(html("STRL",assets.STRL.src)).toContain("STRL ticker; logo unavailable");
    expect(image(frame("STRL",assets.STRL.src))).toBeUndefined();
  });
  it("resets correctly between a mapped and an unmapped symbol while identical renders stay stable", () => {
    const local=StockLogo({symbol:"STRL"}),same=StockLogo({symbol:" strl "}),missing=StockLogo({symbol:"SYN"});
    expect(local.key).toBe(same.key);expect(local.key).not.toBe(missing.key);
    expect(missing.props.src).toBeNull();expect(missing.props.symbol).toBe("SYN");
  });
});

describe("visible Logo.dev attribution", () => {
  it("uses a plain visible public link without nofollow or hidden styling", () => {
    const markup=renderToStaticMarkup(<LogoAttribution/>);
    expect(markup).toContain('href="https://logo.dev"');
    expect(markup).toContain("Logos provided by Logo.dev");
    expect(markup).not.toMatch(/nofollow|sr-only|aria-hidden|display:none/);
  });
  it.each(["ready","empty","loading"])("shows one attribution link below the actual leaderboard for %s", state => {
    const markup=renderToStaticMarkup(<TradesLeaderboard
      rows={state==="ready"?[{rank:1,symbol:"STRL",pl:1}]:[]}
      selectedRange="all" onRangeChange={()=>{}} mode="winners" onModeChange={()=>{}}
      isLoading={state==="loading"}/>);
    expect(markup.match(/href="https:\/\/logo.dev"/g)).toHaveLength(1);
    expect(markup.indexOf('href="https://logo.dev"')).toBeGreaterThan(markup.indexOf("</table>"));
    if(state==="ready")expect(markup).toContain(assets.STRL.src);
  });
});

const hooks = vi.hoisted(() => ({
  status: undefined as string | undefined,
  cleanup: undefined as (() => void) | undefined,
}));
vi.mock("react", async importOriginal => {
  const actual = await importOriginal<typeof import("react")>();
  return {
    ...actual,
    useState: (initial: string) => [
      hooks.status ?? initial,
      (next: string | ((current: string) => string)) => {
        hooks.status = typeof next === "function" ? next(hooks.status ?? initial) : next;
      },
    ],
    useEffect: (effect: () => void | (() => void)) => {
      hooks.cleanup?.();
      hooks.cleanup = effect() || undefined;
    },
  };
});
import StockLogo, { StockLogoImage, stockLogoSource, stockLogoAssetSource, normalizeLogoSymbol, logoPublishableToken, LOGO_LOAD_TIMEOUT_MS } from "./StockLogo";

const src = "https://img.logo.dev/ticker/SYN?token=pk_fictional_test";
type Node = { type: unknown; props: Record<string, any>; key?: string | null };
const frame = (symbol = "SYN", url: string | null = src) => StockLogoImage({ symbol, src: url }) as unknown as Node;
const image = (node: Node): Node | undefined => (node.props.children as Node[]).find(child => child?.type === "img");
const html = (symbol = "SYN", url: string | null = src) => renderToStaticMarkup(<StockLogoImage symbol={symbol} src={url}/>);
beforeEach(() => { vi.useFakeTimers(); hooks.status = undefined; hooks.cleanup = undefined; });
afterEach(() => { hooks.cleanup?.(); vi.useRealTimers(); });

describe("Logo.dev source identity", () => {
  it("keeps exact ticker lookup and encoded publishable token without guessed domains", () => {
    const url = new URL(stockLogoSource(" brk.b ", "", "pk_fictional_test")!);
    expect(url.origin).toBe("https://img.logo.dev");
    expect(url.pathname).toBe("/ticker/BRK.B");
    expect(url.searchParams.get("token")).toBe("pk_fictional_test");
    expect(url.searchParams.get("size")).toBe("64");
    expect(url.searchParams.get("retina")).toBe("true");
    expect(url.searchParams.get("fallback")).toBe("404");
  });
  it("preserves a supplied matching Logo.dev URL and can fill a missing token from existing config", () => {
    const supplied=new URL(stockLogoSource("SYN", src, "")!);
    expect(supplied.pathname).toBe("/ticker/SYN");
    expect(supplied.searchParams.get("token")).toBe("pk_fictional_test");
    expect(supplied.searchParams.get("fallback")).toBe("404");
    expect(new URL(stockLogoSource("SYN", "https://img.logo.dev/ticker/SYN?size=64", "pk_fictional")!).searchParams.get("token")).toBe("pk_fictional");
  });
  it("accepts only publishable configuration and does not let blank legacy config mask a usable key", () => {
    expect(logoPublishableToken("", " pk_fictional_test ")).toBe("pk_fictional_test");
    expect(logoPublishableToken("sk_fictional_secret", "pk_fictional_test")).toBe("pk_fictional_test");
    expect(stockLogoSource("SYN", "", "sk_fictional_secret")).toBeNull();
    expect(stockLogoSource("SYN", "https://img.logo.dev/ticker/SYN?token=sk_fictional_secret", "")).toBeNull();
    expect(stockLogoSource("SYN", "https://img.logo.dev/ticker/SYN", "")).toBeNull();
  });
  it("normalizes a matching ticker and drops unrelated URL parameters and fragments", () => {
    const url=new URL(stockLogoSource("SYN",src.replace("/SYN?","/syn?")+"&password=synthetic-private&size=128#debug", "")!);
    expect(url.pathname).toBe("/ticker/SYN");expect(url.searchParams.has("password")).toBe(false);
    expect(url.hash).toBe("");expect(url.searchParams.get("size")).toBe("128");
  });
  it.each([
    "https://img.logo.dev/ticker/OTHER?token=pk_fictional",
    "https://img.logo.dev.evil.example/ticker/SYN",
    "https://evil.example/logo.dev/ticker/SYN",
    "http://img.logo.dev/ticker/SYN",
    "https://user:password@img.logo.dev/ticker/SYN",
    "https://img.logo.dev:444/ticker/SYN",
    "javascript:alert(1)",
    "/images/placeholder-stock-logo.png",
  ])("does not misidentify the company or leave Logo.dev for %s", url => {
    expect(stockLogoSource("SYN", url, "")).toBeNull();
    expect(new URL(stockLogoSource("SYN", url, "pk_fictional")!).pathname).toBe("/ticker/SYN");
  });
  it.each(["", " ", "--", "UNKNOWN", "SYN/OTHER", "SYN?token=x", "A..B", "A-", "A".repeat(21)])("does not request malformed/unknown ticker %s", symbol => {
    expect(stockLogoSource(symbol, "", "pk_fictional")).toBeNull();
  });
  it("does not create any network image without configured or supplied source", () => {
    expect(stockLogoSource("SYN", "", "")).toBeNull();
    expect(normalizeLogoSymbol(" syn ")).toBe("SYN");
    expect(html("SYN", null)).toContain("SYN ticker; logo unavailable");
    expect(html("SYN", null)).not.toContain("<img");
    expect(html("SYN", null)).not.toContain("placeholder-stock-logo");
    expect(html("", null)).toContain("? ticker; logo unavailable");
  });
});

describe("actual image handlers and rendered lifecycle", () => {
  it("shows an accessible ticker while loading, then a real loaded image", () => {
    const node=frame();
    expect(html()).toContain("SYN ticker; logo loading");
    expect(html()).toContain(">SYN</span>");
    expect(image(node)?.props.loading).toBe("eager");
    image(node)!.props.onLoad({currentTarget:{naturalWidth:64,naturalHeight:64}});
    expect(html()).toContain("SYN company logo");
    expect(html()).toContain("opacity-100");
    expect(html()).not.toContain(">SYN</span>");
  });
  it("removes a failed image without changing src or looping, and retains visible ticker", () => {
    const node=frame(), img=image(node)!;
    const target={src,naturalWidth:0,naturalHeight:0};
    img.props.onError({currentTarget:target});
    expect(image(frame())).toBeUndefined();
    expect(target.src).toBe(src);
    expect(html()).toContain("SYN ticker; logo unavailable");
    expect(html()).toContain(">SYN</span>");
    for(let i=0;i<5;i++) expect(image(frame())).toBeUndefined();
    expect(vi.getTimerCount()).toBe(0);
  });
  it.each([[0,0],[1,1],[64,1]])("rejects blank/tracking-pixel image dimensions %s x %s", (width,height) => {
    image(frame())!.props.onLoad({currentTarget:{naturalWidth:width,naturalHeight:height}});
    expect(html()).toContain("logo unavailable");
    expect(image(frame())).toBeUndefined();
  });
  it("times out to a visible terminal fallback and ignores a late load", () => {
    const node=frame();
    vi.advanceTimersByTime(LOGO_LOAD_TIMEOUT_MS);
    image(node)!.props.onLoad({currentTarget:{naturalWidth:64,naturalHeight:64}});
    expect(html()).toContain("logo unavailable");
    expect(image(frame())).toBeUndefined();
  });
  it("cleans the loading deadline once loaded or unmounted", () => {
    const node=frame();
    expect(vi.getTimerCount()).toBe(1);
    image(node)!.props.onLoad({currentTarget:{naturalWidth:64,naturalHeight:64}});
    frame(); expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(LOGO_LOAD_TIMEOUT_MS*2);
    expect(html()).toContain("company logo");
    hooks.status=undefined; frame(); hooks.cleanup?.();
    expect(vi.getTimerCount()).toBe(0);
  });
  it("keys the actual wrapper by both symbol and source, keeping repeated renders stable", () => {
    const a=StockLogo({symbol:"SYN",logoUrl:src});
    const repeat=StockLogo({symbol:" syn ",logoUrl:src});
    const changedSource=StockLogo({symbol:"SYN",logoUrl:src+"&size=128"});
    const changedSymbol=StockLogo({symbol:"OTHER",logoUrl:"https://img.logo.dev/ticker/OTHER?token=pk_fictional_test"});
    expect(a.key).toBe(repeat.key);
    expect(a.key).not.toBe(changedSource.key);
    expect(a.key).not.toBe(changedSymbol.key);
    expect(changedSymbol.props.symbol).toBe("OTHER");
    // A React key change remounts the child; start that new mount in the harness.
    hooks.cleanup?.(); hooks.status=undefined;
    expect(html("OTHER",changedSymbol.props.src)).toContain("OTHER ticker; logo loading");
    expect(html("OTHER",changedSymbol.props.src)).not.toContain("SYN");
  });
  it("retains the original 32px tile geometry at desktop and mobile breakpoints", () => {
    const markup=html();
    expect(markup).toContain("h-8 w-8");
    expect(markup).toContain("jbravo-border-success");
    expect(markup).toContain("shadow-[0_0_10px_rgba(34,197,94,0.25)]");
    const tileClass=markup.match(/^<span class="([^"]*)"/)![1];
    expect(tileClass).not.toMatch(/(?:^|\\s)(?:[a-z]+:)?hidden(?:\\s|$)/);
    expect(markup).toContain('referrerPolicy="origin"');
    expect(markup).not.toContain("pk_fictional_test</");
  });
  it("keeps long ticker fallbacks inside the compact tile with full accessible identity", () => {
    const markup=html("LONGSYMBOL",null);
    expect(markup).toContain("LONGSYMBOL ticker; logo unavailable");
    expect(markup).toContain('title="LONGSYMBOL"');
    expect(markup).toContain('font-size:8px');
    expect(markup).toContain('>LONG…</span>');
  });
});
