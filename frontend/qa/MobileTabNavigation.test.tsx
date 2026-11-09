// @vitest-environment jsdom
import { act, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import NavbarDesktop from "../src/components/navbar/NavbarDesktop";
import App from "../src/App";

const labels = ["Dashboard", "Account", "Trades", "Positions", "Execute", "Screener", "ML Pipeline"];
function screen() { return { default: ({ activeTab, onTabSelect }: any) => <nav aria-label="Primary">
  {labels.map(label => <button key={label} aria-current={label === activeTab ? "page" : undefined}
    onClick={() => onTabSelect(label)}>{label}</button>)}
</nav> }; }
vi.mock("../src/screens/DashboardHealth", screen);
vi.mock("../src/screens/AccountOverview", screen);
vi.mock("../src/screens/TradesOverview", screen);
vi.mock("../src/screens/PositionsMonitoring", screen);
vi.mock("../src/screens/ExecuteTrades", screen);
vi.mock("../src/screens/ScreenerOverview", screen);
vi.mock("../src/screens/MLPipeline", screen);
vi.mock("../src/screens/PlaceholderScreen", screen);

let root: Root, host: HTMLDivElement;
beforeEach(() => {
  (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
  window.history.replaceState(null, "", "/");
  vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
  vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  host = document.createElement("div"); document.body.append(host); root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});
const render = (element: React.ReactNode) => act(() => root.render(element));
const select = (label: string) => act(() => {
  const button = [...host.querySelectorAll("button")].find(node => node.textContent === label)!;
  button.focus(); button.click();
});
const hashChange = (hash: string) => act(() => {
  window.history.replaceState(null, "", hash || "/");
  window.dispatchEvent(new HashChangeEvent("hashchange"));
});

describe("top-level navigation lifecycle", () => {
  it("starts at a direct hash without resetting initial scroll or stealing focus", () => {
    window.history.replaceState(null, "", "#execute"); render(<App />);
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe("Execute");
    expect(window.scrollTo).not.toHaveBeenCalled(); expect(document.activeElement).toBe(document.body);
  });
  it("resets each actual screen change once and restores the remounted navigation focus", () => {
    const focus = vi.spyOn(HTMLElement.prototype, "focus");
    render(<StrictMode><App /></StrictMode>);
    select("Execute"); expect(window.scrollTo).toHaveBeenCalledTimes(1);
    expect(window.scrollTo).toHaveBeenLastCalledWith({ top: 0, left: 0, behavior: "instant" });
    expect(document.activeElement).toBe(host.querySelector('[aria-current="page"]'));
    expect(focus).toHaveBeenLastCalledWith({ preventScroll: true });
    expect(window.location.hash).toBe("#execute");
    select("Screener"); select("ML Pipeline"); expect(window.scrollTo).toHaveBeenCalledTimes(3);
    expect(document.activeElement?.textContent).toBe("ML Pipeline");
    // Same-page clicks, parent renders, and the queued hashchange for the latest
    // click must not reset a reader's newly established scroll/focus.
    select("ML Pipeline"); render(<StrictMode><App /></StrictMode>); hashChange("#ml-pipeline");
    expect(window.scrollTo).toHaveBeenCalledTimes(3);
  });
  it("handles history/hash changes, ignores unknown hashes, and keeps the latest navigation", () => {
    render(<App />); hashChange("#execute"); hashChange("#screener");
    hashChange("#execute"); // Browser Back's hashchange
    expect(document.activeElement?.textContent).toBe("Execute");
    hashChange("#screener"); // Browser Forward's hashchange
    expect(document.activeElement?.textContent).toBe("Screener");
    expect(window.scrollTo).toHaveBeenCalledTimes(4);
    hashChange("#unknown"); expect(window.scrollTo).toHaveBeenCalledTimes(4);
    hashChange(""); expect(document.activeElement?.textContent).toBe("Dashboard");
    hashChange("#screener"); expect(document.activeElement?.textContent).toBe("Screener");
    select("ML Pipeline"); hashChange("#ml-pipeline");
    expect(host.querySelector('[aria-current="page"]')?.textContent).toBe("ML Pipeline");
    expect(window.scrollTo).toHaveBeenCalledTimes(7);
  });
});

describe("mounted mobile navigation strip", () => {
  let width: number;
  beforeEach(() => {
    // jsdom has no layout: use representative fixed button geometry and a real
    // mounted nav, refs, effects, resize events and scrollLeft property.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      if (this.tagName === "NAV") return { left: 16, right: width - 16, width: width - 32, height: 28 } as DOMRect;
      const index = labels.indexOf(this.textContent ?? "");
      if (this.tagName === "BUTTON" && index >= 0) {
        const left = 16 + index * 88 - (this.parentElement?.scrollLeft ?? 0);
        return { left, right: left + 80, width: 80, height: 28 } as DOMRect;
      }
      return { left: 0, right: 0, width: 0, height: 0 } as DOMRect;
    });
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
      return this.tagName === "NAV" ? width - 32 : 0;
    });
  });
  const nav = (active: string, key = "screen") => <NavbarDesktop key={key}
    tabs={labels.map(label => ({ label, isActive: label === active }))} rightBadges={[]} />;
  const expectVisible = () => {
    const strip = host.querySelector("nav")!;
    const selected = strip.querySelector('[aria-current="page"]')!.getBoundingClientRect();
    expect(selected.left).toBeGreaterThanOrEqual(16);
    expect(selected.right).toBeLessThanOrEqual(width - 16);
    expect(window.scrollTo).not.toHaveBeenCalled();
  };
  it.each([320, 375, 390, 430])("reveals Execute/Screener/ML after screen remounts at %ipx", value => {
    width = value;
    for (const label of ["Execute", "Screener", "ML Pipeline", "Dashboard", "Execute"]) {
      render(nav(label, label)); expectVisible();
    }
  });
  it("disconnects its observer and removes the resize handler when unmounted", () => {
    const observers: { observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];
    vi.stubGlobal("ResizeObserver", class {
      observe = vi.fn(); disconnect = vi.fn();
      constructor() { observers.push(this); }
    });
    const remove = vi.spyOn(window, "removeEventListener");
    width = 375; render(nav("Execute"));
    const strip = host.querySelector("nav")!;
    const observer = observers.find(item => item.observe.mock.calls.some(([node]) => node === strip))!;
    expect(observer).toBeDefined();
    render(<div />); expect(observer.disconnect).toHaveBeenCalledOnce();
    expect(remove.mock.calls.some(([event]) => event === "resize")).toBe(true);
  });
  it("reveals on resize and active change but does not override manual scrolling on clock/poll renders", () => {
    width = 430; render(nav("ML Pipeline")); expectVisible();
    width = 320; act(() => window.dispatchEvent(new Event("resize"))); expectVisible();
    const strip = host.querySelector("nav")!; strip.scrollLeft = 0;
    render(nav("ML Pipeline")); expect(strip.scrollLeft).toBe(0);
    render(nav("Execute")); expectVisible();
    render(nav("Dashboard")); expectVisible(); expect(strip.scrollLeft).toBe(0);
  });
});
