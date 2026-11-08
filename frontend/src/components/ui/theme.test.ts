import { afterEach, describe, expect, it, vi } from "vitest";
import { currentTheme, initializeTheme, setTheme, subscribeTheme } from "./theme";

function environment(stored: string | null = null, systemDark = false, blocked = false) {
  const classes = new Set<string>();
  const events = new EventTarget();
  const storage = {
    getItem: vi.fn(() => { if (blocked) throw new Error("Storage disabled"); return stored; }),
    setItem: vi.fn((_key: string, value: string) => { if (blocked) throw new Error("Storage disabled"); stored = value; }),
  };
  vi.stubGlobal("document", { documentElement: { classList: {
    contains: (name: string) => classes.has(name),
    toggle: (name: string, enabled: boolean) => enabled ? classes.add(name) : classes.delete(name),
  } } });
  vi.stubGlobal("window", { localStorage: storage, matchMedia: () => ({ matches: systemDark }),
    addEventListener: events.addEventListener.bind(events), removeEventListener: events.removeEventListener.bind(events),
    dispatchEvent: events.dispatchEvent.bind(events) });
  return { storage, events };
}

afterEach(() => vi.unstubAllGlobals());

describe("shared page theme", () => {
  it.each([['light',true,'light'],['dark',false,'dark'],[null,true,'dark'],['invalid',false,'light']] as const)(
    "initializes stored %s with system dark %s", (stored, systemDark, expected) => {
      environment(stored, systemDark); initializeTheme(); expect(currentTheme()).toBe(expected);
    });
  it("keeps desktop and mobile toggle subscribers synchronized and cleans up", () => {
    const { storage } = environment(); initializeTheme();
    const mobile = vi.fn(), desktop = vi.fn();
    const stopMobile = subscribeTheme(mobile), stopDesktop = subscribeTheme(desktop);
    setTheme('dark'); expect(currentTheme()).toBe('dark');
    expect(mobile).toHaveBeenCalledTimes(1); expect(desktop).toHaveBeenCalledTimes(1);
    expect(storage.setItem).toHaveBeenCalledWith('theme','dark');
    stopMobile(); setTheme('light'); expect(mobile).toHaveBeenCalledTimes(1); expect(desktop).toHaveBeenCalledTimes(2);
    stopDesktop(); setTheme('dark'); expect(desktop).toHaveBeenCalledTimes(2);
  });
  it("still initializes and switches when storage is blocked", () => {
    environment(null, true, true); initializeTheme(); expect(currentTheme()).toBe('dark');
    expect(() => setTheme('light')).not.toThrow(); expect(currentTheme()).toBe('light');
  });
  it("responds to another tab's theme preference", () => {
    const { events, storage } = environment('light'); initializeTheme();
    const listener = vi.fn(), stop = subscribeTheme(listener);
    storage.setItem('theme','dark');
    events.dispatchEvent(Object.assign(new Event('storage'), {key:'theme'}));
    expect(currentTheme()).toBe('dark'); expect(listener).toHaveBeenCalledOnce(); stop();
  });
});
