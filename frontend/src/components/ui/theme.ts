export type ThemeMode = "light" | "dark";
const themeEvent = "jbravo-theme-change";

export const currentTheme = (): ThemeMode =>
  typeof document !== "undefined" && document.documentElement.classList.contains("dark") ? "dark" : "light";

export function initializeTheme() {
  if (typeof window === "undefined") return;
  let theme: ThemeMode = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  try {
    const stored = window.localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") theme = stored;
  } catch {
    // Storage may be disabled; the system preference still works.
  }
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function setTheme(theme: ThemeMode) {
  document.documentElement.classList.toggle("dark", theme === "dark");
  try {
    window.localStorage.setItem("theme", theme);
  } catch {
    // Theme switching should work even when persistence is unavailable.
  }
  window.dispatchEvent(new Event(themeEvent));
}

export function subscribeTheme(onChange: () => void) {
  const onStorage = (event: StorageEvent) => {
    if (event.key === "theme" || event.key === null) {
      initializeTheme();
      onChange();
    }
  };
  window.addEventListener(themeEvent, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(themeEvent, onChange);
    window.removeEventListener("storage", onStorage);
  };
}
