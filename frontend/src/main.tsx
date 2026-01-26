import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/tailwind.css";
import App from "./App";

const container = document.getElementById("root");

const applyStoredTheme = () => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const stored = window.localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const nextTheme = stored === "dark" || stored === "light" ? stored : prefersDark ? "dark" : "light";
    document.documentElement.classList.toggle("dark", nextTheme === "dark");
  } catch {
    // Ignore storage access issues and fall back to default theme.
  }
};

applyStoredTheme();

if (container) {
  createRoot(container).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
