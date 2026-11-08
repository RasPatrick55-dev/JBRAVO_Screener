import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/tailwind.css";
import App from "./App";
import { initializeTheme } from "./components/ui/theme";

const container = document.getElementById("root");

initializeTheme();

if (container) {
  createRoot(container).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
