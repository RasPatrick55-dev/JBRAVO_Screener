import React from "react";
import { createRoot } from "react-dom/client";
import "./styles/tailwind.css";
import App from "./App";

const container = document.getElementById("root");

if (container) {
  createRoot(container).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
