import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8050",
        changeOrigin: true,
      },
      "/assets": {
        target: "http://127.0.0.1:8050",
        changeOrigin: true,
      },
      "/ui-assets": {
        target: "http://127.0.0.1:8050",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
    emptyOutDir: true,
  },
});
