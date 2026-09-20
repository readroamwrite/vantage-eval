import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds straight into the Python package so `vantage dashboard` can serve it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../vantage/dashboard/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8501" } },
});
