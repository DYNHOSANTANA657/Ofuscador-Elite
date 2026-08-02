import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import path from "node:path";

export default defineConfig({
  root: path.resolve(__dirname, "portable-ui"),
  base: "/",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname) } },
  build: {
    outDir: path.resolve(__dirname, "backend/static"),
    emptyOutDir: true,
  },
});
