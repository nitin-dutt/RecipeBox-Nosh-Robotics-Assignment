import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the API is reached through /api (proxied to the FastAPI service), so the browser only ever talks to ONE origin:
// no CORS. The target is http://localhost:8000 unless API_TARGET is set (for example API_TARGET=http://localhost:8001).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        "/api": { target: env.API_TARGET || "http://localhost:8000", changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api/, "") },
      },
    },
    test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"], css: false },
  };
});
