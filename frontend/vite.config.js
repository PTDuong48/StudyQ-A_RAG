import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  base: process.env.NODE_ENV === 'production' ? '/StudyQ-A_RAG/' : '/',
});
