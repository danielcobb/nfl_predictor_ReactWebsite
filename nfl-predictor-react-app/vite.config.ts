import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves this app from a /nfl_predictor_ReactWebsite/ subpath, but
  // Vercel serves it from the domain root. Vercel sets VERCEL=1 during its build,
  // so branch on that to keep both deploy targets working.
  base: process.env.VERCEL ? "/" : "/nfl_predictor_ReactWebsite/",
})
