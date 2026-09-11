import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built ahead of time into ../static/ and served by the FastAPI process, so the
// CML Application serves the UI and the API from one port.
//
// base must stay absolute: the app uses BrowserRouter, so a deep link such as
// /configurations/abc would resolve relative asset paths against
// /configurations/ and 404. A CML Application owns the root of its own
// subdomain, so '/' is correct there.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Only for editing the UI locally; in CML nothing runs a dev server.
    proxy: {
      '/api': { target: 'http://localhost:8100', changeOrigin: true },
    },
  },
})
