import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Fully static build: no SSR, no server, no API routes. `onnxruntime-web` and the
// Web Audio API are browser-only, so there is no server-render pass to opt out of.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    // ORT ships its own .wasm/.mjs pairs and resolves them at runtime from
    // `ort.env.wasm.wasmPaths`. Pre-bundling rewrites those relative URLs and
    // breaks the lookup, so we keep it out of the dep optimizer entirely.
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        // The parity harness is built like any other page so the browser-side
        // check runs against the real production bundle, not a dev-server one.
        main: resolve(__dirname, 'index.html'),
        parity: resolve(__dirname, 'parity-harness.html'),
      },
    },
    // The model is served from public/ as a plain static asset; nothing large
    // goes through the bundler, so anything big here is a mistake worth seeing.
    chunkSizeWarningLimit: 700,
  },
})
