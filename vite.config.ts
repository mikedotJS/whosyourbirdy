import { createReadStream, existsSync, readdirSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const ORT_DIR = resolve(__dirname, 'node_modules/onnxruntime-web/dist')
const ORT_FILES = /^ort-.*\.(wasm|mjs)$/

/**
 * Serve ONNX Runtime's WASM binaries at a stable `/ort/` path, in dev and build.
 *
 * They cannot live in `public/`. ORT loads its runtime with a *dynamic ES module
 * import* of `${wasmPaths}ort-wasm-simd-threaded.jsep.mjs`, and Vite's dev server
 * refuses to serve anything under `publicDir` that is reached by an import — it
 * answers HTTP 500 with "this file is in /public ... and therefore should not be
 * imported from source code". The production build was unaffected, so the whole
 * app was undevelopable while `pnpm build` looked perfectly healthy.
 *
 * Serving them from a middleware in dev and emitting them as build assets gives
 * both modes the same URL without going through publicDir at all.
 */
function ortRuntime(): Plugin {
  const files = () => (existsSync(ORT_DIR) ? readdirSync(ORT_DIR).filter((f) => ORT_FILES.test(f)) : [])

  return {
    name: 'ort-runtime',

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url?.split('?')[0] ?? ''
        if (!path.startsWith('/ort/')) return next()

        const name = path.slice('/ort/'.length)
        if (!ORT_FILES.test(name)) return next()

        const file = resolve(ORT_DIR, name)
        if (!file.startsWith(ORT_DIR) || !existsSync(file)) return next()

        res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript')
        createReadStream(file).pipe(res)
      })
    },

    generateBundle() {
      const found = files()
      if (found.length === 0) {
        this.warn('onnxruntime-web not installed; /ort/ will 404 at runtime')
        return
      }
      for (const name of found) {
        this.emitFile({
          type: 'asset',
          fileName: `ort/${name}`,
          source: readFileSync(resolve(ORT_DIR, name)),
        })
      }
    },
  }
}

// Fully static build: no SSR, no server, no API routes. `onnxruntime-web` and the
// Web Audio API are browser-only, so there is no server-render pass to opt out of.
export default defineConfig({
  plugins: [react(), tailwindcss(), ortRuntime()],
  optimizeDeps: {
    // ORT resolves its .wasm/.mjs pairs at runtime from `ort.env.wasm.wasmPaths`.
    // Pre-bundling rewrites those relative URLs and breaks the lookup.
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
