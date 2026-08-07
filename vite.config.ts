import { createHash } from 'node:crypto'
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

const SW_TEMPLATE = resolve(__dirname, 'scripts/sw-template.js')

/**
 * Emit the service worker with a real precache list.
 *
 * The list cannot be written by hand: the asset names carry content hashes that
 * only exist after the bundle is generated. Reading them out of
 * `generateBundle` is the same shape as `ortRuntime()` above, and it avoids
 * pulling in Workbox to solve one string substitution.
 *
 * What goes in is the shell — document, chunks, stylesheet, self-hosted font,
 * icons. What stays out is anything large enough that downloading it silently
 * would be a decision the user should get to make: the 52 MB model and the 24 MB
 * WASM runtime are cached at runtime, on first use. The parity harness is a
 * build entry, not part of the app, so it is excluded too.
 */
function pwaAssets(): Plugin {
  const STATIC = ['/favicon.svg', '/manifest.webmanifest']
  const ICONS = ['192.png', '512.png', 'maskable-512.png'].map((n) => `/icons/icon-${n}`)

  return {
    name: 'pwa-assets',
    apply: 'build',

    generateBundle(_options, bundle) {
      const emitted = Object.keys(bundle)
        // Basename, not full path: the harness bundle lands at
        // `assets/parity-<hash>.js`, so a prefix test on the whole name misses
        // it and quietly precached 400 KB of test code.
        .filter((name) => !(name.split('/').pop() ?? '').startsWith('parity'))
        .filter((name) => !name.startsWith('ort/'))
        // The .jsep.wasm copy Rollup emits for the ORT import is 24 MB and is
        // served from /ort/ anyway.
        .filter((name) => !name.endsWith('.wasm'))
        .map((name) => `/${name}`)

      const precache = [...new Set(['/index.html', ...emitted, ...STATIC, ...ICONS])].sort()

      // Any change to the shell changes this, which is what retires the previous
      // cache in the worker's activate step.
      const version = createHash('sha256').update(precache.join('\n')).digest('hex').slice(0, 12)

      // The runtime cache is keyed on this rather than on the build, because the
      // `/ort/` filenames are stable across releases: cache-first on a stable
      // name would serve an obsolete 24 MB runtime forever after an upgrade, and
      // keying on the build would discard those 24 MB on every deploy instead.
      const ortVersion = JSON.parse(
        readFileSync(resolve(__dirname, 'node_modules/onnxruntime-web/package.json'), 'utf8'),
      ).version

      const source = readFileSync(SW_TEMPLATE, 'utf8')
        .replace('__VERSION__', version)
        .replace('__ORT_VERSION__', ortVersion)
        .replace('__PRECACHE__', JSON.stringify(precache, null, 2))

      this.emitFile({ type: 'asset', fileName: 'sw.js', source })
    },
  }
}

// Fully static build: no SSR, no server, no API routes. `onnxruntime-web` and the
// Web Audio API are browser-only, so there is no server-render pass to opt out of.
export default defineConfig({
  plugins: [react(), tailwindcss(), ortRuntime(), pwaAssets()],
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
