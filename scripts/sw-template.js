/*
 * Service worker. Written by hand; the precache list is injected at build time
 * by the `pwaAssets()` plugin in vite.config.ts, which is the only way to know
 * the hashed asset names.
 *
 * No Workbox. The whole policy is thirty lines of `caches` calls, and the
 * interesting decisions here are about *what not to cache* — which a generic
 * tool cannot make for us.
 *
 * ---
 *
 * What is precached at install: the shell only — the document, the application
 * chunks, the stylesheet, the self-hosted font, the icons. About a megabyte.
 *
 * What is deliberately NOT precached:
 *
 *  - **The 52 MB model.** It already has its own Cache API entry in
 *    `src/lib/birdnet/model.ts`, keyed on the manifest's SHA-256 so a rebuild
 *    invalidates it. Precaching it here would download 52 MB silently the moment
 *    someone visits, and then store it twice. It is fetched on the first
 *    analysis, with a byte-accurate progress bar, because that is a large enough
 *    event to deserve being visible.
 *  - **The 24 MB ONNX Runtime WASM.** Same argument, smaller number. It is
 *    cached at runtime the first time an analysis actually needs it, so the
 *    second analysis works offline and the first visit costs nothing.
 *  - **The parity harness.** It is a build entry, not part of the app.
 */

const VERSION = '__VERSION__'
const SHELL = 'shell-' + VERSION
/*
 * Keyed on the ONNX Runtime package version, not on the build.
 *
 * The files under `/ort/` have *stable* names — `ort-wasm-simd-threaded.jsep.wasm`
 * is called that in every release — so a cache-first strategy would keep serving
 * an old 24 MB runtime forever after an upgrade. Naming the cache after the shell
 * version would fix that and throw the 24 MB away on every deploy instead. The
 * package version is the thing that actually changes when those bytes change.
 */
const RUNTIME = 'runtime-ort-__ORT_VERSION__'
const PRECACHE = __PRECACHE__
/**
 * Where this deployment lives. `/` on a domain root, `/whosyourbirdy/` on a
 * GitHub Pages project site. Every path below is built from it: a worker served
 * from a subpath that reasoned in absolute `/…` terms would miss its own assets
 * and answer navigations with a document that does not exist.
 */
const BASE = '__BASE__'

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL).then((cache) => cache.addAll(PRECACHE)))
  // No skipWaiting(). A new worker that activates on its own swaps the running
  // application's code mid-analysis. It waits until the page asks.
})

self.addEventListener('message', (event) => {
  if (event.data?.type === 'skip-waiting') self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter(
              (key) =>
                (key.startsWith('shell-') && key !== SHELL) ||
                (key.startsWith('runtime-ort-') && key !== RUNTIME),
            )
            .map((key) => caches.delete(key)),
        ),
      )
      // Control the pages that are already open, so the first visit is offline-
      // capable without a second load. This is not skipWaiting: a worker that is
      // waiting still waits.
      .then(() => self.clients.claim()),
  )
})

/** Runtime-cached on first use: large, immutable, and not needed to boot. */
function isRuntimeAsset(url) {
  const path = url.pathname.slice(BASE.length)
  return path.startsWith('ort/') || /^models\/labels_\w+\.txt$/.test(path)
}

self.addEventListener('fetch', (event) => {
  const request = event.request
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return

  // The model is somebody else's business — see the note at the top.
  if (url.pathname.endsWith('.onnx')) return

  // Outside this deployment's own subtree — another project site on the same
  // github.io domain, for instance — is none of our business.
  if (!url.pathname.startsWith(BASE)) return

  // A navigation always resolves to the shell document. Cache-first, because the
  // update path is explicit: a new worker parks itself and the page offers to
  // reload. Network-first here would trade that for a spinner on every launch.
  if (request.mode === 'navigate') {
    event.respondWith(caches.match(BASE + 'index.html').then((hit) => hit ?? fetch(request)))
    return
  }

  if (isRuntimeAsset(url)) {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ??
          fetch(request).then((response) => {
            // Opaque or failed responses would poison the cache with something
            // that can never be served successfully.
            if (response.ok && response.type === 'basic') {
              const copy = response.clone()
              void caches.open(RUNTIME).then((cache) => cache.put(request, copy))
            }
            return response
          }),
      ),
    )
    return
  }

  event.respondWith(caches.match(request).then((hit) => hit ?? fetch(request)))
})
