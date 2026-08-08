/**
 * Service worker registration.
 *
 * Two deliberate refusals here, both of which are the easy default elsewhere:
 *
 * 1. **Never in development.** A caching worker between the dev server and the
 *    browser turns every edit into a guess about which version you are looking
 *    at. The `public/ort` episode was expensive enough; this one would be worse
 *    because it survives a reload.
 * 2. **Never `skipWaiting()` on its own.** A new worker that activates
 *    immediately swaps the application's code under whatever is running — during
 *    a 52 MB model download, or a two-minute analysis, or a playing segment. The
 *    new version waits, the user is told, and the swap happens when they say so.
 */

/**
 * Both the URL and the scope follow the base path: a service worker can only
 * control pages at or below its own directory, so registering `/sw.js` from a
 * site served at `/whosyourbirdy/` would register a worker that controls the
 * whole domain and, on GitHub Pages, is not even there.
 */
const SW_URL = `${import.meta.env.BASE_URL}sw.js`
const SW_SCOPE = import.meta.env.BASE_URL

export function watchForUpdates(onWaiting: (worker: ServiceWorker) => void): void {
  if (!import.meta.env.PROD) return
  if (!('serviceWorker' in navigator)) return

  const register = () => {
    void navigator.serviceWorker.register(SW_URL, { scope: SW_SCOPE }).then((registration) => {
      // A worker already parked from a previous visit.
      if (registration.waiting && navigator.serviceWorker.controller) {
        onWaiting(registration.waiting)
      }

      registration.addEventListener('updatefound', () => {
        const next = registration.installing
        if (!next) return
        next.addEventListener('statechange', () => {
          // `controller` is null on the very first install. Announcing an
          // "update" then would tell a first-time visitor to reload the page
          // they just opened.
          if (next.state === 'installed' && navigator.serviceWorker.controller) {
            onWaiting(next)
          }
        })
      })
    })
  }

  // Registering after `load` keeps it from competing with the first paint. The
  // `readyState` test is not belt-and-braces: this runs from a React effect, and
  // on a fast connection `load` has usually already fired by then — so waiting
  // for an event that will never come again meant the worker was never
  // registered at all, and the app was never installable or offline-capable.
  // The smoke test found it by hanging on `navigator.serviceWorker.ready`.
  if (document.readyState === 'complete') register()
  else window.addEventListener('load', register, { once: true })
}

/** Hand over to the waiting worker, then reload once it has taken control. */
export function applyUpdate(worker: ServiceWorker): void {
  navigator.serviceWorker.addEventListener('controllerchange', () => window.location.reload(), {
    once: true,
  })
  worker.postMessage({ type: 'skip-waiting' })
}
