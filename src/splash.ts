/**
 * Take the splash screen away.
 *
 * The splash itself lives inline in `index.html` so it can paint before this
 * bundle exists. All that is left here is dismissing it, and the only interesting
 * decision is *when*.
 *
 * It is dismissed as soon as the app has actually painted — never on a timer that
 * would make everyone wait for a fixed animation. The one concession is a short
 * floor: on a warm cache the app is ready in about 40 ms, and a splash that
 * appears and vanishes inside two frames reads as a glitch rather than as an
 * intro. `MIN_MS` is the smallest value at which it still reads as deliberate.
 *
 * It is also removed from the DOM rather than left at `opacity: 0`, which would
 * keep a full-viewport element and its looping animations alive over the app for
 * the rest of the session.
 */

/** Long enough to read as intentional, short enough not to be a wait. */
const MIN_MS = 620

/** Matches the longest transition on `#splash.out` in index.html. */
const FADE_MS = 560

export function dismissSplash(): void {
  const splash = document.getElementById('splash')
  if (!splash) return

  const start = performance.now()

  const hide = () => {
    splash.classList.add('out')
    // `transitionend` alone is not enough: it never fires if the element is in a
    // hidden tab, or under `prefers-reduced-motion` with transitions disabled at
    // the OS level, and the overlay would stay in the tree forever.
    const remove = () => splash.remove()
    splash.addEventListener('transitionend', remove, { once: true })
    window.setTimeout(remove, FADE_MS + 120)
  }

  // Two frames after mount: the first commits the DOM, the second is the one
  // that actually gets painted. Hiding any earlier can uncover a blank app.
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      const waited = performance.now() - start
      if (waited >= MIN_MS) hide()
      else window.setTimeout(hide, MIN_MS - waited)
    }),
  )
}
