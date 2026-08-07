import { ATTRIBUTION, ATTRIBUTION_URL, MODEL_LICENSE } from '../lib/birdnet/constants'

/**
 * Required by CC BY-NC-SA 4.0, and kept in the page rather than behind an
 * "about" link so it is actually visible.
 */
export function Attribution() {
  return (
    <footer className="border-t border-line pt-4 text-xs leading-relaxed text-ink-3">
      <a
        href={ATTRIBUTION_URL}
        target="_blank"
        rel="noreferrer"
        className="underline underline-offset-2 transition-colors hover:text-ink"
      >
        {ATTRIBUTION}
      </a>
      <p className="mt-1">
        Modèle sous {MODEL_LICENSE} — usage non commercial. Ce projet est non commercial.
      </p>
    </footer>
  )
}
