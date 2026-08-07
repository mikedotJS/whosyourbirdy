import { useEffect, useState } from 'react'
import { applyUpdate, watchForUpdates } from '../lib/pwa'

/**
 * "A new version is ready" — and nothing happens until it is tapped.
 *
 * The alternative, `skipWaiting()` in the worker, replaces the running
 * application without asking. In an app whose main job can take two minutes and
 * involves a 52 MB download, that is a silent disappearance of exactly the kind
 * the rest of this project refuses.
 */
export function UpdatePill() {
  const [waiting, setWaiting] = useState<ServiceWorker | null>(null)

  useEffect(() => watchForUpdates(setWaiting), [])

  if (!waiting) return null

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-[calc(var(--panel-inset)+4.5rem)] z-50 flex justify-center px-4">
      <button
        type="button"
        onClick={() => applyUpdate(waiting)}
        className="pointer-events-auto flex h-11 items-center gap-2 rounded-full border border-line-strong bg-raised px-4 text-sm shadow-lg"
      >
        <span
          aria-hidden
          className="h-1.5 w-1.5 rounded-full bg-focus"
        />
        Nouvelle version disponible — recharger
      </button>
    </div>
  )
}
