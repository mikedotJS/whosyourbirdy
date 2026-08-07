import { useEffect, useRef, useState } from 'react'

interface Props {
  header: React.ReactNode
  /** Bottom action bar. Always rendered — an app's bar does not come and go. */
  bar: React.ReactNode
  children: React.ReactNode
}

/**
 * The shell: a phone, at every size.
 *
 * Three rows that never move — header, scrolling content, action bar — inside a
 * column capped at 420px. Past 480 × 560 the column lifts off the background and
 * becomes a floating panel; the geometry, including the safe-area handling and
 * the height cap, lives in `index.css` next to the variables it shares with the
 * bottom sheet.
 *
 * The point of the fixed rows is not decoration. A bar that scrolls away is a
 * page; a bar that stays is an app, and the primary action stops being something
 * you have to go and find.
 */
export function AppShell({ header, bar, children }: Props) {
  const sentinelRef = useRef<HTMLDivElement>(null)
  const [scrolled, setScrolled] = useState(false)

  /*
   * The header's bottom rule appears only once content has moved under it.
   *
   * Watching a 1px sentinel rather than listening for `scroll`: the observer
   * fires twice for the whole session instead of on every frame of every swipe,
   * and it cannot desynchronise from the layout the way a cached scrollTop can.
   */
  useEffect(() => {
    const sentinel = sentinelRef.current
    if (!sentinel) return
    const observer = new IntersectionObserver(
      ([entry]) => setScrolled(!entry.isIntersecting),
      { threshold: 1 },
    )
    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [])

  return (
    <div className="app-frame grain">
      <div className="app-panel">
        <header
          className={[
            'app-header z-10 px-4 pb-2.5 transition-colors duration-150',
            scrolled ? 'border-b border-line bg-surface' : 'border-b border-transparent',
          ].join(' ')}
        >
          {header}
        </header>

        <main className="app-scroll px-4">
          <div ref={sentinelRef} aria-hidden className="h-px" />
          {children}
        </main>

        <div className="app-bar z-10 border-t border-line bg-surface px-4 pt-2.5 pb-2.5">
          {bar}
        </div>
      </div>
    </div>
  )
}
