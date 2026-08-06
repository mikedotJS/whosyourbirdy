import type { Species } from '../lib/birdnet/labels'
import type { Detection } from '../lib/birdnet/types'

export interface SpeciesGroup {
  species: Species
  count: number
  bestScore: number
  occurrences: Detection[]
}

interface Props {
  groups: SpeciesGroup[]
  /** Pinned by a click. Survives the pointer moving away. */
  pinned: Species | null
  /** Hovered right now. Transient, and never overrides a pin. */
  hovered: Species | null
  playing: number | null
  onPin: (species: Species | null) => void
  onHover: (species: Species | null) => void
  onPlay: (detection: Detection) => void
}

/**
 * Species, with their occurrences.
 *
 * This carries identity — which is why it is text-first. The spectrogram cannot
 * name 6522 possible classes with colour, so the link runs the other way: focus a
 * species here and its bands light up over there.
 */
/**
 * Hover and pin are separate pieces of state on purpose.
 *
 * They used to share one slot, gated on each other, which produced two bugs at
 * once: hovering row A latched the focus so row B never highlighted (leaving the
 * picture pointing at a species the pointer was not on), and clicking a row
 * *deselected* it, because hover had already focused it by the time the click
 * landed. Kept apart, both behaviours fall out for free.
 */
export function SpeciesList({ groups, pinned, hovered, playing, onPin, onHover, onPlay }: Props) {
  if (groups.length === 0) return null

  return (
    <ul className="flex flex-col divide-y divide-neutral-200 dark:divide-neutral-800">
      {groups.map((group, index) => {
        const isPinned = pinned?.index === group.species.index
        // The pin wins; hover only shows through when nothing is pinned.
        const isFocused = pinned ? isPinned : hovered?.index === group.species.index

        return (
          <li
            key={group.species.index}
            className="row-in"
            style={{ '--row-index': index } as React.CSSProperties}
          >
            <button
              type="button"
              onClick={() => onPin(isPinned ? null : group.species)}
              onMouseEnter={() => onHover(group.species)}
              onMouseLeave={() => onHover(null)}
              onFocus={() => onHover(group.species)}
              onBlur={() => onHover(null)}
              aria-expanded={isFocused}
              aria-controls={`occurrences-${group.species.index}`}
              className={[
                'flex w-full items-center gap-3 py-2.5 text-left transition-colors',
                isFocused ? 'bg-neutral-50 dark:bg-neutral-900' : 'hover:bg-neutral-50 dark:hover:bg-neutral-900',
              ].join(' ')}
            >
              <span
                aria-hidden
                className={[
                  'h-8 w-1 shrink-0 rounded-full transition-colors',
                  isFocused ? 'bg-[#2a78d6] dark:bg-[#3987e5]' : 'bg-neutral-400 dark:bg-neutral-600',
                ].join(' ')}
              />

              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{group.species.commonName}</span>
                <span className="block truncate text-sm italic text-neutral-500 dark:text-neutral-400">
                  {group.species.scientificName}
                </span>
              </span>

              <span className="shrink-0 text-sm tabular-nums text-neutral-500 dark:text-neutral-400">
                {group.count}×
              </span>
              <span className="w-10 shrink-0 text-right text-sm tabular-nums">
                {group.bestScore.toFixed(2)}
              </span>
            </button>

            {isFocused && (
              <ul id={`occurrences-${group.species.index}`} className="flex flex-wrap gap-1.5 pb-3 pl-4">
                {group.occurrences.map((occurrence) => {
                  const key = occurrence.windowIndex * 10000 + occurrence.species.index
                  const isPlaying = playing === key
                  return (
                    <li key={key}>
                      <button
                        type="button"
                        onClick={() => onPlay(occurrence)}
                        aria-pressed={isPlaying}
                        aria-label={`${isPlaying ? 'Arrêter' : 'Écouter'} ${clock(occurrence.start)}, confiance ${occurrence.score.toFixed(2)}`}
                        className={[
                          'rounded px-2 py-1 text-xs tabular-nums transition-colors',
                          // Near-black ink on the orange rather than white:
                          // white measured 3.2:1 on #eb6834, under the 4.5:1
                          // that 12px text needs. Ink on the same orange is
                          // 6.6:1 and keeps the hue doing the signalling.
                          isPlaying
                            ? 'bg-[#eb6834] text-neutral-950 dark:bg-[#d95926] dark:text-neutral-950'
                            : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-200 dark:hover:bg-neutral-700',
                        ].join(' ')}
                      >
                        {clock(occurrence.start)}
                        {/* opacity-60 measured 2.04:1 on the orange chip. */}
                        <span className="ml-1.5 opacity-80">{occurrence.score.toFixed(2)}</span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </li>
        )
      })}
    </ul>
  )
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
