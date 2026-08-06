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
  focused: Species | null
  playing: number | null
  onFocus: (species: Species | null) => void
  onPlay: (detection: Detection) => void
}

/**
 * Species, with their occurrences.
 *
 * This carries identity — which is why it is text-first. The spectrogram cannot
 * name 6522 possible classes with colour, so the link runs the other way: focus a
 * species here and its bands light up over there.
 */
export function SpeciesList({ groups, focused, playing, onFocus, onPlay }: Props) {
  if (groups.length === 0) return null

  return (
    <ul className="flex flex-col divide-y divide-neutral-200 dark:divide-neutral-800">
      {groups.map((group, index) => {
        const isFocused = focused?.index === group.species.index

        return (
          <li
            key={group.species.index}
            className="row-in"
            style={{ '--row-index': index } as React.CSSProperties}
          >
            <button
              type="button"
              onClick={() => onFocus(isFocused ? null : group.species)}
              onMouseEnter={() => !focused && onFocus(group.species)}
              onMouseLeave={() => !isFocused && onFocus(null)}
              aria-expanded={isFocused}
              className={[
                'flex w-full items-center gap-3 py-2.5 text-left transition-colors',
                isFocused ? 'bg-neutral-50 dark:bg-neutral-900' : 'hover:bg-neutral-50 dark:hover:bg-neutral-900',
              ].join(' ')}
            >
              <span
                aria-hidden
                className={[
                  'h-8 w-1 shrink-0 rounded-full transition-colors',
                  isFocused ? 'bg-[#2a78d6] dark:bg-[#3987e5]' : 'bg-neutral-300 dark:bg-neutral-700',
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
              <ul className="flex flex-wrap gap-1.5 pb-3 pl-4">
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
                          isPlaying
                            ? 'bg-[#eb6834] text-white dark:bg-[#d95926]'
                            : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200 dark:bg-neutral-800 dark:text-neutral-300 dark:hover:bg-neutral-700',
                        ].join(' ')}
                      >
                        {clock(occurrence.start)}
                        <span className="ml-1.5 opacity-60">{occurrence.score.toFixed(2)}</span>
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
