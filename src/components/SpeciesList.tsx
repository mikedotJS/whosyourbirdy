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
  /** Selected. Drives both the spectrogram's focus and the open sheet. */
  pinned: Species | null
  /** Hovered right now. Transient, and never overrides a selection. */
  hovered: Species | null
  /**
   * Class index of the species currently sounding, if any.
   *
   * The occurrences live in a sheet now, so with the sheet closed nothing on
   * screen would say which bird you are hearing — the keyboard shortcut would
   * start a segment and the list would look inert.
   */
  playingSpecies: number | null
  onPin: (species: Species | null) => void
  onHover: (species: Species | null) => void
}

/**
 * Species, with their best score and occurrence count.
 *
 * This carries identity — which is why it is text-first. The spectrogram cannot
 * name 6522 possible classes with colour, so the link runs the other way: select
 * a species here and its bands light up over there.
 *
 * Hover and selection are separate pieces of state on purpose. They used to
 * share one slot, gated on each other, which produced two bugs at once: hovering
 * row A latched the focus so row B never highlighted, and clicking a row
 * *deselected* it because hover had already focused it by the time the click
 * landed. Kept apart, both behaviours fall out for free.
 *
 * The occurrences themselves are not here any more — selecting a row opens them
 * in a bottom sheet. Inline, they pushed every row below them down the list by a
 * variable amount, so the thing you tapped moved out from under your thumb.
 */
export function SpeciesList({ groups, pinned, hovered, playingSpecies, onPin, onHover }: Props) {
  if (groups.length === 0) return null

  return (
    <ul id="species-list" className="flex flex-col divide-y divide-line">
      {groups.map((group, index) => {
        const isPinned = pinned?.index === group.species.index
        // The selection wins; hover only shows through when nothing is selected.
        const isFocused = pinned ? isPinned : hovered?.index === group.species.index
        const isPlaying = playingSpecies === group.species.index

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
              aria-expanded={isPinned}
              data-playing={isPlaying}
              aria-label={
                `${group.species.commonName}, ${group.species.scientificName}, ` +
                `${group.count} détection${group.count > 1 ? 's' : ''}, ` +
                `confiance maximale ${group.bestScore.toFixed(2)}` +
                (isPlaying ? ', en cours de lecture' : '')
              }
              className={[
                // min-h-14 rather than padding alone: a one-line species name
                // would otherwise make a shorter row than a two-line one, and
                // some rows would fall under the 44px touch minimum.
                'flex min-h-14 w-full items-center gap-3 py-2 text-left transition-colors duration-150',
                isFocused ? 'bg-raised' : 'hover:bg-hover',
              ].join(' ')}
            >
              {/* One rail, three states, and playing wins: while a segment is
                  sounding, which bird it belongs to matters more than which one
                  is selected. */}
              <span
                aria-hidden
                className={[
                  'h-9 w-1 shrink-0 rounded-full transition-colors duration-150',
                  isPlaying ? 'bg-play' : isFocused ? 'bg-focus' : 'bg-line-strong',
                ].join(' ')}
              />

              <span className="min-w-0 flex-1 select-text">
                <span className="block truncate font-medium">{group.species.commonName}</span>
                <span className="block truncate text-sm italic text-ink-3">
                  {group.species.scientificName}
                </span>
              </span>

              <span aria-hidden className="shrink-0 text-sm tabular-nums text-ink-3">
                {group.count}×
              </span>
              <span aria-hidden className="w-9 shrink-0 text-right text-sm tabular-nums">
                {group.bestScore.toFixed(2)}
              </span>
              <Chevron open={isPinned} />
            </button>
          </li>
        )
      })}
    </ul>
  )
}

/** Points right when closed, up when the sheet holding this row is open. */
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className={[
        'h-4 w-4 shrink-0 text-ink-3 transition-transform duration-150',
        open ? '-rotate-90' : '',
      ].join(' ')}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  )
}
