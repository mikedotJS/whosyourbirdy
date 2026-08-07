import { Sheet } from './Sheet'
import type { SpeciesGroup } from './SpeciesList'
import type { Detection } from '../lib/birdnet/types'

interface Props {
  group: SpeciesGroup | null
  playing: number | null
  onClose: () => void
  onPlay: (detection: Detection) => void
}

/**
 * Where a species' occurrences live.
 *
 * They used to expand inline under the row, which on a phone-width column meant
 * the list shifted under the thumb that had just tapped it. In a sheet the list
 * behind stays exactly where it was, and — because the sheet is capped at 60dvh
 * — the spectrogram above it stays visible with this species' bands lit, which
 * is the whole point of selecting one.
 */
export function OccurrenceSheet({ group, playing, onClose, onPlay }: Props) {
  return (
    <Sheet
      open={group !== null}
      onClose={onClose}
      title={group?.species.commonName ?? ''}
      aside={
        group && (
          <span className="shrink-0 text-sm tabular-nums text-ink-3">
            {group.count}× · max {group.bestScore.toFixed(2)}
          </span>
        )
      }
    >
      {group && (
        <>
          <p className="mb-3 select-text text-sm italic text-ink-3">
            {group.species.scientificName}
          </p>
          <ul className="grid grid-cols-3 gap-2" aria-label="Occurrences">
            {group.occurrences.map((occurrence) => {
              const key = occurrence.windowIndex * 10000 + occurrence.species.index
              const isPlaying = playing === key
              return (
                <li key={key}>
                  <button
                    type="button"
                    onClick={() => onPlay(occurrence)}
                    aria-pressed={isPlaying}
                    aria-label={
                      `${isPlaying ? 'Arrêter' : 'Écouter'} le segment à ` +
                      `${clock(occurrence.start)}, confiance ${occurrence.score.toFixed(2)}`
                    }
                    className={[
                      // A 44px minimum, both ways. The old chips were 24px tall
                      // and about 56 wide — fine with a mouse, a coin toss with
                      // a thumb.
                      'flex min-h-11 w-full flex-col items-center justify-center rounded-md',
                      'text-sm tabular-nums transition-colors duration-150',
                      isPlaying
                        ? 'bg-play text-on-play'
                        : 'bg-hover text-ink-2 hover:bg-line hover:text-ink',
                    ].join(' ')}
                  >
                    <span className="font-medium">{clock(occurrence.start)}</span>
                    {/* opacity-60 measured 2.04:1 on the coral; 80 clears it. */}
                    <span className="text-xs opacity-80">{occurrence.score.toFixed(2)}</span>
                  </button>
                </li>
              )
            })}
          </ul>
          <p className="mt-3 text-xs text-ink-3">
            Chaque segment dure 3 secondes — exactement la fenêtre que le modèle a notée.
          </p>
        </>
      )}
    </Sheet>
  )
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
