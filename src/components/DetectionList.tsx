import type { Detection } from '../lib/birdnet/types'

interface Props {
  detections: Detection[]
  playing: number | null
  onPlay: (start: number, key: number) => void
}

export function DetectionList({ detections, playing, onPlay }: Props) {
  if (detections.length === 0) return null

  return (
    <ul className="flex flex-col divide-y divide-neutral-200 dark:divide-neutral-800">
      {detections.map((detection) => {
        // Window index and class index together identify a detection uniquely;
        // the same species can appear in many windows.
        const key = detection.windowIndex * 10000 + detection.species.index
        const isPlaying = playing === key

        return (
          <li key={key}>
            <button
              type="button"
              onClick={() => onPlay(detection.start, key)}
              // A toggle, so it reports its state rather than just its label —
              // screen readers announce the change, and it gives the smoke test
              // something a user can actually perceive to assert on.
              aria-pressed={isPlaying}
              aria-label={`${isPlaying ? 'Arrêter' : 'Écouter'} ${detection.species.commonName} à ${formatRange(detection.start, detection.end)}`}
              className="flex w-full items-center gap-4 py-3 text-left transition-colors hover:bg-neutral-50 dark:hover:bg-neutral-900"
            >
              <span
                className={[
                  'flex size-8 shrink-0 items-center justify-center rounded-full border text-xs',
                  isPlaying
                    ? 'border-neutral-900 bg-neutral-900 text-white dark:border-neutral-100 dark:bg-neutral-100 dark:text-neutral-900'
                    : 'border-neutral-300 text-neutral-500 dark:border-neutral-700',
                ].join(' ')}
                aria-hidden
              >
                {/* SVG rather than ▶ / ❙❙: those code points render as colour
                    emoji on several platforms, which is both off-palette and
                    inconsistent between machines. */}
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </span>

              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{detection.species.commonName}</span>
                <span className="block truncate text-sm italic text-neutral-500">
                  {detection.species.scientificName}
                </span>
              </span>

              <span className="shrink-0 tabular-nums text-sm text-neutral-500">
                {formatRange(detection.start, detection.end)}
              </span>

              <span className="w-24 shrink-0">
                <ScoreBar score={detection.score} />
              </span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function ScoreBar({ score }: { score: number }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-1 flex-1 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
        <span
          className="block h-full bg-neutral-900 dark:bg-neutral-100"
          style={{ width: `${Math.round(score * 100)}%` }}
        />
      </span>
      <span className="w-8 shrink-0 text-right text-xs tabular-nums text-neutral-500">
        {score.toFixed(2)}
      </span>
    </span>
  )
}

function formatRange(start: number, end: number): string {
  return `${clock(start)}–${clock(end)}`
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 12 12" className="size-3 fill-current" role="presentation">
      <path d="M3 1.5 10 6 3 10.5z" />
    </svg>
  )
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 12 12" className="size-3 fill-current" role="presentation">
      <path d="M3 2h2.2v8H3zM6.8 2H9v8H6.8z" />
    </svg>
  )
}
