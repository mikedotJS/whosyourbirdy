import { useEffect, useMemo, useState } from 'react'
import { useBirdNet } from './hooks/useBirdNet'
import { useSegmentPlayer } from './hooks/useSegmentPlayer'
import { DEFAULT_MIN_CONFIDENCE } from './lib/birdnet/constants'
import { DropZone } from './components/DropZone'
import { ProgressPanel } from './components/ProgressPanel'
import { DetectionList } from './components/DetectionList'
import { ThresholdSlider } from './components/ThresholdSlider'
import { Attribution } from './components/Attribution'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(DEFAULT_MIN_CONFIDENCE)
  const { state, analyze, reset, floor } = useBirdNet()
  // Only wire the player once the file has actually been decoded. Creating an
  // <audio> for every dropped file made an undecodable or 0-byte one emit a
  // console error (ERR_REQUEST_RANGE_NOT_SATISFIABLE) for a element nothing
  // would ever play.
  const player = useSegmentPlayer(state.phase === 'done' ? file : null)

  // Without this, dropping a file anywhere outside the drop zone — including
  // over the results, where no drop zone is rendered at all — makes the browser
  // navigate to it and throws the session away.
  useEffect(() => {
    const swallow = (event: DragEvent) => event.preventDefault()
    window.addEventListener('dragover', swallow)
    window.addEventListener('drop', swallow)
    return () => {
      window.removeEventListener('dragover', swallow)
      window.removeEventListener('drop', swallow)
    }
  }, [])

  const handleFile = (next: File) => {
    player.stop()
    setFile(next)
    void analyze(next)
  }

  const handleReset = () => {
    player.stop()
    setFile(null)
    reset()
  }

  // The analysis ran at `floor`; the slider only filters what is already here,
  // which is why moving it is instant.
  const visible = useMemo(
    () => state.detections.filter((d) => d.score >= threshold),
    [state.detections, threshold],
  )

  const speciesCount = useMemo(
    () => new Set(visible.map((d) => d.species.index)).size,
    [visible],
  )

  const busy = state.phase === 'loading-model' || state.phase === 'decoding' || state.phase === 'analyzing'

  return (
    <div className="min-h-dvh bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <div className="mx-auto flex min-h-dvh max-w-3xl flex-col gap-8 px-6 py-10">
        <header className="flex items-baseline justify-between gap-4">
          <div>
            <h1 className="text-xl font-medium tracking-tight">whosyourbirdy</h1>
            <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400">
              Identification d'oiseaux au chant. Tout se passe dans votre navigateur — aucun fichier
              n'est envoyé.
            </p>
          </div>
          {file && (
            <button
              type="button"
              onClick={handleReset}
              className="shrink-0 rounded-md border border-neutral-300 px-3 py-1.5 text-sm text-neutral-600 transition-colors hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-400 dark:hover:bg-neutral-900"
            >
              Autre fichier
            </button>
          )}
        </header>

        <main className="flex flex-1 flex-col gap-6">
          {!file && <DropZone onFile={handleFile} />}

          {file && (
            <div className="flex items-baseline gap-3 text-sm">
              <span className="truncate font-medium">{file.name}</span>
              {state.duration > 0 && (
                <span className="shrink-0 text-neutral-500 dark:text-neutral-400">
                  {formatDuration(state.duration)}
                </span>
              )}
            </div>
          )}

          {busy && <ProgressPanel state={state} />}

          {state.phase === 'error' && (
            <div
              role="alert"
              className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300"
            >
              <p className="font-medium">L'analyse a échoué</p>
              <p className="mt-1">{state.error}</p>
            </div>
          )}

          {state.phase === 'done' && (
            <>
              {state.detections.length > 0 && (
                <ThresholdSlider
                  value={threshold}
                  min={floor}
                  onChange={setThreshold}
                  total={state.detections.length}
                  visible={visible.length}
                />
              )}

              <div className="flex items-baseline justify-between text-sm text-neutral-500 dark:text-neutral-400">
                <span role="status" aria-live="polite">
                  {state.detections.length === 0
                    ? 'Aucun oiseau détecté dans cet enregistrement'
                    : visible.length === 0
                      ? 'Aucune détection à ce seuil — abaissez le curseur'
                      : `${visible.length} détection${visible.length > 1 ? 's' : ''} · ` +
                        `${speciesCount} espèce${speciesCount > 1 ? 's' : ''}`}
                </span>
                <span className="tabular-nums">
                  {state.medianInferenceMs.toFixed(0)} ms/fenêtre
                </span>
              </div>

              {state.truncatedWindows > 0 && (
                <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
                  {state.truncatedWindows} fenêtre(s) ont produit plus de détections que la limite
                  par fenêtre : la liste est incomplète à ce seuil.
                </p>
              )}

              <DetectionList
                detections={visible}
                playing={player.playing}
                onPlay={player.play}
              />
            </>
          )}
        </main>

        <Attribution />
      </div>
    </div>
  )
}

// Floors, to match the row timecodes in DetectionList; rounding here made a
// 1.5 s file read "0:02" above a row labelled "0:00–0:03".
function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
