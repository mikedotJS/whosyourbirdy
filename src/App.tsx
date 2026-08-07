import { useCallback, useEffect, useMemo, useState } from 'react'
import { useBirdNet } from './hooks/useBirdNet'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useSegmentPlayer } from './hooks/useSegmentPlayer'
import { DEFAULT_MIN_CONFIDENCE, WINDOW_SECONDS } from './lib/birdnet/constants'
import type { Species } from './lib/birdnet/labels'
import type { Detection } from './lib/birdnet/types'
import { DropZone } from './components/DropZone'
import { ProgressPanel } from './components/ProgressPanel'
import { Spectrogram } from './components/Spectrogram'
import { SpeciesList, type SpeciesGroup } from './components/SpeciesList'
import { ThresholdSlider } from './components/ThresholdSlider'
import { Attribution } from './components/Attribution'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(DEFAULT_MIN_CONFIDENCE)
  // Pinned by a click, hovered by the pointer. Separate slots — sharing one is
  // what made hover latch and a click deselect the row under the cursor.
  const [pinned, setPinned] = useState<Species | null>(null)
  const [hovered, setHovered] = useState<Species | null>(null)
  const focused = pinned ?? hovered
  const { state, analyze, reset, floor } = useBirdNet()

  // Only wire the player once the file has actually been decoded. Creating an
  // <audio> for every dropped file made an undecodable or 0-byte one emit a
  // console error for an element nothing would ever play.
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
    setPinned(null)
    setHovered(null)
    setFile(next)
    void analyze(next)
  }

  const handleReset = () => {
    player.stop()
    setPinned(null)
    setHovered(null)
    setFile(null)
    reset()
  }

  // The analysis ran at `floor`; the slider only filters what is already here,
  // which is why moving it is instant.
  const visible = useMemo(
    () => state.detections.filter((d) => d.score >= threshold),
    [state.detections, threshold],
  )

  const groups = useMemo<SpeciesGroup[]>(() => {
    const byIndex = new Map<number, SpeciesGroup>()
    for (const detection of visible) {
      const existing = byIndex.get(detection.species.index)
      if (existing) {
        existing.count++
        existing.bestScore = Math.max(existing.bestScore, detection.score)
        existing.occurrences.push(detection)
      } else {
        byIndex.set(detection.species.index, {
          species: detection.species,
          count: 1,
          bestScore: detection.score,
          occurrences: [detection],
        })
      }
    }
    const result = [...byIndex.values()]
    result.sort((a, b) => b.bestScore - a.bestScore)
    for (const group of result) group.occurrences.sort((a, b) => a.start - b.start)
    return result
  }, [visible])

  // A focused species the threshold has just filtered out would leave the
  // spectrogram highlighting nothing.
  useEffect(() => {
    if (pinned && !groups.some((g) => g.species.index === pinned.index)) setPinned(null)
    if (hovered && !groups.some((g) => g.species.index === hovered.index)) setHovered(null)
  }, [groups, pinned, hovered])

  const playDetection = useCallback(
    (detection: Detection) => {
      player.play(detection.start, detection.windowIndex * 10000 + detection.species.index)
    },
    [player],
  )

  const busy =
    state.phase === 'loading-model' || state.phase === 'decoding' || state.phase === 'analyzing'

  // Space plays what the playhead is sitting on, preferring the focused
  // species' own occurrences — pressing it with a species selected should not
  // start some other bird. Failing that, the best detection in the file, so the
  // shortcut always does something on a fresh result.
  const togglePlay = useCallback(() => {
    if (player.playing !== null) {
      player.stop()
      return
    }
    if (visible.length === 0) return
    const at = player.positionRef.current ?? 0
    const pool = focused ? visible.filter((d) => d.species.index === focused.index) : visible
    const under = pool.find((d) => at >= d.start && at < d.start + WINDOW_SECONDS)
    const best = pool.reduce((a, b) => (b.score > a.score ? b : a), pool[0])
    const target = under ?? best
    if (target) playDetection(target)
  }, [player, visible, focused, playDetection])

  const seekBy = useCallback(
    (delta: number) => {
      const at = player.positionRef.current ?? 0
      player.seek(Math.max(0, Math.min(state.duration, at + delta)))
    },
    [player, state.duration],
  )

  const focusList = useCallback(() => {
    const first = document.querySelector<HTMLButtonElement>('#species-list button')
    first?.focus()
  }, [])

  const clearFocus = useCallback(() => setPinned(null), [])

  useKeyboardShortcuts({
    onTogglePlay: togglePlay,
    onSeek: seekBy,
    onFocusList: focusList,
    onEscape: clearFocus,
  })

  return (
    <div className="grain relative min-h-dvh bg-surface text-ink">
      <div className="relative z-10 mx-auto flex min-h-dvh max-w-4xl flex-col gap-6 px-6 py-10">
        <header className="flex items-baseline justify-between gap-4">
          <div>
            <h1 className="text-xl font-medium tracking-tight">whosyourbirdy</h1>
            <p className="mt-1 text-sm text-ink-3">
              Identification d'oiseaux au chant. Tout se passe dans votre navigateur — aucun fichier
              n'est envoyé.
            </p>
          </div>
          {file && (
            <button
              type="button"
              onClick={handleReset}
              className="shrink-0 rounded-md border border-line px-3 py-1.5 text-sm text-ink-2 transition-colors duration-150 hover:border-line-strong hover:bg-hover"
            >
              Autre fichier
            </button>
          )}
        </header>

        <main className="flex flex-1 flex-col gap-5">
          {!file && <DropZone onFile={handleFile} />}

          {file && (
            <div className="flex items-baseline gap-3 text-sm">
              <span className="truncate font-medium">{file.name}</span>
              {state.duration > 0 && (
                <span className="shrink-0 tabular-nums text-ink-3">
                  {formatDuration(state.duration)}
                </span>
              )}
            </div>
          )}

          {/* The picture arrives before the first window, so the analysis front
              advances across something already on screen. */}
          {state.spectrogram && (
            /* The only ambient light in the interface, and it sits behind the
               one object that earns it. */
            <div className="relative">
              <div
                aria-hidden
                className="pointer-events-none absolute -inset-x-6 -inset-y-4 -z-10 rounded-lg opacity-60 blur-2xl"
                style={{
                  background:
                    'radial-gradient(60% 70% at 50% 55%, var(--color-gold) 0%, transparent 70%)',
                  opacity: 0.09,
                }}
              />
            <Spectrogram
              spectrogram={state.spectrogram}
              detections={visible}
              focused={focused}
              analysedUntil={state.analysedUntil}
              duration={state.duration}
              positionRef={player.positionRef}
              positionVersion={player.positionVersion}
              isPlaying={player.playing !== null}
              onScrub={player.seek}
            />
            </div>
          )}

          {busy && <ProgressPanel state={state} />}

          {state.phase === 'error' && (
            <div
              role="alert"
              className="rounded-lg border border-play/40 bg-play/10 p-4 text-sm text-ink"
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

              <div className="flex items-baseline justify-between text-sm text-ink-3">
                <span role="status" aria-live="polite">
                  {state.detections.length === 0
                    ? 'Aucun oiseau détecté dans cet enregistrement'
                    : groups.length === 0
                      ? 'Aucune détection à ce seuil — abaissez le curseur'
                      : `${groups.length} espèce${groups.length > 1 ? 's' : ''} · ` +
                        `${visible.length} détection${visible.length > 1 ? 's' : ''}`}
                </span>
                <span className="tabular-nums">{state.medianInferenceMs.toFixed(0)} ms/fenêtre</span>
              </div>

              {state.truncatedWindows > 0 && (
                <p className="rounded-md bg-gold/10 px-3 py-2 text-xs text-ink-2">
                  {state.truncatedWindows} fenêtre(s) ont produit plus de détections que la limite
                  par fenêtre : la liste est incomplète à ce seuil.
                </p>
              )}

              <SpeciesList
                groups={groups}
                pinned={pinned}
                hovered={hovered}
                playing={player.playing}
                onPin={setPinned}
                onHover={setHovered}
                onPlay={playDetection}
              />
            </>
          )}
        </main>

        <Attribution />
      </div>
    </div>
  )
}

// Floors, to match the row timecodes; rounding here made a 1.5 s file read
// "0:02" above a row labelled "0:00–0:03".
function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
