import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useBirdNet } from './hooks/useBirdNet'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useSegmentPlayer } from './hooks/useSegmentPlayer'
import { DEFAULT_MIN_CONFIDENCE, WINDOW_SECONDS } from './lib/birdnet/constants'
import type { Species } from './lib/birdnet/labels'
import type { Detection } from './lib/birdnet/types'
import { AppShell } from './components/AppShell'
import { DropZone } from './components/DropZone'
import { OccurrenceSheet } from './components/OccurrenceSheet'
import { ProgressPanel } from './components/ProgressPanel'
import { Spectrogram } from './components/Spectrogram'
import { SpeciesList, type SpeciesGroup } from './components/SpeciesList'
import { ThresholdSlider } from './components/ThresholdSlider'
import { Attribution } from './components/Attribution'
import { UpdatePill } from './components/UpdatePill'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(DEFAULT_MIN_CONFIDENCE)
  // Selected by a tap, hovered by the pointer. Separate slots — sharing one is
  // what made hover latch and a tap deselect the row under the cursor.
  const [pinned, setPinned] = useState<Species | null>(null)
  const [hovered, setHovered] = useState<Species | null>(null)
  const focused = pinned ?? hovered
  const { state, analyze, reset, floor } = useBirdNet()
  // One file input for the whole app, so the drop zone and the action bar open
  // the same picker and the element exists in every phase.
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  const browse = useCallback(() => fileInputRef.current?.click(), [])

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
  // spectrogram highlighting nothing and the sheet describing nobody.
  useEffect(() => {
    if (pinned && !groups.some((g) => g.species.index === pinned.index)) setPinned(null)
    if (hovered && !groups.some((g) => g.species.index === hovered.index)) setHovered(null)
  }, [groups, pinned, hovered])

  const openGroup = useMemo(
    () => (pinned ? (groups.find((g) => g.species.index === pinned.index) ?? null) : null),
    [groups, pinned],
  )

  const playDetection = useCallback(
    (detection: Detection) => {
      player.play(detection.start, detection.windowIndex * 10000 + detection.species.index)
    },
    [player],
  )

  const busy =
    state.phase === 'loading-model' || state.phase === 'decoding' || state.phase === 'analyzing'

  // The playback key packs a window and a class index into one number; the list
  // only needs the class back out of it.
  const playingSpecies = player.playing === null ? null : player.playing % 10000

  // Space plays what the playhead is sitting on, preferring the selected
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
    if (pool.length === 0) return
    const under = pool.find((d) => at >= d.start && at < d.start + WINDOW_SECONDS)
    const best = pool.reduce((a, b) => (b.score > a.score ? b : a), pool[0])
    playDetection(under ?? best)
  }, [player, visible, focused, playDetection])

  const seekBy = useCallback(
    (delta: number) => {
      const at = player.positionRef.current ?? 0
      player.seek(Math.max(0, Math.min(state.duration, at + delta)))
    },
    [player, state.duration],
  )

  const focusList = useCallback(() => {
    document.querySelector<HTMLButtonElement>('#species-list button')?.focus()
  }, [])

  const clearFocus = useCallback(() => setPinned(null), [])

  useKeyboardShortcuts({
    onTogglePlay: togglePlay,
    onSeek: seekBy,
    onFocusList: focusList,
    onEscape: clearFocus,
  })

  const summary =
    state.detections.length === 0
      ? 'Aucun oiseau détecté'
      : groups.length === 0
        ? 'Aucune détection à ce seuil'
        : `${groups.length} espèce${groups.length > 1 ? 's' : ''} · ` +
          `${visible.length} détection${visible.length > 1 ? 's' : ''}`

  return (
    <>
      <AppShell
        header={
          file ? (
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{file.name}</p>
                {state.duration > 0 && (
                  <p className="text-xs tabular-nums text-ink-3">
                    {formatDuration(state.duration)}
                    {state.phase === 'done' && ` · ${state.medianInferenceMs.toFixed(0)} ms/fenêtre`}
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={handleReset}
                aria-label="Recommencer avec un autre fichier"
                className="-mr-2 flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-ink-3 transition-colors duration-150 hover:bg-hover hover:text-ink"
              >
                <CloseIcon />
              </button>
            </div>
          ) : (
            <div>
              <h1 className="text-lg font-medium tracking-tight">whosyourbirdy</h1>
              <p className="text-xs text-ink-3">
                Identification d'oiseaux au chant, entièrement dans votre navigateur
              </p>
            </div>
          )
        }
        bar={
          busy ? (
            <ProgressPanel state={state} />
          ) : (
            /*
             * Two slots, and only one is filled today. The left one is where the
             * Fichier / Micro switch goes when live listening lands — reserving
             * it now is why the bar will not need re-laying-out then.
             */
            <div className="flex items-center gap-3">
              <span role="status" aria-live="polite" className="min-w-0 flex-1 truncate text-sm text-ink-3">
                {state.phase === 'done' ? summary : ''}
              </span>
              <button
                type="button"
                onClick={browse}
                className="flex h-11 shrink-0 items-center justify-center rounded-lg bg-play px-4 text-sm font-medium text-on-play transition-opacity duration-150 hover:opacity-90"
              >
                {file ? 'Autre fichier' : 'Choisir un fichier'}
              </button>
            </div>
          )
        }
      >
        {/* `min-h-full` so the attribution below can be pushed to the bottom of
            the scroller instead of floating under a short drop zone. */}
        <div className="flex min-h-full flex-col gap-4 pt-2 pb-4">
          {!file && <DropZone onFile={handleFile} onBrowse={browse} />}

          {/* The picture arrives before the first window, so the analysis front
              advances across something already on screen. */}
          {state.spectrogram && (
            /* The only ambient light in the interface, and it sits behind the
               one object that earns it. */
            <div className="relative">
              <div
                aria-hidden
                className="pointer-events-none absolute -inset-x-6 -inset-y-4 -z-10 rounded-lg blur-2xl"
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
              {state.detections.length === 0 && (
                <p className="text-sm text-ink-3">
                  Aucun oiseau détecté dans cet enregistrement.
                </p>
              )}

              {state.detections.length > 0 && (
                <ThresholdSlider
                  value={threshold}
                  min={floor}
                  onChange={setThreshold}
                  total={state.detections.length}
                  visible={visible.length}
                />
              )}

              {groups.length === 0 && state.detections.length > 0 && (
                <p className="text-sm text-ink-3">
                  Aucune détection à ce seuil — abaissez le curseur.
                </p>
              )}

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
                playingSpecies={playingSpecies}
                onPin={setPinned}
                onHover={setHovered}
              />
            </>
          )}

          <div className="mt-auto pt-2">
            <Attribution />
          </div>
        </div>
      </AppShell>

      <OccurrenceSheet
        group={openGroup}
        playing={player.playing}
        onClose={clearFocus}
        onPlay={playDetection}
      />

      <UpdatePill />

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
        className="hidden"
        onChange={(e) => {
          const next = e.target.files?.[0]
          if (next) handleFile(next)
          // Allow re-selecting the same file after a reset.
          e.target.value = ''
        }}
      />
    </>
  )
}

function CloseIcon() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    >
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  )
}

// Floors, to match the row timecodes; rounding here made a 1.5 s file read
// "0:02" above a row labelled "0:00–0:03".
function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
