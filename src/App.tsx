import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useBirdNet } from './hooks/useBirdNet'
import { useGeoFilter } from './hooks/useGeoFilter'
import { useMicrophone } from './hooks/useMicrophone'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useSegmentPlayer } from './hooks/useSegmentPlayer'
import { DEFAULT_MIN_CONFIDENCE, WINDOW_SECONDS } from './lib/birdnet/constants'
import { describeWeek } from './lib/birdnet/geo'
import type { Species } from './lib/birdnet/labels'
import type { Detection } from './lib/birdnet/types'
import { AppShell } from './components/AppShell'
import { DropZone } from './components/DropZone'
import { GeoSheet } from './components/GeoSheet'
import { ListenView } from './components/ListenView'
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
  const { state, analyze, reset, analyzer, floor } = useBirdNet()
  const geo = useGeoFilter(analyzer)
  const [geoOpen, setGeoOpen] = useState(false)
  /** File or live. The bar's left slot was reserved for this from the start. */
  const [mode, setMode] = useState<'file' | 'live'>('file')
  const mic = useMicrophone(analyzer)
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

  const allGroups = useMemo<SpeciesGroup[]>(() => {
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

  /**
   * The geo filter partitions the species; it does not silently shorten the list.
   *
   * BirdNET's own semantics are a hard cut — `invalid_mask = res < min_confidence`
   * — so a filtered species really is out of the results, and the spectrogram and
   * the counts have to agree with that. What this app adds is that the removed
   * ones stay reachable, with the geo score that removed them. A filter that
   * makes a detection disappear without saying so is exactly what the rest of
   * this project refuses.
   */
  const { groups, maskedGroups } = useMemo(() => {
    const mask = geo.settings.enabled ? geo.mask : null
    if (!mask) return { groups: allGroups, maskedGroups: [] as SpeciesGroup[] }
    const kept: SpeciesGroup[] = []
    const removed: SpeciesGroup[] = []
    for (const group of allGroups) {
      (mask[group.species.index] ? kept : removed).push(group)
    }
    return { groups: kept, maskedGroups: removed }
  }, [allGroups, geo.mask, geo.settings.enabled])

  /**
   * The detections that survive both filters, which is what the picture and the
   * playback shortcut must agree with — a band lit over a species the geo filter
   * removed would contradict the list right next to it.
   */
  const keptDetections = useMemo(() => groups.flatMap((g) => g.occurrences), [groups])
  const visibleCount = keptDetections.length

  /** Live detections, grouped the same way the file ones are. */
  const liveGroups = useMemo<SpeciesGroup[]>(() => {
    const byIndex = new Map<number, SpeciesGroup>()
    for (const detection of mic.detections) {
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
    // Most recently heard first: live listening is about what is happening now.
    const result = [...byIndex.values()]
    for (const group of result) group.occurrences.sort((a, b) => b.start - a.start)
    result.sort((a, b) => b.occurrences[0].start - a.occurrences[0].start)
    return result
  }, [mic.detections])

  const shownGroups = mode === 'live' ? liveGroups : groups

  // A focused species the threshold has just filtered out would leave the
  // spectrogram highlighting nothing and the sheet describing nobody.
  useEffect(() => {
    if (pinned && !shownGroups.some((g) => g.species.index === pinned.index)) setPinned(null)
    if (hovered && !shownGroups.some((g) => g.species.index === hovered.index)) setHovered(null)
  }, [shownGroups, pinned, hovered])


  const openGroup = useMemo(
    () => (pinned ? (shownGroups.find((g) => g.species.index === pinned.index) ?? null) : null),
    [shownGroups, pinned],
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
    if (keptDetections.length === 0) return
    const at = player.positionRef.current ?? 0
    const pool = focused ? keptDetections.filter((d) => d.species.index === focused.index) : keptDetections
    if (pool.length === 0) return
    const under = pool.find((d) => at >= d.start && at < d.start + WINDOW_SECONDS)
    const best = pool.reduce((a, b) => (b.score > a.score ? b : a), pool[0])
    playDetection(under ?? best)
  }, [player, keptDetections, focused, playDetection])

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

  const switchMode = useCallback(
    (next: 'file' | 'live') => {
      if (next === mode) return
      // Leaving live listening must free the device, not merely hide the screen:
      // a track left running keeps the browser's recording indicator lit and
      // holds the microphone against other applications.
      if (mode === 'live') mic.stop()
      if (next === 'live') player.stop()
      setPinned(null)
      setHovered(null)
      setMode(next)
    },
    [mode, mic, player],
  )

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
          `${visibleCount} détection${visibleCount > 1 ? 's' : ''}`

  return (
    <>
      <AppShell
        header={
          mode === 'live' ? (
            <div>
              <h1 className="text-lg font-medium tracking-tight">Écoute en direct</h1>
              <p className="text-xs text-ink-3">
                Fenêtres de 3 s, une par seconde — rien n'est enregistré
              </p>
            </div>
          ) : file ? (
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
          busy && mode === 'file' ? (
            <ProgressPanel state={state} />
          ) : (
            // The left slot was reserved for this switch from the start, which
            // is why the bar did not need re-laying-out when live mode landed.
            <div className="flex items-center gap-3">
              <ModeSwitch mode={mode} onChange={switchMode} />
              {mode === 'file' ? (
                <button
                  type="button"
                  onClick={browse}
                  className="btn-primary flex h-11 shrink-0 items-center justify-center rounded-xl px-4 text-sm font-medium"
                >
                  {file ? 'Autre fichier' : 'Choisir un fichier'}
                </button>
              ) : (
                <span
                  role="status"
                  aria-live="polite"
                  className="min-w-0 flex-1 truncate text-right text-sm tabular-nums text-ink-3"
                >
                  {mic.phase === 'listening' ? `${mic.windows} fenêtre${mic.windows > 1 ? 's' : ''}` : ''}
                </span>
              )}
            </div>
          )
        }
      >
        {/* `flex-1` so the attribution below can be pushed to the bottom of the
            scroller instead of floating under a short drop zone. */}
        <div className="flex flex-1 flex-col gap-4 pt-2 pb-4">
          {mode === 'live' && (
            <>
              <ListenView
                phase={mic.phase}
                levelRef={mic.levelRef}
                elapsed={mic.elapsed}
                species={liveGroups.length}
                error={mic.error}
                revealKey={liveGroups.length}
                onStart={() => void mic.start()}
                onStop={mic.stop}
              />
              <SpeciesList
                groups={liveGroups}
                pinned={pinned}
                hovered={hovered}
                playingSpecies={null}
                onPin={setPinned}
                onHover={setHovered}
              />
            </>
          )}

          {mode === 'file' && !file && <DropZone onFile={handleFile} onBrowse={browse} />}

          {/* The picture arrives before the first window, so the analysis front
              advances across something already on screen. */}
          {mode === 'file' && state.spectrogram && (
            /* The only ambient light in the interface, and it sits behind the
               one object that earns it. */
            <div className="glass-card relative px-2 pt-1 pb-2">
              <div
                aria-hidden
                className="pointer-events-none absolute -inset-x-8 -inset-y-6 -z-10 rounded-2xl blur-3xl"
                style={{
                  background:
                    'radial-gradient(60% 70% at 50% 55%, var(--color-gold) 0%, transparent 70%)',
                  opacity: 0.16,
                }}
              />
              <Spectrogram
                spectrogram={state.spectrogram}
                detections={keptDetections}
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

          {mode === 'file' && state.phase === 'error' && (
            <div
              role="alert"
              className="rounded-lg border border-play/40 bg-play/10 p-4 text-sm text-ink"
            >
              <p className="font-medium">L'analyse a échoué</p>
              <p className="mt-1">{state.error}</p>
            </div>
          )}

          {mode === 'file' && state.phase === 'done' && (
            <>
              {state.detections.length === 0 && (
                <p className="text-sm text-ink-3">
                  Aucun oiseau détecté dans cet enregistrement.
                </p>
              )}

              {state.detections.length > 0 && (
                <>
                  <ThresholdSlider
                    value={threshold}
                    min={floor}
                    onChange={setThreshold}
                    total={state.detections.length}
                    visible={visibleCount}
                  />
                  <GeoRow
                    geo={geo}
                    masked={maskedGroups.length}
                    onOpen={() => setGeoOpen(true)}
                  />
                </>
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

              {maskedGroups.length > 0 && <MaskedSpecies groups={maskedGroups} scores={geo.scores} />}

              {/* Above the list it describes. It used to live in the action bar,
                  which the mode switch now shares — and a count belongs next to
                  the thing counted anyway. */}
              <p role="status" aria-live="polite" className="text-sm text-ink-3">
                {summary}
              </p>

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

      <GeoSheet
        open={geoOpen}
        onClose={() => setGeoOpen(false)}
        geo={geo}
        masked={maskedGroups.length}
      />

      <OccurrenceSheet
        group={openGroup}
        playing={player.playing}
        onClose={clearFocus}
        onPlay={playDetection}
        playable={mode === 'file'}
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

/** The filter's current setting, and the way in. Sits with the other filter. */
function GeoRow({
  geo,
  masked,
  onOpen,
}: {
  geo: ReturnType<typeof useGeoFilter>
  masked: number
  onOpen: () => void
}) {
  const { settings, loading } = geo
  const detail = !settings.enabled
    ? 'désactivé'
    : loading
      ? 'calcul en cours…'
      : `${settings.latitude.toFixed(2)}, ${settings.longitude.toFixed(2)} · ${describeWeek(settings.week)}`

  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex min-h-14 w-full items-center gap-3 rounded-xl border border-line px-3 text-left transition-colors duration-150 hover:border-line-strong hover:bg-hover"
    >
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium">Lieu et saison</span>
        <span className="block truncate text-xs text-ink-3">{detail}</span>
      </span>
      {masked > 0 && (
        <span className="shrink-0 rounded-full bg-gold/15 px-2 py-0.5 text-xs tabular-nums text-ink-2">
          −{masked}
        </span>
      )}
      <svg
        aria-hidden
        viewBox="0 0 24 24"
        className="h-4 w-4 shrink-0 text-ink-3"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="m9 18 6-6-6-6" />
      </svg>
    </button>
  )
}

/**
 * The species the geo filter removed, and why.
 *
 * A `<details>` rather than a state flag: the platform already has a disclosure
 * that is keyboard-operable and announced correctly, and `<summary>` is one of
 * the elements the space shortcut deliberately stands aside for.
 */
function MaskedSpecies({
  groups,
  scores,
}: {
  groups: SpeciesGroup[]
  scores: Float32Array | null
}) {
  return (
    <details className="rounded-xl border border-line px-3 py-2">
      <summary className="flex min-h-9 cursor-pointer list-none items-center text-sm text-ink-2 marker:content-['']">
        {groups.length} espèce{groups.length > 1 ? 's' : ''} masquée
        {groups.length > 1 ? 's' : ''} par le filtre géographique
      </summary>
      <ul className="mt-2 flex flex-col gap-1.5 border-t border-line pt-2">
        {groups.map((group) => (
          <li key={group.species.index} className="flex items-center gap-3 text-sm">
            <span className="min-w-0 flex-1 select-text">
              <span className="block truncate">{group.species.commonName}</span>
              <span className="block truncate text-xs italic text-ink-3">
                {group.species.scientificName}
              </span>
            </span>
            {/* The number that removed it, next to the number that found it. */}
            <span className="shrink-0 text-right text-xs tabular-nums text-ink-3">
              <span className="block">chant {group.bestScore.toFixed(2)}</span>
              <span className="block">lieu {scores ? scores[group.species.index].toFixed(3) : '—'}</span>
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}

/** Fichier / Micro. A segmented control, in the bar's long-reserved left slot. */
function ModeSwitch({
  mode,
  onChange,
}: {
  mode: 'file' | 'live'
  onChange: (mode: 'file' | 'live') => void
}) {
  return (
    <div
      role="tablist"
      aria-label="Source"
      // No border and no padding on the container: each of those steals from the
      // buttons inside, and the tap-target audit caught them at 38 px. The
      // buttons *are* the control, at a full 44.
      className="flex shrink-0 items-center rounded-xl bg-hover"
    >
      {(['file', 'live'] as const).map((value) => (
        <button
          key={value}
          type="button"
          role="tab"
          aria-selected={mode === value}
          onClick={() => onChange(value)}
          className={[
            'flex min-h-11 items-center rounded-xl px-3 text-sm transition-colors duration-150',
            mode === value
              ? 'bg-raised font-medium text-ink shadow-sm'
              : 'text-ink-3 hover:text-ink',
          ].join(' ')}
        >
          {value === 'file' ? 'Fichier' : 'Micro'}
        </button>
      ))}
    </div>
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
