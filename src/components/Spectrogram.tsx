import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SpectrogramData } from '../lib/birdnet/spectrogram'
import type { Detection } from '../lib/birdnet/types'
import type { Species } from '../lib/birdnet/labels'

interface Props {
  spectrogram: SpectrogramData | null
  detections: Detection[]
  /** The species whose bands are promoted; everything else recedes. */
  focused: Species | null
  /** End of the analysed region, in seconds. Drives the analysis front. */
  analysedUntil: number
  duration: number
  /**
   * Live playback/scrub position, read every frame. A ref rather than a prop
   * value so playback does not re-render the tree 60 times a second.
   */
  positionRef: React.RefObject<number | null>
  /**
   * Changes once per discrete seek. The ref above is invisible to React, so
   * without this a keyboard scrub moved the audio and left the playhead painted
   * where it was. Only in the dependency list — its value is never read.
   */
  positionVersion: number
  /** True only while sound is actually playing — drives the animation loop. */
  isPlaying: boolean
  onScrub: (seconds: number) => void
}

/**
 * The spectrogram, and the object the rest of the interface orbits.
 *
 * Colour decisions, in short (the long version is in the README):
 *
 *  - The spectrogram is a *sequential* encoding of magnitude, so it gets one
 *    monotone ramp — night to gold. One hue family, moving in one direction, is
 *    what makes it readable without a legend; a rainbow would not be.
 *  - Species identity is NOT carried by colour. There are 6522 possible classes;
 *    no palette survives that, and cycling hues would make two species share one.
 *    Identity lives in the list, and the link between list and picture is a
 *    single focused colour at a time.
 *  - Detection bands therefore have three states: recessive (present, not the
 *    focus), focused (cool blue), and playing (coral). Both accents are checked
 *    against the ramp, not only against the surface — an overlay that melts into
 *    the picture's own bright end is not an overlay.
 */

const AXIS_WIDTH = 34
const AXIS_HEIGHT = 18
/**
 * The detection bands get their own lane under the plot rather than sitting on
 * top of it. Overlaid, they covered the low-frequency end of the very signal
 * they describe, and there was no room to read them.
 */
const BAND_HEIGHT = 16

/**
 * Palette « Aube ». Validated as a categorical pair against each surface with
 * the dataviz checker: ΔE CVD 22.2 dark / 25.2 light, normal-vision 29.4 / 31.5,
 * inside the lightness band, ≥3:1 on their surface.
 *
 * The focus accent is deliberately COOL. The spectrogram ramp climbs to gold, so
 * a warm focus accent would sit on top of the picture's own brightest values and
 * disappear into them.
 */
const COLORS = {
  light: {
    focus: '#2a6fd6',
    playing: '#c9492f',
    // Solid, not translucent: this colour is drawn under a globalAlpha, and two
    // alphas multiplied down to 0.16 — a 1.6:1 contrast for the default state of
    // every detection, where WCAG 1.4.11 wants 3:1. Opacity lives in one place.
    band: '#14131a',
    ink: '#14131a',
    muted: '#6b6878',
    grid: 'rgba(251, 250, 246, 0.30)',
    unanalysed: 'rgba(251, 250, 246, 0.58)',
  },
  dark: {
    focus: '#3e8fdb',
    playing: '#e05c42',
    band: '#f4f2ec',
    ink: '#f4f2ec',
    muted: '#9a98a8',
    grid: 'rgba(244, 242, 236, 0.16)',
    unanalysed: 'rgba(10, 11, 20, 0.62)',
  },
}

/**
 * The spectrogram's sequential ramp, « nuit → or ».
 *
 * One hue family throughout — night, ember, amber, pale gold — so it stays a
 * legitimate sequential encoding rather than a rainbow. Light mode runs the same
 * hue the other way: paper down to deep bronze, because there high magnitude has
 * to be *dark* to read on a light surface.
 *
 * 256 entries built once per theme; the per-pixel loop then costs one lookup.
 */
const RAMP_STOPS = {
  dark: [
    [0.0, 0x0a, 0x0b, 0x14],
    [0.28, 0x3a, 0x22, 0x2c],
    [0.55, 0x8c, 0x4d, 0x2e],
    [0.8, 0xef, 0xc3, 0x73],
    [1.0, 0xff, 0xf1, 0xd2],
  ],
  light: [
    [0.0, 0xfb, 0xfa, 0xf6],
    [0.25, 0xe8, 0xd8, 0xb4],
    [0.55, 0xc0, 0x8a, 0x3c],
    [0.8, 0x7a, 0x4c, 0x14],
    [1.0, 0x2e, 0x1d, 0x08],
  ],
} as const

function buildRamp(theme: 'light' | 'dark'): Uint8Array {
  const stops = RAMP_STOPS[theme]
  const ramp = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let k = 0
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++
    const [t0, r0, g0, b0] = stops[k]
    const [t1, r1, g1, b1] = stops[k + 1]
    const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0)
    ramp[i * 3] = Math.round(r0 + (r1 - r0) * f)
    ramp[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f)
    ramp[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f)
  }
  return ramp
}

/** Honour the OS setting: every animation below collapses to its end state. */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const query = matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(query.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])
  return reduced
}

/** Ease-out cubic. Fast start, settled finish — reads as arriving, not sliding. */
function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3)
}

/** How long a newly-arrived detection band takes to appear. */
const BAND_IN_MS = 320
function useTheme(): 'light' | 'dark' {
  const [dark, setDark] = useState(
    () => typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches,
  )
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const query = matchMedia('(prefers-color-scheme: dark)')
    const update = () => setDark(query.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])
  return dark ? 'dark' : 'light'
}

/**
 * Paint the magnitude grid once into an offscreen canvas at its native
 * resolution; the visible canvas then just scales it. Redrawing 1600×320 cells
 * on every scrub or hover would make the whole thing crawl.
 */
function useSpectrogramBitmap(
  spectrogram: SpectrogramData | null,
  theme: 'light' | 'dark',
): HTMLCanvasElement | null {
  return useMemo(() => {
    if (!spectrogram) return null
    const { columns, bins, magnitudes } = spectrogram

    const offscreen = document.createElement('canvas')
    offscreen.width = columns
    offscreen.height = bins
    const ctx = offscreen.getContext('2d')
    if (!ctx) return null

    const image = ctx.createImageData(columns, bins)
    const ramp = buildRamp(theme)

    for (let c = 0; c < columns; c++) {
      for (let b = 0; b < bins; b++) {
        // Flip the frequency axis: bin 0 is DC, and low frequencies belong at
        // the bottom of the picture.
        const y = bins - 1 - b
        const target = (y * columns + c) * 4
        const v = magnitudes[c * bins + b] * 3

        image.data[target] = ramp[v]
        image.data[target + 1] = ramp[v + 1]
        image.data[target + 2] = ramp[v + 2]
        image.data[target + 3] = 255
      }
    }
    ctx.putImageData(image, 0, 0)
    return offscreen
  }, [spectrogram, theme])
}

export function Spectrogram({
  spectrogram,
  detections,
  focused,
  analysedUntil,
  duration,
  positionRef,
  positionVersion,
  isPlaying,
  onScrub,
}: Props) {
  const theme = useTheme()
  const palette = COLORS[theme]
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const bitmap = useSpectrogramBitmap(spectrogram, theme)
  const [size, setSize] = useState({ width: 0, height: 288 })
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const reduced = useReducedMotion()
  /**
   * Full-viewport mode.
   *
   * A 420px-wide column is a deliberate shape for the app and a poor one for a
   * two-minute spectrogram: at that width one pixel is nearly a second. Rather
   * than argue with the shape, the picture gets to leave it.
   *
   * Implemented by moving the existing element, not by mounting a second one —
   * the `ResizeObserver` above already redraws at whatever size it lands in, so
   * there is one canvas, one animation loop, and no state to reconcile.
   */
  const [expanded, setExpanded] = useState(false)

  // Escape leaves full-viewport mode. `preventDefault` matters: the global
  // shortcut handler treats a defaulted-prevented event as already dealt with,
  // so this closes the picture instead of also clearing the species selection.
  useEffect(() => {
    if (!expanded) return
    const handle = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      setExpanded(false)
    }
    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [expanded])

  /**
   * Animation state, deliberately in refs rather than React state.
   *
   * These change every frame; routing them through setState would re-render the
   * whole tree 60 times a second to repaint one canvas.
   */
  const frontRef = useRef(0)              // the eased analysis front, in seconds
  const seenRef = useRef(new Map<number, number>())  // detection key -> first seen (ms)
  const focusRef = useRef(0)              // 0..1 cross-fade of the focused state
  const rafRef = useRef<number | null>(null)
  const drawRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    const element = wrapRef.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const plot = useMemo(
    () => ({
      x: AXIS_WIDTH,
      y: 0,
      width: Math.max(0, size.width - AXIS_WIDTH),
      height: Math.max(0, size.height - AXIS_HEIGHT - BAND_HEIGHT),
    }),
    [size],
  )

  const timeToX = useCallback(
    (seconds: number) => plot.x + (duration > 0 ? (seconds / duration) * plot.width : 0),
    [plot, duration],
  )
  const xToTime = useCallback(
    (x: number) => (plot.width > 0 ? ((x - plot.x) / plot.width) * duration : 0),
    [plot, duration],
  )

  // Record when each detection first appeared, so bands can arrive individually
  // instead of the whole set fading in at once.
  useEffect(() => {
    const now = performance.now()
    const seen = seenRef.current
    for (const detection of detections) {
      const key = detection.windowIndex * 10000 + detection.species.index
      if (!seen.has(key)) seen.set(key, reduced ? 0 : now)
    }
  }, [detections, reduced])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || plot.width <= 0) return

    const now = performance.now()

    // The analysis front eases toward the true position instead of jumping a
    // window at a time. This is the moment the user is waiting, so the front
    // advancing is the whole feedback — a 3-second hop every ~100 ms reads as
    // stuttering, a glide reads as progress.
    if (reduced) {
      frontRef.current = analysedUntil
    } else {
      const gap = analysedUntil - frontRef.current
      frontRef.current += gap * 0.12
      if (Math.abs(gap) < 0.02) frontRef.current = analysedUntil
    }
    const front = frontRef.current

    // Focus cross-fades rather than snapping, so the link between the list and
    // the picture is a state change you can follow.
    const playhead = positionRef.current
    const focusTarget = focused ? 1 : 0
    if (reduced) focusRef.current = focusTarget
    else focusRef.current += (focusTarget - focusRef.current) * 0.25

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size.width * dpr
    canvas.height = size.height * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.width, size.height)

    // ---- the spectrogram itself ------------------------------------------
    if (bitmap) {
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(bitmap, plot.x, plot.y, plot.width, plot.height)
    }

    // ---- frequency guides, recessive -------------------------------------
    const maxHz = spectrogram?.maxHz ?? 15_000
    ctx.strokeStyle = palette.grid
    ctx.fillStyle = palette.muted
    ctx.lineWidth = 1
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let hz = 5000; hz < maxHz; hz += 5000) {
      const y = plot.y + plot.height * (1 - hz / maxHz)
      ctx.beginPath()
      ctx.moveTo(plot.x, Math.round(y) + 0.5)
      ctx.lineTo(plot.x + plot.width, Math.round(y) + 0.5)
      ctx.stroke()
      ctx.fillText(`${hz / 1000}k`, plot.x - 6, y)
    }

    // ---- the region not yet analysed --------------------------------------
    // A veil rather than emptiness: the recording is all there from the start,
    // and what advances is knowledge about it, not the picture.
    // Never let the eased front fall behind a detection that has already
    // arrived: the veil says "not analysed yet", and a band inside it would be
    // a picture contradicting itself.
    let lastKnown = front
    for (const detection of detections) {
      if (detection.end > lastKnown) lastKnown = Math.min(detection.end, analysedUntil)
    }
    const veilFrom = Math.max(front, lastKnown)

    if (veilFrom < duration - 0.01 && bitmap) {
      const frontX = timeToX(veilFrom)
      ctx.fillStyle = palette.unanalysed
      ctx.fillRect(frontX, plot.y, plot.x + plot.width - frontX, plot.height)
      ctx.strokeStyle = palette.focus
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(Math.round(frontX) + 0.5, plot.y)
      ctx.lineTo(Math.round(frontX) + 0.5, plot.y + plot.height)
      ctx.stroke()
    }

    // ---- detection bands ---------------------------------------------------
    // Their own lane beneath the plot, so they never obscure the signal they
    // describe. Focused species drawn last, so they sit above the rest.
    const bandTop = plot.y + plot.height + 3
    const ordered = [...detections].sort((a, b) => {
      const af = focused && a.species.index === focused.index ? 1 : 0
      const bf = focused && b.species.index === focused.index ? 1 : 0
      return af - bf
    })

    for (const detection of ordered) {
      const key = detection.windowIndex * 10000 + detection.species.index
      const isFocused = focused !== null && detection.species.index === focused.index
      const underPlayhead =
        isPlaying && playhead !== null && playhead >= detection.start && playhead < detection.end

      // Each band arrives on its own clock, so a burst of detections staggers in
      // rather than the strip flashing as one block.
      const bornAt = seenRef.current.get(key) ?? now
      const age = reduced ? 1 : Math.min(1, (now - bornAt) / BAND_IN_MS)
      if (age <= 0) continue
      const enter = easeOut(age)

      const x0 = timeToX(detection.start)
      const x1 = timeToX(detection.end)
      const w = Math.max(2, x1 - x0 - 2)

      // 2px surface gap between adjacent fills, per the mark spec.
      // The focused colour fades in over the recessive one rather than swapping.
      const promote = isFocused ? focusRef.current : 0
      ctx.fillStyle = underPlayhead ? palette.playing : promote > 0.5 ? palette.focus : palette.band
      // Confidence is magnitude: it rides on opacity, not on hue.
      // Floor of 0.45 so even the least confident recessive band clears 3:1
      // against the surface (measured 3.42:1 light, 4.43:1 dark).
      const base =
        underPlayhead || promote > 0.5
          ? 0.65 + 0.35 * detection.score
          : 0.45 + 0.25 * detection.score
      ctx.globalAlpha = base * enter

      // Bands grow up from the baseline as they arrive: 4px rounded ends on the
      // data mark, anchored to the lane, per the mark spec.
      const h = (BAND_HEIGHT - 6) * enter
      ctx.beginPath()
      ctx.roundRect(x0 + 1, bandTop + (BAND_HEIGHT - 6 - h), w, h, 3)
      ctx.fill()
      ctx.globalAlpha = 1
    }

    // ---- playhead ----------------------------------------------------------
    if (playhead !== null) {
      const x = timeToX(playhead)
      ctx.strokeStyle = palette.playing
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(Math.round(x) + 0.5, plot.y)
      ctx.lineTo(Math.round(x) + 0.5, plot.y + plot.height + BAND_HEIGHT - 4)
      ctx.stroke()
    }

    // ---- hover crosshair ---------------------------------------------------
    if (hoverTime !== null) {
      const x = timeToX(hoverTime)
      ctx.strokeStyle = palette.muted
      ctx.lineWidth = 1
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(Math.round(x) + 0.5, plot.y)
      ctx.lineTo(Math.round(x) + 0.5, plot.y + plot.height)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // ---- time axis ---------------------------------------------------------
    ctx.fillStyle = palette.muted
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    const step = niceTimeStep(duration, plot.width)
    const axisY = plot.y + plot.height + BAND_HEIGHT - 1
    for (let t = 0; t <= duration + 0.001; t += step) {
      // The label at t=0 and the one nearest the end would otherwise hang off
      // the canvas and be clipped mid-glyph ("2:0").
      const x = timeToX(t)
      if (x - 16 < plot.x) ctx.textAlign = 'left'
      else if (x + 16 > plot.x + plot.width) ctx.textAlign = 'right'
      else ctx.textAlign = 'center'
      ctx.fillText(clock(t), x, axisY)
    }
    ctx.textAlign = 'center'
  }, [
    bitmap, plot, size, palette, detections, focused, isPlaying, hoverTime,
    analysedUntil, duration, spectrogram, timeToX, reduced, positionRef,
  ])

  drawRef.current = draw

  // One animation loop, running only while something is actually moving.
  useEffect(() => {
    let stop = false

    const settled = () => {
      const frontSettled = Math.abs(frontRef.current - analysedUntil) < 0.02
      const focusSettled = Math.abs(focusRef.current - (focused ? 1 : 0)) < 0.01
      const now = performance.now()
      const bandsSettled = detections.every((d) => {
        const born = seenRef.current.get(d.windowIndex * 10000 + d.species.index)
        return born === undefined || now - born > BAND_IN_MS
      })
      // Keyed on real playback: a scrub leaves a playhead parked on screen, and
      // treating that as "still animating" kept a 60 fps redraw of a static
      // canvas running forever (measured at 25-29 ms of main thread per second).
      return frontSettled && focusSettled && bandsSettled && !isPlaying
    }

    const tick = () => {
      if (stop) return
      drawRef.current?.()
      // Stop as soon as everything has arrived; a permanently running rAF on a
      // static picture is a battery leak, not an animation.
      if (settled()) {
        rafRef.current = null
        return
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    draw()
    if (!settled()) rafRef.current = requestAnimationFrame(tick)

    return () => {
      stop = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [draw, analysedUntil, focused, detections, isPlaying, positionVersion])

  const handlePointer = (event: React.PointerEvent) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const time = xToTime(event.clientX - rect.left)
    return Math.max(0, Math.min(duration, time))
  }

  const [dragging, setDragging] = useState(false)

  const nudge = (event: React.KeyboardEvent) => {
    const current = positionRef.current ?? 0
    const step = event.shiftKey ? 10 : 3 // one analysis window by default
    const moves: Record<string, number> = {
      ArrowLeft: -step,
      ArrowRight: step,
      Home: -duration,
      End: duration,
      PageDown: -30,
      PageUp: 30,
    }
    const delta = moves[event.key]
    if (delta === undefined) return
    event.preventDefault()
    onScrub(Math.max(0, Math.min(duration, current + delta)))
  }

  return (
    <figure
      className={[
        'flex flex-col',
        expanded
          ? 'fixed inset-0 z-50 gap-2 bg-surface px-3 pt-[max(0.5rem,env(safe-area-inset-top))] pb-[max(0.5rem,env(safe-area-inset-bottom))]'
          : 'gap-1',
      ].join(' ')}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-ink-3">
          {spectrogram ? `0 – ${(spectrogram.maxHz / 1000).toFixed(0)} kHz` : ''}
        </span>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          aria-pressed={expanded}
          aria-label={expanded ? 'Réduire le spectrogramme' : 'Agrandir le spectrogramme'}
          className="-mr-2 flex h-11 w-11 items-center justify-center rounded-md text-ink-3 transition-colors duration-150 hover:bg-hover hover:text-ink"
        >
          <ExpandIcon expanded={expanded} />
        </button>
      </div>
      <div
        ref={wrapRef}
        className={[
          'relative w-full cursor-crosshair touch-none select-none rounded-sm outline-offset-2 focus-visible:outline-2 focus-visible:outline-focus',
          expanded ? 'min-h-0 flex-1' : 'h-72',
        ].join(' ')}
        onPointerMove={(e) => {
          const time = handlePointer(e)
          setHoverTime(time)
          // Dragging has to scrub. Capturing the pointer and then only acting on
          // pointerdown meant a drag looked live and did nothing.
          if (dragging) onScrub(time)
        }}
        onPointerLeave={() => setHoverTime(null)}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          setDragging(true)
          onScrub(handlePointer(e))
        }}
        onPointerUp={(e) => {
          setDragging(false)
          e.currentTarget.releasePointerCapture(e.pointerId)
        }}
        onPointerCancel={() => setDragging(false)}
        onKeyDown={nudge}
        // It reads as a picture but it behaves as a position control, so it is
        // typed as one: role="img" hid the fact that the region is actionable,
        // and without a tab stop the timeline was mouse-only.
        tabIndex={0}
        role="slider"
        aria-label={
          spectrogram
            ? `Position dans le spectrogramme — ${clock(duration)}, 0 à ` +
              `${(spectrogram.maxHz / 1000).toFixed(0)} kHz, ${detections.length} détections`
            : 'Spectrogramme en cours de calcul'
        }
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(positionRef.current ?? 0)}
        aria-valuetext={clock(positionRef.current ?? 0)}
      >
        <canvas
          ref={canvasRef}
          style={{ width: size.width, height: size.height }}
          className="block"
        />
        {hoverTime !== null && (
          <span
            className="pointer-events-none absolute top-1 rounded bg-ink/90 px-1.5 py-0.5 text-[10px] tabular-nums text-surface"
            style={{ left: Math.min(timeToX(hoverTime) + 6, size.width - 44) }}
          >
            {clock(hoverTime)}
          </span>
        )}
      </div>
      <figcaption className="sr-only">
        Fréquence en ordonnée (0 à 15 kHz), temps en abscisse. La bande sous le spectrogramme
        marque les fenêtres de 3 secondes où une espèce a été détectée ; l'opacité suit la
        confiance. Sélectionner une espèce dans la liste met ses fenêtres en évidence.
      </figcaption>
    </figure>
  )
}

/** Four arrows out, or four arrows in. */
function ExpandIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className="h-5 w-5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {expanded ? (
        <path d="M9 3v6H3M15 3v6h6M9 21v-6H3M15 21v-6h6" />
      ) : (
        <path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6" />
      )}
    </svg>
  )
}

/** Round time gridlines to something a person would choose. */
function niceTimeStep(duration: number, width: number): number {
  const target = duration / Math.max(2, Math.floor(width / 90))
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
  return steps.find((s) => s >= target) ?? 900
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
