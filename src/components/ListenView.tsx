import { useEffect, useRef, useState } from 'react'
import type { MicPhase } from '../hooks/useMicrophone'

interface Props {
  phase: MicPhase
  /** Live RMS, read every frame. A ref so a vumeter does not re-render the app. */
  levelRef: React.RefObject<number>
  elapsed: number
  species: number
  /** Windows the model could not keep up with. Zero on anything modern. */
  dropped: number
  error: string | null
  /**
   * Bumped when a new species is found. The ring flashes on the change — the
   * reveal, and the one moment in this screen that is about an event rather
   * than about a level.
   */
  revealKey: number
  onStart: () => void
  onStop: () => void
}

/** Rings live this long, in ms. */
const RING_LIFE = 2600

/** The reveal ring is faster and brighter — it marks a moment, not a level. */
const REVEAL_LIFE = 900

/** Fastest and slowest spawn interval, in ms. Loud audio ripples more often. */
const SPAWN_FAST = 260
const SPAWN_SLOW = 1400

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
  )
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(query.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])
  return reduced
}

function cssColour(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

/**
 * Live listening.
 *
 * No spectrogram here, and that is a decision rather than an omission. A
 * scrolling spectrogram is an *instrument*: it exists to read a recording you
 * already have, with a timeline you can scrub and occurrences you can revisit.
 * Listening live is not reading, it is waiting — so the screen is one living
 * object and a reveal, and the file mode keeps the spectrogram it was designed
 * for.
 *
 * The rings are **driven by the microphone's real RMS**: their cadence and their
 * size follow what the device is picking up. That keeps the project's rule from
 * P2 — nothing moves that does not mean something — and it makes the most common
 * failure visible without a word of copy: if the rings are flat, the microphone
 * is not hearing you.
 *
 * One rAF loop, and it stops when capture stops. That was the defect the P2
 * review found; it is not being reintroduced here.
 */
export function ListenView({
  phase,
  levelRef,
  elapsed,
  species,
  dropped,
  error,
  revealKey,
  onStart,
  onStop,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState(0)
  const reduced = useReducedMotion()
  const listening = phase === 'listening'
  /**
   * Read inside the animation loop, so a reveal does not restart the loop.
   * Stamped from an effect rather than from the render body: writing a ref
   * during render is a hazard under StrictMode's double invocation, and this
   * app renders inside one.
   */
  const revealRef = useRef({ at: 0 })
  useEffect(() => {
    if (revealKey > 0) revealRef.current.at = performance.now()
  }, [revealKey])

  useEffect(() => {
    const element = wrapRef.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => {
      setSize(Math.min(entry.contentRect.width, entry.contentRect.height))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const focus = cssColour('--color-focus', '#3e8fdb')
    const gold = cssColour('--color-gold', '#efc373')

    const centre = size / 2
    const buttonRadius = size * 0.19
    const maxRadius = size * 0.48

    const rings: { born: number; strength: number }[] = []
    let smoothed = 0
    let lastSpawn = 0
    let frame: number | null = null
    let stopped = false

    const draw = (now: number) => {
      ctx.clearRect(0, 0, size, size)

      // A soft curve, because RMS is tiny for ordinary field audio and linear
      // scaling would leave the rings flat for anything short of shouting.
      const raw = Math.min(1, Math.sqrt(Math.max(0, levelRef.current ?? 0)) * 2.4)
      // Asymmetric smoothing: rise fast so a call registers, fall slowly so the
      // ring does not flicker between syllables.
      smoothed += (raw - smoothed) * (raw > smoothed ? 0.35 : 0.06)

      // A reveal is an event, not a level, so it gets its own ring in the focus
      // hue — the same colour that means "this one" everywhere else in the app.
      const sinceReveal = now - revealRef.current.at
      if (revealRef.current.at > 0 && sinceReveal < REVEAL_LIFE && listening) {
        const t = sinceReveal / REVEAL_LIFE
        const radius = buttonRadius + (maxRadius - buttonRadius) * t * 1.15
        ctx.beginPath()
        ctx.arc(centre, centre, radius, 0, Math.PI * 2)
        ctx.strokeStyle = focus
        ctx.globalAlpha = (1 - t) * 0.9
        ctx.lineWidth = 3 + (1 - t) * 4
        ctx.stroke()
        ctx.globalAlpha = 1
      }

      if (listening && !reduced) {
        const interval = SPAWN_SLOW - (SPAWN_SLOW - SPAWN_FAST) * smoothed
        if (now - lastSpawn > interval) {
          rings.push({ born: now, strength: 0.3 + smoothed * 0.7 })
          lastSpawn = now
        }
        while (rings.length && now - rings[0].born > RING_LIFE) rings.shift()

        for (const ring of rings) {
          const t = (now - ring.born) / RING_LIFE
          const radius = buttonRadius + (maxRadius - buttonRadius) * t
          ctx.beginPath()
          ctx.arc(centre, centre, radius, 0, Math.PI * 2)
          ctx.strokeStyle = gold
          // Linear rather than squared: the squared falloff made every ring but
          // the youngest invisible, which is not a ripple, it is a blink.
          ctx.globalAlpha = (1 - t) * ring.strength * 0.85
          ctx.lineWidth = 1.5 + ring.strength * 3
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }

      // The level ring, hugging the button. Under `prefers-reduced-motion` this
      // is the whole display: the information survives, only the travel is
      // removed — the point of the preference is not to be told nothing.
      if (listening) {
        const radius = buttonRadius * (1.12 + 0.5 * smoothed)
        ctx.beginPath()
        ctx.arc(centre, centre, radius, 0, Math.PI * 2)
        ctx.strokeStyle = gold
        ctx.globalAlpha = 0.5 + smoothed * 0.45
        ctx.lineWidth = 3 + smoothed * 7
        ctx.stroke()
        ctx.globalAlpha = 1
      }

      if (!stopped && listening) frame = requestAnimationFrame(draw)
    }

    if (listening) {
      frame = requestAnimationFrame(draw)
    } else {
      // One paint to clear whatever the last session left behind, then nothing.
      ctx.clearRect(0, 0, size, size)
    }

    return () => {
      stopped = true
      if (frame !== null) cancelAnimationFrame(frame)
    }
  }, [size, listening, reduced, levelRef])

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 py-6">
      <div ref={wrapRef} className="relative flex aspect-square w-full max-w-[320px] items-center justify-center">
        {listening && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-[18%] rounded-full blur-2xl"
            style={{
              background: 'radial-gradient(circle, var(--color-gold) 0%, transparent 70%)',
              opacity: 0.22,
            }}
          />
        )}
        <canvas
          ref={canvasRef}
          aria-hidden
          style={{ width: size, height: size }}
          className="absolute inset-0 m-auto block"
        />
        <button
          type="button"
          onClick={listening ? onStop : onStart}
          disabled={phase === 'starting'}
          aria-pressed={listening}
          // A real attribute rather than a visually-hidden span: the button is
          // icon-only, and this is the name both assistive tech and the test
          // harness look up.
          aria-label={listening ? "Arrêter l'écoute" : 'Écouter le micro'}
          className={[
            'relative flex h-[38%] w-[38%] items-center justify-center rounded-full',
            'text-sm font-medium transition-transform duration-150 active:scale-95',
            'disabled:opacity-70',
            'btn-primary',
          ].join(' ')}
        >
          {phase === 'starting' ? '…' : listening ? <StopIcon /> : <MicIcon />}
        </button>
      </div>

      <div className="flex flex-col items-center gap-1 text-center">
        {phase === 'idle' && (
          <>
            <p className="text-base font-medium">Écouter autour de vous</p>
            <p className="max-w-[28ch] text-sm text-ink-3">
              Le micro n'est demandé qu'au moment où vous appuyez, et rien n'est envoyé nulle part.
            </p>
          </>
        )}
        {phase === 'starting' && <p className="text-sm text-ink-3">Ouverture du micro…</p>}
        {listening && (
          <>
            <p className="text-sm tabular-nums text-ink-2" role="status" aria-live="off">
              à l'écoute depuis {clock(elapsed)}
            </p>
            <p className="text-sm text-ink-3" role="status" aria-live="polite">
              {species === 0
                ? 'aucune espèce pour l’instant'
                : `${species} espèce${species > 1 ? 's' : ''} trouvée${species > 1 ? 's' : ''}`}
            </p>
            {/* Same rule as `truncatedWindows` on the file path: a list that is
                short because the device could not keep up must say so. */}
            {dropped > 0 && (
              <p className="max-w-[30ch] text-xs text-ink-3">
                {dropped} fenêtre{dropped > 1 ? 's' : ''} non analysée
                {dropped > 1 ? 's' : ''} — cet appareil n'arrive pas à suivre le direct.
              </p>
            )}
          </>
        )}
        {error && (
          <p role="alert" className="max-w-[32ch] rounded-lg border border-play/40 bg-play/10 p-3 text-sm">
            {error}
          </p>
        )}
      </div>
    </div>
  )
}

function clock(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function MicIcon() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className="h-7 w-7"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v4" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg aria-hidden viewBox="0 0 24 24" className="h-6 w-6" fill="currentColor">
      <rect x="7" y="7" width="10" height="10" rx="2" />
    </svg>
  )
}
