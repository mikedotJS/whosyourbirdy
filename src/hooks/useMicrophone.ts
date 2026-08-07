import { useCallback, useEffect, useRef, useState } from 'react'
import type { BirdNetAnalyzer } from '../lib/birdnet/analyze'
import { DEFAULT_MIN_CONFIDENCE, SAMPLE_RATE, WINDOW_SECONDS } from '../lib/birdnet/constants'
import { isNonEvent, loadLabels, type Species } from '../lib/birdnet/labels'
import { LIVE_HOP_SAMPLES, SlidingWindower } from '../lib/birdnet/stream'
import { LIVE_WORKLET_NAME, createWorkletUrl } from '../lib/birdnet/worklet'
import type { Detection } from '../lib/birdnet/types'

export type MicPhase = 'idle' | 'starting' | 'listening' | 'error'

export interface MicState {
  phase: MicPhase
  /** Detections so far, most recent first. */
  detections: Detection[]
  /** Seconds of audio captured. */
  elapsed: number
  error: string | null
  /** Windows scored so far — the honest measure of progress while listening. */
  windows: number
  /**
   * Windows the model could not keep up with.
   *
   * Non-zero means this device is slower than real time at a 1 s hop, so some
   * audio was never scored. Reporting it is the same rule as `truncatedWindows`
   * on the file path: a short list must never look like a complete one.
   */
  dropped: number
}

const initial: MicState = {
  phase: 'idle',
  detections: [],
  elapsed: 0,
  error: null,
  windows: 0,
  dropped: 0,
}

function describeError(error: unknown): string {
  const name = error instanceof DOMException ? error.name : ''
  if (name === 'NotAllowedError') {
    return "L'accès au micro a été refusé. Autorisez-le dans les réglages du navigateur pour écouter en direct."
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return "Aucun micro utilisable n'a été trouvé."
  }
  if (name === 'NotReadableError') {
    return 'Le micro est déjà utilisé par une autre application.'
  }
  return error instanceof Error ? error.message : String(error)
}

/**
 * Live listening.
 *
 * Three decisions here are not comfort settings, they are correctness:
 *
 * 1. **`echoCancellation`, `noiseSuppression` and `autoGainControl` are all
 *    explicitly `false`.** These are designed for speech and would destroy bird
 *    song — noise suppression in particular treats a sustained tonal signal as
 *    something to remove. And AGC is *amplitude normalisation*, exactly what the
 *    README has forbidden since P0: the model normalises every window itself,
 *    so a second normalisation upstream changes the statistics it was trained on.
 * 2. **The context is pinned to 48 kHz**, the rate the model was trained at, so
 *    the browser resamples the device once at the source rather than leaving us
 *    to do it per window.
 * 3. **Stopping releases the device**, `track.stop()` and not just the loop.
 *    Leaving a track live keeps the browser's recording indicator on and holds
 *    the microphone against other applications.
 *
 * Windows are 3 s, hopping every second: a bird singing now is scored within
 * about 3.1 s instead of up to 6.
 */
export function useMicrophone(analyzer: BirdNetAnalyzer | null) {
  const [state, setState] = useState<MicState>(initial)
  /** Live RMS in a ref, not state: it updates ten times a second and only the
      canvas reads it. As state it would re-render the tree for a vumeter. */
  const levelRef = useRef(0)

  const streamRef = useRef<MediaStream | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const nodeRef = useRef<AudioWorkletNode | null>(null)
  const windowerRef = useRef<SlidingWindower | null>(null)
  const allowedRef = useRef<Int32Array | null>(null)
  const labelsRef = useRef<Species[] | null>(null)
  /** Bumped on every stop, so a window still in flight cannot report late. */
  const sessionRef = useRef(0)
  /** One inference at a time: queueing them would grow without bound if the
      device is faster than the model. */
  const busyRef = useRef(false)
  const pendingRef = useRef<{ samples: Float32Array; offset: number } | null>(null)
  /** Samples actually captured, which is what "listening for" means. */
  const capturedRef = useRef(0)

  const stop = useCallback(() => {
    sessionRef.current++
    nodeRef.current?.port.close()
    nodeRef.current?.disconnect()
    nodeRef.current = null
    // Releasing the tracks, not only the graph: closing the AudioContext alone
    // leaves the device open and the browser's recording indicator lit.
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    void contextRef.current?.close().catch(() => {})
    contextRef.current = null
    windowerRef.current = null
    pendingRef.current = null
    busyRef.current = false
    levelRef.current = 0
    capturedRef.current = 0
    setState((s) => (s.phase === 'error' ? s : { ...s, phase: 'idle' }))
  }, [])

  useEffect(() => stop, [stop])

  const start = useCallback(async () => {
    if (!analyzer) return
    setState({ ...initial, phase: 'starting' })
    const session = ++sessionRef.current

    try {
      // Permission is requested here and nowhere else — this function only runs
      // from a click.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      })
      if (sessionRef.current !== session) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      streamRef.current = stream

      const context = new AudioContext({ sampleRate: SAMPLE_RATE })
      contextRef.current = context
      // Normally already running, since this only ever happens inside a click —
      // but a context created while the tab is backgrounded starts suspended,
      // and a suspended context delivers no audio at all.
      if (context.state === 'suspended') await context.resume()

      const url = createWorkletUrl()
      try {
        await context.audioWorklet.addModule(url)
      } finally {
        URL.revokeObjectURL(url)
      }
      if (sessionRef.current !== session) return

      const [labels] = await Promise.all([loadLabels('fr'), analyzer.prepare()])
      if (sessionRef.current !== session) return
      labelsRef.current = labels
      allowedRef.current = Int32Array.from(
        labels.filter((s) => !isNonEvent(s)).map((s) => s.index),
      )

      const windower = new SlidingWindower()
      windowerRef.current = windower

      const source = context.createMediaStreamSource(stream)
      const node = new AudioWorkletNode(context, LIVE_WORKLET_NAME)
      nodeRef.current = node

      node.port.onmessage = (event: MessageEvent<{ samples: Float32Array; rms: number }>) => {
        if (sessionRef.current !== session) return
        levelRef.current = event.data.rms

        for (const window of windower.push(event.data.samples)) {
          // Newest wins. If the model falls behind, the useful thing to score is
          // what is happening now, not a backlog from ten seconds ago — and the
          // window it displaces is counted rather than lost quietly.
          if (busyRef.current) {
            if (pendingRef.current) setState((s) => ({ ...s, dropped: s.dropped + 1 }))
            pendingRef.current = window
            continue
          }
          void score(window, session)
        }

        capturedRef.current += event.data.samples.length
        setState((s) => ({ ...s, elapsed: capturedRef.current / SAMPLE_RATE }))
      }

      // The node has no output; connecting it to the destination would echo the
      // microphone into the speakers. `source -> node` is enough to pull audio
      // through, since `process()` runs whenever the node has an input.
      source.connect(node)

      setState((s) => ({ ...s, phase: 'listening' }))
    } catch (error) {
      if (sessionRef.current !== session) return
      stop()
      setState({ ...initial, phase: 'error', error: describeError(error) })
    }

    async function score(
      window: { samples: Float32Array; offset: number },
      forSession: number,
    ): Promise<void> {
      if (!analyzer) return
      busyRef.current = true
      try {
        const result = await analyzer.analyzeWindow(window.samples, window.offset, {
          minConfidence: DEFAULT_MIN_CONFIDENCE,
          allowedClasses: allowedRef.current,
        })
        if (sessionRef.current !== forSession) return

        const labels = labelsRef.current
        if (!labels) return
        const start = window.offset / SAMPLE_RATE
        const found: Detection[] = []
        for (let i = 0; i < result.classes.length; i++) {
          found.push({
            species: labels[result.classes[i]],
            score: result.scores[i],
            start,
            end: start + WINDOW_SECONDS,
            windowIndex: Math.round(window.offset / LIVE_HOP_SAMPLES),
          })
        }
        setState((s) => ({
          ...s,
          windows: s.windows + 1,
          // Newest first: in live listening the interesting thing is what just
          // happened, not what the recording started with.
          detections: found.length ? [...found, ...s.detections] : s.detections,
        }))
      } finally {
        busyRef.current = false
        const next = pendingRef.current
        pendingRef.current = null
        if (next && sessionRef.current === forSession) void score(next, forSession)
      }
    }
  }, [analyzer, stop])

  return { ...state, levelRef, start, stop }
}
