/**
 * Browser side of the parity harness (levels B and C).
 *
 * This is not a demo page: it runs the *production* `lib/birdnet` modules inside
 * a real browser so the comparison covers the parts that only exist there —
 * `decodeAudioData`, `OfflineAudioContext` resampling, the Web Worker, ORT's WASM
 * backend under a real browser's WASM engine.
 *
 * It does two passes over the file:
 *
 *   1. the raw path (`decodeAudio` -> `planWindows` -> `sliceWindow` ->
 *      `inferWindow`), which yields full 6522-class logits per window so they can
 *      be diffed against the Python reference;
 *   2. the shipping path (`BirdNetAnalyzer.analyze`), whose detections are then
 *      checked against what pass 1's logits imply. That is what proves the
 *      worker, the sigmoid, the thresholding and the label mapping agree with the
 *      tensors — a bug there would otherwise hide behind a passing level B.
 */
import { decodeAudio } from './lib/birdnet/audio'
import { inferWindow, loadModel } from './lib/birdnet/model'
import { planWindows, sliceWindow } from './lib/birdnet/windows'
import { flatSigmoid } from './lib/birdnet/sigmoid'
import { BirdNetAnalyzer } from './lib/birdnet/analyze'
import { DEFAULT_MIN_CONFIDENCE, N_CLASSES } from './lib/birdnet/constants'
import { isNonEvent, loadLabels } from './lib/birdnet/labels'

interface ParityOutput {
  logits: number[]
  meta: {
    decodedRate: number
    windows: number
    expectedWindows: number
    windowsMatch: boolean
    samples: number
    medianMs: number
    duration: number
    /** Largest |score| gap between the raw path and the analyzer path. */
    analyzerDrift: number
    /** Detections the analyzer reported that the raw logits do not support. */
    analyzerMismatches: number
  }
}

interface BenchOutput {
  timings: number[]
  median: number
  decodeMs: number
  modelLoadMs: number
}

declare global {
  interface Window {
    runParity: (audioUrl: string, expectedWindows: number, classes: number) => Promise<ParityOutput>
    benchmark: (audioUrl: string) => Promise<BenchOutput>
  }
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

window.runParity = async (audioUrl, expectedWindows, classes) => {
  if (classes !== N_CLASSES) throw new Error(`harness expects ${N_CLASSES} classes, got ${classes}`)

  const bytes = await (await fetch(audioUrl)).arrayBuffer()

  // ---- pass 1: raw path, full logits -------------------------------------
  const audio = await decodeAudio(bytes.slice(0))
  const windows = planWindows(audio.samples.length, 0)
  // A window-count difference is itself a result, not a crash: resamplers do not
  // agree on output length (librosa rounds up, the browser does not), and one
  // extra sample buys a whole extra window. The caller decides whether that is
  // fatal — it must be for level B, and is merely reported for level C.
  const windowsMatch = windows.length === expectedWindows

  const { session } = await loadModel()
  const logits = new Float32Array(windows.length * N_CLASSES)
  const timings: number[] = []

  for (const window of windows) {
    const t0 = performance.now()
    const out = await inferWindow(session, sliceWindow(audio.samples, window.offsetSamples))
    timings.push(performance.now() - t0)
    logits.set(out, window.index * N_CLASSES)
  }

  // ---- pass 2: the shipping path, cross-checked against pass 1 -----------
  const labels = await loadLabels('en')
  const analyzer = new BirdNetAnalyzer()
  const result = await analyzer.analyze(bytes.slice(0), {
    overlap: 0,
    minConfidence: DEFAULT_MIN_CONFIDENCE,
    locale: 'en',
    excludeNonEvents: false,
  })
  analyzer.dispose()

  let analyzerDrift = 0
  let analyzerMismatches = 0
  for (const detection of result.detections) {
    const expected = flatSigmoid(
      logits.subarray(detection.windowIndex * N_CLASSES, (detection.windowIndex + 1) * N_CLASSES),
    )[detection.species.index]
    const drift = Math.abs(expected - detection.score)
    if (drift > analyzerDrift) analyzerDrift = drift
    if (drift > 1e-6) analyzerMismatches++
    if (labels[detection.species.index].scientificName !== detection.species.scientificName) {
      analyzerMismatches++
    }
  }

  // The analyzer must also not *miss* anything the raw logits put above the bar.
  let expectedCount = 0
  for (let w = 0; w < windows.length; w++) {
    const scores = flatSigmoid(logits.subarray(w * N_CLASSES, (w + 1) * N_CLASSES))
    for (let c = 0; c < N_CLASSES; c++) {
      if (scores[c] >= DEFAULT_MIN_CONFIDENCE && !isNonEvent(labels[c])) expectedCount++
    }
  }
  const reported = result.detections.filter((d) => !isNonEvent(d.species)).length
  if (reported !== expectedCount) analyzerMismatches += Math.abs(expectedCount - reported)

  return {
    logits: Array.from(logits),
    meta: {
      decodedRate: audio.decodedSampleRate,
      windows: windows.length,
      expectedWindows,
      windowsMatch,
      samples: audio.samples.length,
      medianMs: median(timings),
      duration: audio.duration,
      analyzerDrift,
      analyzerMismatches,
    },
  }
}

/**
 * Timing entry point for `scripts/bench.mjs`.
 *
 * Measures the shipping path through `BirdNetAnalyzer`, so the numbers include
 * the worker round-trip and the sigmoid pass, not just `session.run`. The model
 * load is timed separately because it is a once-per-session cost, and the Cache
 * API makes the second run of a session unrepresentative.
 */
window.benchmark = async (audioUrl) => {
  const bytes = await (await fetch(audioUrl)).arrayBuffer()

  const analyzer = new BirdNetAnalyzer()
  const loadStart = performance.now()
  await analyzer.prepare()
  const modelLoadMs = performance.now() - loadStart

  const decodeStart = performance.now()
  await decodeAudio(bytes.slice(0))
  const decodeMs = performance.now() - decodeStart

  const timings: number[] = []
  const result = await analyzer.analyze(
    bytes.slice(0),
    { overlap: 0, locale: 'en' },
    { onWindow: (w) => timings.push(w.inferenceMs) },
  )
  analyzer.dispose()

  return { timings, median: result.medianInferenceMs, decodeMs, modelLoadMs }
}
