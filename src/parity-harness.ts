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
import { SlidingWindower } from './lib/birdnet/stream'
import { WORKLET_BATCH_SAMPLES } from './lib/birdnet/worklet'
import { flatSigmoid } from './lib/birdnet/sigmoid'
import { BirdNetAnalyzer } from './lib/birdnet/analyze'
import { DEFAULT_MIN_CONFIDENCE, N_CLASSES, WINDOW_SAMPLES } from './lib/birdnet/constants'
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

interface StreamParityOutput {
  streamedWindows: number
  plannedWindows: number
  paddedWindowsSkipped: number
  sampleMismatches: number
  firstMismatchAt: number
  detectionsCompared: number
  scoreMismatches: number
  worstDelta: number
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
    runStreamParity: (audioUrl: string) => Promise<StreamParityOutput>
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
  // Deliberately the shipping defaults: excludeNonEvents stays true, so the
  // worker takes its allowedClasses branch — the one every real call uses. An
  // earlier version passed false here and left that branch untested, which meant
  // an indexing bug in it could have passed the whole suite.
  const result = await analyzer.analyze(bytes.slice(0), {
    overlap: 0,
    minConfidence: DEFAULT_MIN_CONFIDENCE,
    locale: 'en',
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
    // Comparing detection.species against labels[detection.species.index] would
    // be a tautology — they are the same object. The drift check above is the
    // real label test: it looks the score up in pass 1's logits *by class index*,
    // so any shift between class index and reported species shows up as drift.
  }

  // The analyzer must also not *miss* anything the raw logits put above the bar.
  let expectedCount = 0
  for (let w = 0; w < windows.length; w++) {
    const scores = flatSigmoid(logits.subarray(w * N_CLASSES, (w + 1) * N_CLASSES))
    for (let c = 0; c < N_CLASSES; c++) {
      if (scores[c] >= DEFAULT_MIN_CONFIDENCE && !isNonEvent(labels[c])) expectedCount++
    }
  }
  // The analyzer ran with excludeNonEvents:true, so it should report exactly the
  // above-threshold bird classes and nothing else.
  const reported = result.detections.length
  if (reported !== expectedCount) analyzerMismatches += Math.abs(expectedCount - reported)
  const leakedNonEvents = result.detections.filter((d) => isNonEvent(d.species)).length
  analyzerMismatches += leakedNonEvents

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
 * Level F: the streaming path against the file path.
 *
 * Live listening cannot be compared to BirdNET — there is no reference for audio
 * that only existed once. What *can* be proved is that the streaming path and
 * the file path are the same computation, and that is the whole risk: the model
 * is identical, so any divergence is a windowing bug.
 *
 * The file is pushed through `SlidingWindower` in 100 ms blocks, exactly as the
 * audio worklet delivers them, with the hop set to a full window so the stream's
 * windows land on the same offsets `planWindows` produces. Then:
 *
 *   1. every streamed window is compared **sample for sample** against
 *      `sliceWindow` at the same offset — this is the windowing proof;
 *   2. every streamed window is scored through `analyzeWindow` (worker,
 *      transfer, `selectDetections`) and its scores are required to be **exactly
 *      equal** to what `analyze` reported for that window. Not "within a
 *      tolerance": same model, same machine, same process, so anything other
 *      than bit equality is a defect rather than noise.
 *
 * The file path's final window is zero-padded and has no streaming counterpart —
 * a live stream has no end — so it is excluded and counted, not quietly skipped.
 */
window.runStreamParity = async (audioUrl) => {
  const bytes = await (await fetch(audioUrl)).arrayBuffer()
  const audio = await decodeAudio(bytes.slice(0))

  const planned = planWindows(audio.samples.length, 0)
  const windower = new SlidingWindower(WINDOW_SAMPLES, WINDOW_SAMPLES)

  const streamed: { samples: Float32Array; offset: number }[] = []
  for (let read = 0; read < audio.samples.length; read += WORKLET_BATCH_SAMPLES) {
    const block = audio.samples.subarray(read, Math.min(read + WORKLET_BATCH_SAMPLES, audio.samples.length))
    streamed.push(...windower.push(block))
  }

  // 1. The windows themselves.
  let sampleMismatches = 0
  let firstMismatchAt = -1
  for (const [i, window] of streamed.entries()) {
    if (window.offset !== planned[i]?.offsetSamples) {
      sampleMismatches++
      if (firstMismatchAt < 0) firstMismatchAt = i
      continue
    }
    const reference = sliceWindow(audio.samples, window.offset)
    for (let s = 0; s < WINDOW_SAMPLES; s++) {
      if (window.samples[s] !== reference[s]) {
        sampleMismatches++
        if (firstMismatchAt < 0) firstMismatchAt = i
        break
      }
    }
  }

  // 2. The scores, through both code paths.
  const analyzer = new BirdNetAnalyzer()
  const labels = await loadLabels('en')
  const allowedClasses = Int32Array.from(labels.filter((s) => !isNonEvent(s)).map((s) => s.index))

  const fileResult = await analyzer.analyze(bytes.slice(0), {
    overlap: 0,
    minConfidence: DEFAULT_MIN_CONFIDENCE,
    locale: 'en',
  })
  const byWindow = new Map<number, Map<number, number>>()
  for (const detection of fileResult.detections) {
    let bucket = byWindow.get(detection.windowIndex)
    if (!bucket) byWindow.set(detection.windowIndex, (bucket = new Map()))
    bucket.set(detection.species.index, detection.score)
  }

  let scoreMismatches = 0
  let worstDelta = 0
  let compared = 0
  for (const [i, window] of streamed.entries()) {
    const result = await analyzer.analyzeWindow(window.samples, window.offset, {
      minConfidence: DEFAULT_MIN_CONFIDENCE,
      allowedClasses,
    })
    const expected = byWindow.get(i) ?? new Map<number, number>()
    if (result.classes.length !== expected.size) scoreMismatches++
    for (let k = 0; k < result.classes.length; k++) {
      compared++
      const want = expected.get(result.classes[k])
      if (want === undefined) {
        scoreMismatches++
        continue
      }
      // Exact, not approximate. See the note above.
      const delta = Math.abs(want - result.scores[k])
      if (delta > worstDelta) worstDelta = delta
      if (result.scores[k] !== want) scoreMismatches++
    }
  }
  analyzer.dispose()

  return {
    streamedWindows: streamed.length,
    plannedWindows: planned.length,
    /** Windows the file path pads and the stream never sees. Expected: 0 or 1. */
    paddedWindowsSkipped: planned.length - streamed.length,
    sampleMismatches,
    firstMismatchAt,
    detectionsCompared: compared,
    scoreMismatches,
    worstDelta,
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
