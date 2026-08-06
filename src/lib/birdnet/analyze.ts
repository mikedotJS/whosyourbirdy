import { DEFAULT_MIN_CONFIDENCE, DEFAULT_SENSITIVITY, MODEL_BASE_URL, ORT_WASM_PATH } from './constants'
import { decodeAudio, type DecodedAudio } from './audio'
import { isNonEvent, loadLabels, type Species } from './labels'
import type { SpectrogramData } from './spectrogram'
import type {
  AnalysisProgress,
  AnalysisResult,
  AnalyzeOptions,
  Detection,
  ModelLoadProgress,
  WindowResult,
  WorkerRequest,
  WorkerResponse,
} from './types'

/** Cap on detections reported per window; only bites at very low thresholds. */
const TOP_K_PER_WINDOW = 64

export interface AnalyzeCallbacks {
  onModelProgress?: (progress: ModelLoadProgress) => void
  onProgress?: (progress: AnalysisProgress) => void
  /** Called once per window, in order, as results stream back. */
  onWindow?: (result: WindowResult) => void
  /** Fires once, before the first window, so the picture can render early. */
  onSpectrogram?: (spectrogram: SpectrogramData) => void
}

function median(values: ArrayLike<number>): number {
  if (values.length === 0) return 0
  const sorted = Array.from(values).sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/**
 * Owns the worker and the model that lives inside it.
 *
 * The session is expensive (52 MB of weights, a WASM compile), so it is created
 * once and reused across files. Callers that never analyse anything never pay
 * for it: nothing is loaded until `analyze` is called.
 */
export class BirdNetAnalyzer {
  private worker: Worker | null = null
  private ready: Promise<void> | null = null
  private nextRequestId = 1
  /** In-flight analyses, so `dispose()` can settle them instead of stranding them. */
  private readonly pending = new Map<number, (reason: unknown) => void>()
  private pendingReady = false

  constructor(
    private readonly options: { modelBaseUrl?: string; wasmPath?: string } = {},
  ) {}

  private ensureWorker(): Worker {
    this.worker ??= new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    return this.worker
  }

  /** Load the model. Safe to call early to warm the cache. */
  async prepare(onModelProgress?: (progress: ModelLoadProgress) => void): Promise<void> {
    const worker = this.ensureWorker()
    this.ready ??= new Promise<void>((resolve, reject) => {
      const onMessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data
        if (message.type === 'load-progress') {
          onModelProgress?.(message.progress)
        } else if (message.type === 'ready') {
          worker.removeEventListener('message', onMessage)
          this.pendingReady = false
          resolve()
        } else if (message.type === 'error') {
          worker.removeEventListener('message', onMessage)
          this.ready = null
          this.pendingReady = false
          reject(new Error(message.message))
        }
      }
      this.pendingReady = true
      worker.addEventListener('message', onMessage)
      const request: WorkerRequest = {
        type: 'init',
        modelUrl: this.options.modelBaseUrl ?? MODEL_BASE_URL,
        wasmPath: this.options.wasmPath ?? ORT_WASM_PATH,
      }
      worker.postMessage(request)
    })
    return this.ready
  }

  /** True once the model is loaded, so callers can report the right wait. */
  get isReady(): boolean {
    return this.ready !== null && this.pendingReady === false
  }

  /**
   * Analyse a file end to end: decode, window, infer, label.
   *
   * Decoding happens on the calling thread because `decodeAudioData` is not
   * available in a worker; the PCM is then transferred (not copied) into the
   * worker, so a two-minute file moves as a pointer rather than 23 MB.
   */
  async analyze(
    file: Blob | ArrayBuffer,
    options: AnalyzeOptions = {},
    callbacks: AnalyzeCallbacks = {},
  ): Promise<AnalysisResult> {
    const {
      overlap = 0,
      minConfidence = DEFAULT_MIN_CONFIDENCE,
      sensitivity = DEFAULT_SENSITIVITY,
      locale = 'fr',
      excludeNonEvents = true,
      signal,
    } = options

    // The worker processes one analysis at a time. Leaving a previous run alive
    // would both delay this one and let its results arrive afterwards, under the
    // new file's name.
    this.cancelAll()

    const [labels] = await Promise.all([
      loadLabels(locale, this.options.modelBaseUrl),
      this.prepare(callbacks.onModelProgress),
    ])

    const audio: DecodedAudio = await decodeAudio(file)
    signal?.throwIfAborted()

    const allowedClasses = excludeNonEvents
      ? Int32Array.from(labels.filter((s) => !isNonEvent(s)).map((s) => s.index))
      : null

    return this.run(audio, labels, allowedClasses, {
      overlap,
      minConfidence,
      sensitivity,
      signal,
    }, callbacks)
  }

  private run(
    audio: DecodedAudio,
    labels: Species[],
    allowedClasses: Int32Array | null,
    options: { overlap: number; minConfidence: number; sensitivity: number; signal?: AbortSignal },
    callbacks: AnalyzeCallbacks,
  ): Promise<AnalysisResult> {
    const worker = this.ensureWorker()
    const requestId = this.nextRequestId++

    return new Promise<AnalysisResult>((resolve, reject) => {
      const detections: Detection[] = []
      let completed = 0
      let truncatedWindows = 0

      this.pending.set(requestId, reject)

      const cleanup = () => {
        this.pending.delete(requestId)
        worker.removeEventListener('message', onMessage)
        options.signal?.removeEventListener('abort', onAbort)
      }

      const onAbort = () => {
        worker.postMessage({ type: 'cancel', requestId } satisfies WorkerRequest)
      }

      const onMessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data
        if ('requestId' in message && message.requestId !== undefined && message.requestId !== requestId) {
          return
        }

        switch (message.type) {
          case 'spectrogram': {
            callbacks.onSpectrogram?.({
              columns: message.columns,
              bins: message.bins,
              magnitudes: message.magnitudes,
              duration: message.duration,
              maxHz: message.maxHz,
            })
            break
          }
          case 'window': {
            const windowDetections: Detection[] = []
            for (let i = 0; i < message.classes.length; i++) {
              windowDetections.push({
                species: labels[message.classes[i]],
                score: message.scores[i],
                start: message.start,
                end: message.end,
                windowIndex: message.windowIndex,
              })
            }
            detections.push(...windowDetections)
            completed++
            if (message.truncated) truncatedWindows++

            callbacks.onWindow?.({
              windowIndex: message.windowIndex,
              start: message.start,
              end: message.end,
              detections: windowDetections,
              truncated: message.truncated,
              inferenceMs: message.inferenceMs,
            })
            callbacks.onProgress?.({
              completed,
              total: message.total,
              // Clamp to the real duration so the analysis front never overshoots
              // the waveform on a zero-padded final window.
              seconds: Math.min(message.end, audio.duration),
            })
            break
          }
          case 'done': {
            cleanup()
            detections.sort((a, b) => b.score - a.score)
            resolve({
              detections,
              truncatedWindows,
              windowCount: message.windowCount,
              duration: audio.duration,
              medianInferenceMs: median(message.timings),
              totalMs: message.totalMs,
            })
            break
          }
          case 'cancelled':
            cleanup()
            reject(new DOMException('Analysis cancelled', 'AbortError'))
            break
          case 'error':
            cleanup()
            reject(new Error(message.message))
            break
        }
      }

      worker.addEventListener('message', onMessage)
      options.signal?.addEventListener('abort', onAbort, { once: true })

      // The PCM is transferred; `audio.samples` is detached in this thread after
      // this call, which is why callers get `duration` rather than the buffer.
      const request: WorkerRequest = {
        type: 'analyze',
        requestId,
        samples: audio.samples,
        overlap: options.overlap,
        minConfidence: options.minConfidence,
        sensitivity: options.sensitivity,
        allowedClasses,
        topKPerWindow: TOP_K_PER_WINDOW,
      }
      worker.postMessage(request, [audio.samples.buffer])
    })
  }

  /** Cancel every in-flight analysis, without tearing down the model. */
  cancelAll(): void {
    for (const requestId of this.pending.keys()) {
      this.worker?.postMessage({ type: 'cancel', requestId } satisfies WorkerRequest)
    }
  }

  /**
   * Release the worker and the 52 MB session inside it.
   *
   * Any analysis still in flight is rejected first. Terminating the worker means
   * its `done`/`cancelled`/`error` message can never arrive, so without this the
   * pending promise would simply never settle — a spinner that spins forever
   * after the user navigates away or cancels.
   */
  dispose(): void {
    for (const reject of this.pending.values()) {
      reject(new DOMException('Analyzer disposed', 'AbortError'))
    }
    this.pending.clear()
    this.worker?.terminate()
    this.worker = null
    this.ready = null
  }
}

/** Group detections by species, keeping the best score and every occurrence. */
export function groupBySpecies(detections: Detection[]): Array<{
  species: Species
  count: number
  bestScore: number
  occurrences: Detection[]
}> {
  const groups = new Map<number, { species: Species; count: number; bestScore: number; occurrences: Detection[] }>()

  for (const detection of detections) {
    const existing = groups.get(detection.species.index)
    if (existing) {
      existing.count++
      existing.bestScore = Math.max(existing.bestScore, detection.score)
      existing.occurrences.push(detection)
    } else {
      groups.set(detection.species.index, {
        species: detection.species,
        count: 1,
        bestScore: detection.score,
        occurrences: [detection],
      })
    }
  }

  const result = Array.from(groups.values())
  result.sort((a, b) => b.bestScore - a.bestScore)
  for (const group of result) group.occurrences.sort((a, b) => a.start - b.start)
  return result
}
