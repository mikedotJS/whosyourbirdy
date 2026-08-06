/// <reference lib="webworker" />
import { inferWindow, loadModel } from './model'
import { flatSigmoid } from './sigmoid'
import { planWindows, sliceWindow } from './windows'
import type { WorkerRequest, WorkerResponse } from './types'
import type * as ort from 'onnxruntime-web'

/**
 * Inference worker.
 *
 * 6522 classes times N windows is far too much work for the main thread: a
 * two-minute file is 40 inferences, and the UI has to stay responsive enough to
 * animate the analysis front while they run. The worker emits one message per
 * window so results stream in rather than landing all at once.
 */

declare const self: DedicatedWorkerGlobalScope

let session: ort.InferenceSession | null = null
let loading: Promise<void> | null = null
const cancelled = new Set<number>()

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer)
}

async function ensureModel(modelUrl: string, wasmPath: string): Promise<void> {
  if (session) return
  loading ??= (async () => {
    const loaded = await loadModel({
      baseUrl: modelUrl,
      wasmPath,
      onProgress: (progress) => post({ type: 'load-progress', progress }),
    })
    session = loaded.session
  })().catch((error: unknown) => {
    loading = null // let the caller retry rather than wedging the worker
    throw error
  })
  await loading
}

/**
 * Collect the classes above threshold for one window.
 *
 * Thresholding here rather than in the UI keeps 6522 floats per window off the
 * message channel; `topKPerWindow` bounds the worst case when the threshold is
 * set very low.
 */
function selectDetections(
  scores: Float32Array,
  minConfidence: number,
  allowedClasses: Int32Array | null,
  topK: number,
): { classes: Int32Array; scores: Float32Array; truncated: boolean } {
  const hits: number[] = []

  if (allowedClasses) {
    for (let i = 0; i < allowedClasses.length; i++) {
      const c = allowedClasses[i]
      if (scores[c] >= minConfidence) hits.push(c)
    }
  } else {
    for (let c = 0; c < scores.length; c++) {
      if (scores[c] >= minConfidence) hits.push(c)
    }
  }

  hits.sort((a, b) => scores[b] - scores[a])
  const kept = hits.length > topK ? hits.slice(0, topK) : hits

  const classes = new Int32Array(kept.length)
  const values = new Float32Array(kept.length)
  for (let i = 0; i < kept.length; i++) {
    classes[i] = kept[i]
    values[i] = scores[kept[i]]
  }
  return { classes, scores: values, truncated: hits.length > topK }
}

async function analyze(request: Extract<WorkerRequest, { type: 'analyze' }>): Promise<void> {
  if (!session) throw new Error('worker received analyze before init')

  const { requestId, samples, overlap, minConfidence, sensitivity, allowedClasses, topKPerWindow } =
    request
  const windows = planWindows(samples.length, overlap)
  const timings = new Float32Array(windows.length)
  const started = performance.now()

  for (const window of windows) {
    if (cancelled.has(requestId)) {
      cancelled.delete(requestId)
      post({ type: 'cancelled', requestId })
      return
    }

    const t0 = performance.now()
    const logits = await inferWindow(session, sliceWindow(samples, window.offsetSamples))
    const inferenceMs = performance.now() - t0
    timings[window.index] = inferenceMs

    const scores = flatSigmoid(logits, sensitivity)
    const picked = selectDetections(scores, minConfidence, allowedClasses, topKPerWindow)

    post(
      {
        type: 'window',
        requestId,
        windowIndex: window.index,
        total: windows.length,
        start: window.start,
        end: window.end,
        classes: picked.classes,
        scores: picked.scores,
        inferenceMs,
        truncated: picked.truncated,
      },
      [picked.classes.buffer, picked.scores.buffer],
    )
  }

  // A cancel that lands after the last window would otherwise sit in the set for
  // the lifetime of the worker.
  cancelled.delete(requestId)
  post({ type: 'done', requestId, windowCount: windows.length, totalMs: performance.now() - started, timings },
    [timings.buffer])
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data
  try {
    switch (message.type) {
      case 'init':
        await ensureModel(message.modelUrl, message.wasmPath)
        post({ type: 'ready' })
        break
      case 'analyze':
        await analyze(message)
        break
      case 'cancel':
        cancelled.add(message.requestId)
        break
    }
  } catch (error) {
    post({
      type: 'error',
      requestId: message.type === 'analyze' ? message.requestId : undefined,
      message: error instanceof Error ? error.message : String(error),
    })
  }
}
