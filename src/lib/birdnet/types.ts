import type { Species } from './labels'

export interface Detection {
  species: Species
  /** Confidence in [0, 1], after the flat sigmoid. */
  score: number
  /** Window start in seconds. */
  start: number
  /** Window end in seconds. */
  end: number
  /** Index of the window this detection came from. */
  windowIndex: number
}

export interface AnalyzeOptions {
  /** Overlap between windows in seconds, [0, 2.9]. Default 0. */
  overlap?: number
  /** Minimum confidence to report. Default 0.25, BirdNET's own default. */
  minConfidence?: number
  /** Sigmoid sensitivity, [0.5, 1.5]. Default 1.0. */
  sensitivity?: number
  /** Label language. Default 'fr'. */
  locale?: 'fr' | 'en'
  /** Drop BirdNET's 10 non-bird classes (noise, dog, siren, ...). Default true. */
  excludeNonEvents?: boolean
  /** Abort an in-flight analysis. */
  signal?: AbortSignal
}

export type ModelLoadPhase = 'idle' | 'downloading' | 'compiling' | 'ready'

export interface ModelLoadProgress {
  phase: ModelLoadPhase
  /** Bytes received so far. */
  loaded: number
  /** Total bytes, or 0 when the server sends no content-length. */
  total: number
  /** True when the model came from the Cache API instead of the network. */
  fromCache: boolean
}

export interface AnalysisProgress {
  /** Windows completed so far. */
  completed: number
  /** Total windows planned for this file. */
  total: number
  /** End of the analysed region in seconds — the "analysis front". */
  seconds: number
}

/** Per-window result, streamed as the analysis advances. */
export interface WindowResult {
  windowIndex: number
  start: number
  end: number
  detections: Detection[]
  /** True when the per-window cap discarded detections above the threshold. */
  truncated: boolean
  /** Wall-clock inference time for this window, in milliseconds. */
  inferenceMs: number
}

export interface AnalysisResult {
  detections: Detection[]
  /**
   * Windows where the per-window cap discarded detections that cleared the
   * threshold. Non-zero means the list is incomplete — surface it rather than
   * quietly showing a short list.
   */
  truncatedWindows: number
  windowCount: number
  duration: number
  /** Median per-window inference time, in milliseconds. */
  medianInferenceMs: number
  totalMs: number
}

/** Messages the worker accepts. */
export type WorkerRequest =
  | { type: 'init'; modelUrl: string; wasmPath: string }
  | {
      type: 'analyze'
      requestId: number
      samples: Float32Array
      overlap: number
      minConfidence: number
      sensitivity: number
      /** Class indices to report; `null` means all of them. */
      allowedClasses: Int32Array | null
      topKPerWindow: number
    }
  | { type: 'cancel'; requestId: number }
  /**
   * Score every class for a place and a week. Loads the 29 MB geo model on
   * first use — a session that never turns the filter on never pays for it.
   */
  | { type: 'geo'; requestId: number; latitude: number; longitude: number; week: number }

/** Messages the worker emits. */
export type WorkerResponse =
  | { type: 'load-progress'; progress: ModelLoadProgress }
  | {
      /**
       * Sent before the first window so the picture is on screen while the
       * analysis front sweeps across it.
       */
      type: 'spectrogram'
      requestId: number
      columns: number
      bins: number
      magnitudes: Uint8Array
      duration: number
      maxHz: number
    }
  | { type: 'ready' }
  | {
      type: 'window'
      requestId: number
      windowIndex: number
      total: number
      start: number
      end: number
      /** Parallel arrays: class index and score, already thresholded. */
      classes: Int32Array
      scores: Float32Array
      inferenceMs: number
      /** True when more classes cleared the threshold than `topKPerWindow` allowed. */
      truncated: boolean
    }
  | {
      type: 'done'
      requestId: number
      windowCount: number
      totalMs: number
      timings: Float32Array
    }
  | { type: 'cancelled'; requestId: number }
  /** Download progress for the geo model, kept apart from the acoustic one so
      the UI can say which of the two it is waiting on. */
  | { type: 'geo-progress'; requestId: number; progress: ModelLoadProgress }
  /** One probability per class, already in [0, 1] — this model has its own sigmoid. */
  | { type: 'geo-scores'; requestId: number; scores: Float32Array }
  | { type: 'error'; requestId?: number; message: string }
