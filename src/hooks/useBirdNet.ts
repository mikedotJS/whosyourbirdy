import { useCallback, useEffect, useRef, useState } from 'react'
import { BirdNetAnalyzer } from '../lib/birdnet/analyze'
import type { AnalysisProgress, Detection, ModelLoadProgress } from '../lib/birdnet/types'

/**
 * Analysis floor.
 *
 * We run the model once at a low threshold and filter in the UI, so moving the
 * confidence slider is instant instead of re-running every window. 0.01 is low
 * enough that the slider never reveals a species the analysis did not keep, and
 * high enough to stay well under the worker's 64-detections-per-window cap
 * (measured on the reference fixture: 27 classes clear 0.01 in the busiest
 * window). `truncatedWindows` reports it if that ever stops being true.
 */
const ANALYSIS_FLOOR = 0.01

export type Phase = 'idle' | 'loading-model' | 'decoding' | 'analyzing' | 'done' | 'error'

export interface AnalysisState {
  phase: Phase
  detections: Detection[]
  modelProgress: ModelLoadProgress | null
  progress: AnalysisProgress | null
  duration: number
  medianInferenceMs: number
  totalMs: number
  truncatedWindows: number
  error: string | null
}

const initial: AnalysisState = {
  phase: 'idle',
  detections: [],
  modelProgress: null,
  progress: null,
  duration: 0,
  medianInferenceMs: 0,
  totalMs: 0,
  truncatedWindows: 0,
  error: null,
}

/** Turn a raw failure into something a French-speaking user can act on. */
function describeError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)

  if (/decode/i.test(message)) {
    return (
      "Ce fichier n'a pas pu être décodé. Vérifiez qu'il s'agit bien d'un enregistrement " +
      'audio (wav, mp3, flac, m4a, ogg) et qu\'il n\'est pas vide ou corrompu.'
    )
  }
  if (/fetch|network|HTTP/i.test(message)) {
    return (
      "Le modèle n'a pas pu être téléchargé. Vérifiez votre connexion, puis réessayez — " +
      'il fait environ 52 Mo et n\'est téléchargé qu\'une fois.'
    )
  }
  if (/backend/i.test(message)) {
    return (
      "Le moteur d'inférence n'a pas pu démarrer. Si vous hébergez ce site vous-même, " +
      'vérifiez que les fichiers /ort/*.mjs sont servis avec un type MIME JavaScript.'
    )
  }
  return message
}

export function useBirdNet() {
  const analyzerRef = useRef<BirdNetAnalyzer | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  /**
   * Identifies the analysis the UI is currently showing.
   *
   * Every state write is gated on it. Cancellation is best-effort — the worker
   * may already be mid-window — so without this guard a run the user abandoned
   * could still deliver progress and results, and they would land under whatever
   * file is on screen now. That is exactly how 24 bird detections once ended up
   * displayed for a 29-byte text file.
   */
  const runIdRef = useRef(0)
  const [state, setState] = useState<AnalysisState>(initial)

  useEffect(() => {
    return () => {
      runIdRef.current++
      abortRef.current?.abort()
      analyzerRef.current?.dispose()
      analyzerRef.current = null
    }
  }, [])

  const analyze = useCallback(async (file: File) => {
    const runId = ++runIdRef.current
    const isCurrent = () => runIdRef.current === runId

    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort

    analyzerRef.current ??= new BirdNetAnalyzer()
    const analyzer = analyzerRef.current

    // Skip the "downloading" phase when the model is already resident, otherwise
    // the second file sits on an indeterminate "Préparation…" for its whole decode.
    setState({ ...initial, phase: analyzer.isReady ? 'decoding' : 'loading-model' })

    try {
      const result = await analyzer.analyze(
        file,
        { minConfidence: ANALYSIS_FLOOR, overlap: 0, locale: 'fr', signal: abort.signal },
        {
          onModelProgress: (modelProgress) => {
            if (!isCurrent()) return
            setState((s) => ({
              ...s,
              modelProgress,
              phase: modelProgress.phase === 'ready' ? 'decoding' : 'loading-model',
            }))
          },
          onProgress: (progress) => {
            if (!isCurrent()) return
            setState((s) => ({ ...s, phase: 'analyzing', progress }))
          },
        },
      )

      if (!isCurrent()) return
      setState((s) => ({
        ...s,
        phase: 'done',
        detections: result.detections,
        duration: result.duration,
        medianInferenceMs: result.medianInferenceMs,
        totalMs: result.totalMs,
        truncatedWindows: result.truncatedWindows,
      }))
    } catch (error) {
      if (!isCurrent()) return
      if (error instanceof DOMException && error.name === 'AbortError') return
      setState((s) => ({ ...s, phase: 'error', error: describeError(error) }))
    }
  }, [])

  const reset = useCallback(() => {
    // Bump first: a run in flight must not be able to write after this.
    runIdRef.current++
    abortRef.current?.abort()
    analyzerRef.current?.cancelAll()
    setState(initial)
  }, [])

  return { state, analyze, reset, floor: ANALYSIS_FLOOR }
}
