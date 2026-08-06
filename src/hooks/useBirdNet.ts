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
 * (measured on the reference fixture: 44 classes clear 0.005 in the busiest
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

export function useBirdNet() {
  const analyzerRef = useRef<BirdNetAnalyzer | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [state, setState] = useState<AnalysisState>(initial)

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      analyzerRef.current?.dispose()
      analyzerRef.current = null
    }
  }, [])

  const analyze = useCallback(async (file: File) => {
    abortRef.current?.abort()
    const abort = new AbortController()
    abortRef.current = abort

    analyzerRef.current ??= new BirdNetAnalyzer()
    const analyzer = analyzerRef.current

    setState({ ...initial, phase: 'loading-model' })

    try {
      const result = await analyzer.analyze(
        file,
        { minConfidence: ANALYSIS_FLOOR, overlap: 0, locale: 'fr', signal: abort.signal },
        {
          onModelProgress: (modelProgress) =>
            setState((s) => ({
              ...s,
              modelProgress,
              // The decode happens after the model is ready; showing "decoding"
              // while the 52 MB download runs would misreport what is slow.
              phase: modelProgress.phase === 'ready' ? 'decoding' : 'loading-model',
            })),
          onProgress: (progress) =>
            setState((s) => ({ ...s, phase: 'analyzing', progress })),
        },
      )

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
      if (error instanceof DOMException && error.name === 'AbortError') return
      setState((s) => ({
        ...s,
        phase: 'error',
        error: error instanceof Error ? error.message : String(error),
      }))
    }
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState(initial)
  }, [])

  return { state, analyze, reset, floor: ANALYSIS_FLOOR }
}
