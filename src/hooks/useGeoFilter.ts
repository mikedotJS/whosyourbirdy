import { useCallback, useEffect, useRef, useState } from 'react'
import type { BirdNetAnalyzer } from '../lib/birdnet/analyze'
import {
  DEFAULT_SF_THRESH,
  allowedMask,
  clampWeek,
  isValidLatitude,
  isValidLongitude,
  weekOfYear,
  type GeoQuery,
} from '../lib/birdnet/geo'
import type { ModelLoadProgress } from '../lib/birdnet/types'

export interface GeoSettings extends GeoQuery {
  enabled: boolean
  /** Species below this geo probability are removed. BirdNET's own default: 0.03. */
  threshold: number
}

export interface GeoState {
  settings: GeoSettings
  /** Which classes survive, indexed by class index. Null until scored. */
  mask: Uint8Array | null
  /** Raw geo probability per class, so a masked species can show its score. */
  scores: Float32Array | null
  loading: boolean
  progress: ModelLoadProgress | null
  error: string | null
}

const initialSettings = (): GeoSettings => ({
  enabled: false,
  latitude: 48.85,
  longitude: 2.35,
  week: weekOfYear(new Date()),
  threshold: DEFAULT_SF_THRESH,
})

/**
 * The geo-temporal filter's state.
 *
 * Scoring is debounced and guarded by a run id, for the same reason the analysis
 * is: dragging the week slider fires a request per step, they finish out of
 * order, and without the guard the mask ends up describing whatever request
 * happened to land last rather than the settings on screen.
 */
export function useGeoFilter(analyzer: BirdNetAnalyzer | null) {
  const [state, setState] = useState<GeoState>({
    settings: initialSettings(),
    mask: null,
    scores: null,
    loading: false,
    progress: null,
    error: null,
  })
  const runIdRef = useRef(0)

  const update = useCallback((patch: Partial<GeoSettings>) => {
    setState((s) => ({ ...s, settings: { ...s.settings, ...patch } }))
  }, [])

  const { enabled, latitude, longitude, week } = state.settings

  useEffect(() => {
    if (!analyzer || !enabled) return
    if (!isValidLatitude(latitude) || !isValidLongitude(longitude)) return

    const runId = ++runIdRef.current
    // 250 ms: long enough that dragging the week slider does not queue an
    // inference per step, short enough that typing coordinates still feels live.
    const timer = window.setTimeout(() => {
      setState((s) => ({ ...s, loading: true, error: null }))
      analyzer
        .geoScores(latitude, longitude, clampWeek(week), (progress) => {
          if (runIdRef.current === runId) setState((s) => ({ ...s, progress }))
        })
        .then((scores) => {
          if (runIdRef.current !== runId) return
          setState((s) => ({
            ...s,
            scores,
            mask: allowedMask(scores, s.settings.threshold),
            loading: false,
            progress: null,
          }))
        })
        .catch((error: unknown) => {
          if (runIdRef.current !== runId) return
          setState((s) => ({
            ...s,
            loading: false,
            progress: null,
            error: error instanceof Error ? error.message : String(error),
          }))
        })
    }, 250)

    return () => window.clearTimeout(timer)
  }, [analyzer, enabled, latitude, longitude, week])

  // The threshold only re-slices scores we already have — no inference, so
  // moving it is instant, exactly like the confidence slider.
  useEffect(() => {
    setState((s) =>
      s.scores ? { ...s, mask: allowedMask(s.scores, s.settings.threshold) } : s,
    )
  }, [state.settings.threshold])

  /**
   * Ask the browser where we are. Called only from a click — the Geolocation
   * prompt is never raised on load, because a page that asks for your position
   * before you have done anything is a page you close.
   */
  const locate = useCallback((): Promise<void> => {
    if (!('geolocation' in navigator)) {
      setState((s) => ({ ...s, error: "Ce navigateur n'expose pas la géolocalisation." }))
      return Promise.resolve()
    }
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          update({
            latitude: Number(position.coords.latitude.toFixed(4)),
            longitude: Number(position.coords.longitude.toFixed(4)),
          })
          resolve()
        },
        (error) => {
          setState((s) => ({
            ...s,
            error:
              error.code === error.PERMISSION_DENIED
                ? 'Position refusée — saisissez les coordonnées à la main.'
                : "La position n'a pas pu être obtenue.",
          }))
          resolve()
        },
        { enableHighAccuracy: false, timeout: 10_000, maximumAge: 600_000 },
      )
    })
  }, [update])

  return { ...state, update, locate }
}
