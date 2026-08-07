import { N_CLASSES } from './constants'

/**
 * The geo-temporal species filter.
 *
 * BirdNET ships a second, tiny model — "MData" — that takes latitude, longitude
 * and a week of the year and returns, for each of the 6522 classes, how likely
 * that species is to be *there, then*. BirdNET-Analyzer uses it to strike
 * implausible species off the report entirely.
 *
 * Two facts about it are not guessable and are the whole reason this file
 * exists rather than a couple of inline constants:
 *
 * 1. **The week runs 1 to 48, not 1 to 52.** Upstream splits every month into
 *    four (`birdnet/geo/inference/configs.py`). Feeding it an ISO week number
 *    would silently shift the season by up to a month — worst in spring and
 *    autumn, exactly when migration makes the answer change fastest.
 * 2. **It is a hard filter, not a weighting.** `invalid_mask = res <
 *    min_confidence` (`birdnet/geo/inference/session.py`). A species under the
 *    threshold is *removed*, not attenuated. Treating it as a multiplier would
 *    produce a different report from BirdNET's for the same inputs.
 *
 * Unlike the acoustic model, this one ends in its own sigmoid: the numbers that
 * come out are already probabilities and must not be passed through
 * `flatSigmoid`.
 */

/** Four weeks per month, twelve months. */
export const WEEKS_PER_YEAR = 48

/** Sentinel accepted by the model for "any time of year". */
export const WEEK_WHOLE_YEAR = -1

/**
 * BirdNET-Analyzer's default species-filter threshold (`analyze/core.py`).
 * Exposed in the UI because it is the single knob that decides how aggressive
 * the filter is.
 */
export const DEFAULT_SF_THRESH = 0.03

export interface GeoQuery {
  latitude: number
  longitude: number
  /** 1 to 48, or `WEEK_WHOLE_YEAR`. */
  week: number
}

/**
 * BirdNET's week number for a date.
 *
 * `(month - 1) * 4 + min(3, floor((day - 1) / 7)) + 1`. The `min(3, …)` is what
 * keeps a 31-day month from spilling into a fifth week: days 22-31 all land in
 * week 4 of their month.
 */
export function weekOfYear(date: Date): number {
  const month = date.getMonth() // 0-11
  const day = date.getDate() // 1-31
  return month * 4 + Math.min(3, Math.floor((day - 1) / 7)) + 1
}

/** e.g. "semaine 20 · mi-mai". Weeks are abstract; the month makes them legible. */
export function describeWeek(week: number): string {
  if (week === WEEK_WHOLE_YEAR) return "toute l'année"
  const months = [
    'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
    'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
  ]
  const month = Math.floor((week - 1) / 4)
  const quarter = (week - 1) % 4
  const when = ['début', 'mi-', 'fin', 'fin']
  const prefix = quarter === 1 ? 'mi-' : `${when[quarter]} `
  return `semaine ${week} · ${prefix}${months[month]}`
}

export function isValidLatitude(value: number): boolean {
  return Number.isFinite(value) && value >= -90 && value <= 90
}

export function isValidLongitude(value: number): boolean {
  return Number.isFinite(value) && value >= -180 && value <= 180
}

export function clampWeek(value: number): number {
  if (value === WEEK_WHOLE_YEAR) return WEEK_WHOLE_YEAR
  return Math.min(WEEKS_PER_YEAR, Math.max(1, Math.round(value)))
}

/**
 * Class indices the filter keeps, as a lookup by class index.
 *
 * A `Uint8Array` rather than a `Set`: it is consulted once per detection, and
 * 6522 bytes is nothing next to the 29 MB model that produced it.
 */
export function allowedMask(scores: Float32Array, threshold: number): Uint8Array {
  if (scores.length !== N_CLASSES) {
    throw new Error(`geo scores must have ${N_CLASSES} entries, got ${scores.length}`)
  }
  const mask = new Uint8Array(N_CLASSES)
  for (let i = 0; i < N_CLASSES; i++) mask[i] = scores[i] >= threshold ? 1 : 0
  return mask
}
