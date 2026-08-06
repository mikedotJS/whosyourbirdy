import { MODEL_BASE_URL, N_CLASSES } from './constants'

export type Locale = 'fr' | 'en'

export interface Species {
  /** Row index in the label file, i.e. the model's class index. */
  index: number
  /** e.g. "Turdus merula" */
  scientificName: string
  /** e.g. "Merle noir" (fr) or "Eurasian Blackbird" (en) */
  commonName: string
}

/**
 * The non-event classes among the 6522. They are real predictions, not padding,
 * and are useful for diagnosing a noisy recording — but they are not birds, so
 * callers can filter them out.
 *
 * Upstream documentation says "10 non-event classes"; the label file actually
 * contains 11. This list was read off the shipped labels rather than the prose,
 * and `assertNonEventsPresent` re-checks it at parse time — a name that stops
 * matching would otherwise make `excludeNonEvents` silently filter nothing.
 * These entries are identical in every locale (they are not translated).
 */
const NON_EVENT_SCIENTIFIC = new Set([
  'Dog',
  'Engine',
  'Environmental',
  'Fireworks',
  'Gun',
  'Human non-vocal',
  'Human vocal',
  'Human whistle',
  'Noise',
  'Power tools',
  'Siren',
])

export function isNonEvent(species: Species): boolean {
  return NON_EVENT_SCIENTIFIC.has(species.scientificName)
}

/**
 * Parse a BirdNET label file.
 *
 * Format is one class per line, `Scientific name_Common name`, in model output
 * order. The common name may itself contain underscores in some locales, so we
 * split on the first separator only.
 */
export function parseLabels(text: string): Species[] {
  const lines = text.split('\n')
  // Tolerate a trailing newline; anything else short is a corrupt file.
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop()

  if (lines.length !== N_CLASSES) {
    throw new Error(`label file has ${lines.length} entries, expected ${N_CLASSES}`)
  }

  const species = lines.map((line, index) => {
    const sep = line.indexOf('_')
    if (sep === -1) {
      throw new Error(`label ${index} is missing its "_" separator: ${line}`)
    }
    return {
      index,
      scientificName: line.slice(0, sep).trim(),
      commonName: line.slice(sep + 1).trim(),
    }
  })

  assertNonEventsPresent(species)
  return species
}

/** Fail loudly if the non-event names drift, rather than filtering nothing. */
function assertNonEventsPresent(species: Species[]): void {
  const found = new Set(
    species.filter((s) => NON_EVENT_SCIENTIFIC.has(s.scientificName)).map((s) => s.scientificName),
  )
  if (found.size !== NON_EVENT_SCIENTIFIC.size) {
    const missing = [...NON_EVENT_SCIENTIFIC].filter((name) => !found.has(name))
    throw new Error(`non-event classes missing from the label file: ${missing.join(', ')}`)
  }
}

const cache = new Map<Locale, Promise<Species[]>>()

/** Fetch and parse the label list for a locale, memoised per session. */
export function loadLabels(locale: Locale = 'fr', baseUrl = MODEL_BASE_URL): Promise<Species[]> {
  const existing = cache.get(locale)
  if (existing) return existing

  const pending = (async () => {
    const response = await fetch(`${baseUrl}/labels_${locale}.txt`)
    if (!response.ok) {
      throw new Error(`could not load labels_${locale}.txt (HTTP ${response.status})`)
    }
    return parseLabels(await response.text())
  })().catch((error: unknown) => {
    // Do not memoise a failure: a transient network error should not poison the
    // rest of the session.
    cache.delete(locale)
    throw error
  })

  cache.set(locale, pending)
  return pending
}
