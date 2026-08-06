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
 * BirdNET ships 10 non-event classes among the 6522. They are real predictions,
 * not padding, and are useful for diagnosing a noisy recording — but they are not
 * birds, so callers can filter them out.
 */
const NON_EVENT_SCIENTIFIC = new Set([
  'Human vocal',
  'Human non-vocal',
  'Human whistle',
  'Noise',
  'Dog',
  'Engine',
  'Environmental',
  'Fireworks',
  'Gun',
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

  return lines.map((line, index) => {
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
