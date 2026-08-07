import * as ort from 'onnxruntime-web'
import { MODEL_BASE_URL, N_CLASSES, ORT_WASM_PATH, WINDOW_SAMPLES } from './constants'
import type { ModelLoadProgress } from './types'

export interface GeoManifest {
  file: string
  sha256: string
  bytes: number
  classes: number
  defaultThreshold: number
}

export interface ModelManifest {
  file: string
  sha256: string
  bytes: number
  classes: number
  windowSamples: number
  sampleRate: number
  opset: number
  /** Present once `pnpm model:build` has exported the geo model too. */
  geo?: GeoManifest
}

const CACHE_NAME = 'birdnet-model-v1'

let configured = false

/**
 * Point ORT at the WASM binaries we copied into `public/ort/` and pin it to a
 * single thread.
 *
 * Multi-threaded ORT needs `SharedArrayBuffer`, which needs COOP/COEP headers,
 * which break third-party iframes and require server configuration this project
 * deliberately does not have (it is a static site). SIMD stays on — it is the
 * larger win and costs nothing.
 */
function configureRuntime(wasmPath: string = ORT_WASM_PATH): void {
  if (configured) return
  ort.env.wasm.wasmPaths = wasmPath
  ort.env.wasm.numThreads = 1
  ort.env.wasm.simd = true
  ort.env.logLevel = 'error'
  configured = true
}

/**
 * Fetch the model, reporting progress and caching the bytes.
 *
 * The file is ~52 MB, so the first load is a visible event and the user gets a
 * byte-accurate progress reading rather than an indeterminate spinner. Later
 * loads come from the Cache API keyed on the manifest digest, so a model rebuild
 * invalidates the entry instead of serving stale weights forever.
 */
async function fetchModel(
  url: string,
  digest: string,
  onProgress?: (progress: ModelLoadProgress) => void,
): Promise<ArrayBuffer> {
  const cacheKey = `${url}?sha256=${digest}`
  const caches = globalThis.caches

  if (caches) {
    const cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(cacheKey)
    if (hit) {
      const buffer = await hit.arrayBuffer()
      onProgress?.({
        phase: 'compiling',
        loaded: buffer.byteLength,
        total: buffer.byteLength,
        fromCache: true,
      })
      return buffer
    }
    // A new digest means every older entry is dead weight; 52 MB each.
    for (const key of await cache.keys()) {
      if (key.url.split('?')[0].endsWith(url.split('/').pop() ?? '')) await cache.delete(key)
    }
  }

  const response = await fetch(url)
  if (!response.ok) throw new Error(`could not fetch the model (HTTP ${response.status})`)

  const total = Number(response.headers.get('content-length') ?? 0)
  const reader = response.body?.getReader()

  let buffer: ArrayBuffer
  if (!reader) {
    // No streaming support: still correct, just without incremental progress.
    buffer = await response.arrayBuffer()
    onProgress?.({ phase: 'downloading', loaded: buffer.byteLength, total, fromCache: false })
  } else {
    const chunks: Uint8Array[] = []
    let loaded = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      onProgress?.({ phase: 'downloading', loaded, total, fromCache: false })
    }
    const merged = new Uint8Array(loaded)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.byteLength
    }
    buffer = merged.buffer
  }

  if (caches) {
    const cache = await caches.open(CACHE_NAME)
    await cache.put(
      cacheKey,
      new Response(buffer.slice(0), {
        headers: { 'content-type': 'application/octet-stream', 'content-length': String(buffer.byteLength) },
      }),
    )
  }

  onProgress?.({ phase: 'compiling', loaded: buffer.byteLength, total, fromCache: false })
  return buffer
}

export interface LoadedModel {
  session: ort.InferenceSession
  manifest: ModelManifest
}

/**
 * Load the BirdNET session.
 *
 * The manifest is checked against the constants the pipeline is written for: a
 * model with a different window length or class count would otherwise produce
 * confident nonsense rather than an error.
 */
export async function loadModel(options: {
  baseUrl?: string
  wasmPath?: string
  onProgress?: (progress: ModelLoadProgress) => void
} = {}): Promise<LoadedModel> {
  const baseUrl = options.baseUrl ?? MODEL_BASE_URL
  configureRuntime(options.wasmPath)

  const manifestResponse = await fetch(`${baseUrl}/manifest.json`)
  if (!manifestResponse.ok) {
    throw new Error(
      `no model manifest at ${baseUrl}/manifest.json — run \`pnpm model:build\` first`,
    )
  }
  const manifest = (await manifestResponse.json()) as ModelManifest

  if (manifest.classes !== N_CLASSES || manifest.windowSamples !== WINDOW_SAMPLES) {
    throw new Error(
      `model manifest does not match this build: ${manifest.classes} classes / ` +
        `${manifest.windowSamples} samples, expected ${N_CLASSES} / ${WINDOW_SAMPLES}`,
    )
  }

  const bytes = await fetchModel(`${baseUrl}/${manifest.file}`, manifest.sha256, options.onProgress)

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  options.onProgress?.({
    phase: 'ready',
    loaded: bytes.byteLength,
    total: bytes.byteLength,
    fromCache: false,
  })

  return { session, manifest }
}

/** Run one 3-second window. Returns raw logits — apply `flatSigmoid` to score them. */
export async function inferWindow(
  session: ort.InferenceSession,
  window: Float32Array,
): Promise<Float32Array> {
  if (window.length !== WINDOW_SAMPLES) {
    throw new Error(`window must be ${WINDOW_SAMPLES} samples, got ${window.length}`)
  }
  const tensor = new ort.Tensor('float32', window, [1, WINDOW_SAMPLES])
  const output = await session.run({ input: tensor })
  return output.output.data as Float32Array
}

/**
 * Load the geo-temporal (MData) session.
 *
 * Kept separate from `loadModel` and loaded lazily, because it is 29 MB for a
 * feature most sessions never turn on. It shares the same on-disk cache and the
 * same digest-keying, so enabling the filter costs one download ever.
 */
export async function loadGeoModel(options: {
  baseUrl?: string
  wasmPath?: string
  onProgress?: (progress: ModelLoadProgress) => void
} = {}): Promise<ort.InferenceSession> {
  const baseUrl = options.baseUrl ?? MODEL_BASE_URL
  configureRuntime(options.wasmPath)

  const response = await fetch(`${baseUrl}/manifest.json`)
  if (!response.ok) throw new Error(`no model manifest at ${baseUrl}/manifest.json`)
  const manifest = (await response.json()) as ModelManifest

  if (!manifest.geo) {
    throw new Error(
      'this build has no geo model — re-run `pnpm model:build` to export it alongside the acoustic one',
    )
  }
  if (manifest.geo.classes !== N_CLASSES) {
    throw new Error(
      `geo model has ${manifest.geo.classes} classes, expected ${N_CLASSES} — ` +
        'it would be filtering against the wrong label list',
    )
  }

  const bytes = await fetchModel(
    `${baseUrl}/${manifest.geo.file}`,
    manifest.geo.sha256,
    options.onProgress,
  )
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  options.onProgress?.({
    phase: 'ready',
    loaded: bytes.byteLength,
    total: bytes.byteLength,
    fromCache: false,
  })
  return session
}

/**
 * Score every class for one place and time.
 *
 * Returns probabilities directly — this model ends in its own sigmoid, so
 * passing the result through `flatSigmoid` would apply one twice.
 */
export async function inferGeo(
  session: ort.InferenceSession,
  latitude: number,
  longitude: number,
  week: number,
): Promise<Float32Array> {
  const tensor = new ort.Tensor('float32', Float32Array.from([latitude, longitude, week]), [1, 3])
  const output = await session.run({ input: tensor })
  return output.output.data as Float32Array
}
