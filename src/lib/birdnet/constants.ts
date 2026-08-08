/**
 * BirdNET v2.4 constants.
 *
 * These are the model's contract, not tunables. They are asserted at build time
 * by `scripts/convert_to_onnx.py` and re-checked at load time against
 * `public/models/manifest.json`.
 */

/** The model was trained at 48 kHz. Anything else must be resampled first. */
export const SAMPLE_RATE = 48_000

/** 3 seconds at 48 kHz. The input tensor is exactly `[1, 144000]`. */
export const WINDOW_SAMPLES = 144_000

export const WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE

/** 6522 classes, including 10 non-event classes (noise, human, dog, ...). */
export const N_CLASSES = 6522

export const MODEL_INPUT = 'input'
export const MODEL_OUTPUT = 'output'

/** BirdNET-Analyzer's default minimum confidence. */
export const DEFAULT_MIN_CONFIDENCE = 0.25

/**
 * BirdNET-Analyzer's sigmoid sensitivity, valid range [0.5, 1.5].
 * 1.0 leaves the activation as the plain logistic function.
 */
export const DEFAULT_SENSITIVITY = 1.0

/** BirdNET clamps logits before the sigmoid; see `flatSigmoid`. */
export const SIGMOID_CLIP = 15.0

/** Overlap between consecutive windows, in seconds. BirdNET allows [0, 2.9]. */
export const MAX_OVERLAP_SECONDS = 2.9

/**
 * Runtime asset roots, resolved against the deployment's base path.
 *
 * `import.meta.env.BASE_URL` is `/` in development and at the root of a domain,
 * and `/whosyourbirdy/` when the site is served from a GitHub Pages project
 * subpath. These two are fetched at runtime rather than imported, so the bundler
 * never rewrites them — hard-coding a leading slash would 404 everything the
 * moment the site moved off a domain root.
 */
export const MODEL_BASE_URL = `${import.meta.env.BASE_URL}models`

export const ORT_WASM_PATH = `${import.meta.env.BASE_URL}ort/`

/**
 * Required by CC BY-NC-SA 4.0 and shown permanently in the UI, not hidden in an
 * "about" dialog.
 */
export const ATTRIBUTION =
  'Powered by BirdNET — K. Lisa Yang Center for Conservation Bioacoustics, ' +
  'Cornell Lab of Ornithology & Chemnitz University of Technology'

export const ATTRIBUTION_URL = 'https://birdnet.cornell.edu/'

export const MODEL_LICENSE = 'CC BY-NC-SA 4.0'
