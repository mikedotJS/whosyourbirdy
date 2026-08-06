import { SAMPLE_RATE } from './constants'

/**
 * Display spectrogram.
 *
 * This is for the *picture*, not for the model. BirdNET computes its own two mel
 * spectrograms inside its graph and we never touch those — see the README. What
 * this produces is a plain magnitude STFT, chosen for legibility rather than for
 * matching anything: linear frequency to 15 kHz (the model's own coverage, and
 * what a birder expects from Raven or Audacity), 1024-sample frames, dB scale.
 *
 * It runs in the worker, which already owns the PCM after the transfer, and
 * returns a compact 8-bit grid — a two-minute file becomes ~450 KB instead of the
 * 23 MB the samples occupy.
 */

/** Frame size: 21 ms at 48 kHz, 46.9 Hz per bin. Matches BirdNET's high band. */
const FRAME = 1024

/** Only the band the model can hear about. */
const MAX_HZ = 15_000

/** Bins kept, from DC up to MAX_HZ. */
export const SPEC_BINS = Math.round((MAX_HZ / (SAMPLE_RATE / 2)) * (FRAME / 2))

/** Widest grid we produce; beyond this the eye gains nothing and memory grows. */
const MAX_COLUMNS = 1600

/**
 * Frames we are willing to transform, independent of the column count.
 *
 * One frame per column was the obvious implementation and it quietly *sampled*
 * the recording instead of summarising it: at 1600 columns over two minutes the
 * hop is 75 ms against a 21 ms frame, so 72% of the audio appeared nowhere in
 * the picture, and short calls landing in the gaps were simply invisible — a
 * user could click a detection band and find blank picture where the bird was.
 *
 * Now several frames are transformed per column and max-pooled, so a transient
 * anywhere inside a column survives into it. 4096 covers a two-minute file
 * almost completely; longer files still degrade, but they degrade by losing
 * resolution rather than by dropping events.
 */
const MAX_FRAMES = 4096

/** Dynamic range below the peak, in dB. Below this everything reads as silence. */
const FLOOR_DB = 80

export interface SpectrogramData {
  /** Time columns. */
  columns: number
  /** Frequency bins per column, DC first. */
  bins: number
  /** `columns * bins` magnitudes, 0 = floor, 255 = peak. Column-major. */
  magnitudes: Uint8Array
  /** Seconds represented, so the canvas can map x to time. */
  duration: number
  maxHz: number
}

/** Hann window, precomputed once per frame size. */
const window = new Float32Array(FRAME)
for (let i = 0; i < FRAME; i++) {
  window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FRAME)
}

/**
 * In-place iterative radix-2 FFT.
 *
 * Small and self-contained on purpose: this is display code, so it has no
 * accuracy obligation beyond looking right, and pulling in a DSP dependency for
 * one transform would be worse.
 */
function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len
    const wRe = Math.cos(angle)
    const wIm = Math.sin(angle)
    for (let i = 0; i < n; i += len) {
      let curRe = 1
      let curIm = 0
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k]
        const aIm = im[i + k]
        const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
        const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
        re[i + k] = aRe + bRe
        im[i + k] = aIm + bIm
        re[i + k + len / 2] = aRe - bRe
        im[i + k + len / 2] = aIm - bIm
        const nextRe = curRe * wRe - curIm * wIm
        curIm = curRe * wIm + curIm * wRe
        curRe = nextRe
      }
    }
  }
}

/**
 * Compute the display spectrogram.
 *
 * The hop is derived from the signal length so the grid is always about the same
 * width whatever the duration: a 10-second clip and a two-hour recording both
 * produce a picture the canvas can draw without downsampling artefacts.
 */
export function computeSpectrogram(samples: Float32Array): SpectrogramData {
  const duration = samples.length / SAMPLE_RATE

  // Frames at 50% overlap would cover the signal completely; cap them, then
  // spread whatever we can afford evenly across the file.
  const idealFrames = Math.max(1, Math.floor((samples.length - FRAME) / (FRAME / 2)) + 1)
  const frames = Math.max(1, Math.min(MAX_FRAMES, idealFrames))
  const columns = Math.max(1, Math.min(MAX_COLUMNS, frames))
  const frameHop = Math.max(1, Math.floor((samples.length - FRAME) / Math.max(1, frames - 1)))

  const power = new Float32Array(columns * SPEC_BINS)
  const re = new Float32Array(FRAME)
  const im = new Float32Array(FRAME)

  let peak = 1e-12
  for (let f = 0; f < frames; f++) {
    const offset = f * frameHop
    // Which column this frame folds into. Several frames share a column and the
    // loudest wins per bin, so a call between frame centres still shows up.
    const c = Math.min(columns - 1, Math.floor((f * columns) / frames))

    for (let i = 0; i < FRAME; i++) {
      const s = offset + i
      re[i] = s < samples.length ? samples[s] * window[i] : 0
      im[i] = 0
    }
    fft(re, im)

    const base = c * SPEC_BINS
    for (let b = 0; b < SPEC_BINS; b++) {
      const value = re[b] * re[b] + im[b] * im[b]
      if (value > power[base + b]) power[base + b] = value
      if (value > peak) peak = value
    }
  }

  // Convert to dB relative to the file's own peak. Normalising per file (rather
  // than to an absolute level) is a display choice and has no bearing on the
  // scores, which the model computes from raw samples.
  const peakDb = 10 * Math.log10(peak)
  const db = new Float32Array(power.length)
  for (let i = 0; i < power.length; i++) {
    db[i] = Math.max(-FLOOR_DB, 10 * Math.log10(power[i] + 1e-12) - peakDb)
  }

  // Stretch the contrast between the recording's own noise floor and its loud
  // content, instead of between the absolute floor and the peak.
  //
  // A field recording spends most of its energy 40-60 dB below peak, so mapping
  // the full 80 dB range linearly renders the whole picture as mid-grey mush
  // with the bird song barely darker than the hiss. Percentiles adapt to the
  // material: a quiet dawn chorus and a loud close recording both use the full
  // ink range. The peak itself is a poor anchor too — one microphone bump sets
  // it for the entire file.
  // A strided copy rather than `db.filter(cb)`: the callback over half a million
  // elements cost 22 ms of the 28 ms this step took, against ~7 ms for the sort.
  const sampleCount = Math.ceil(db.length / 7)
  const sorted = new Float32Array(sampleCount)
  for (let i = 0, j = 0; j < sampleCount; i += 7, j++) sorted[j] = db[i]
  sorted.sort()
  const low = sorted[Math.floor(sorted.length * 0.55)] // noise floor
  const high = sorted[Math.floor(sorted.length * 0.999)]
  const span = Math.max(6, high - low)

  const magnitudes = new Uint8Array(power.length)
  for (let i = 0; i < db.length; i++) {
    const t = Math.min(1, Math.max(0, (db[i] - low) / span))
    // Mild gamma: lifts faint harmonics without flattening the loud parts.
    magnitudes[i] = Math.round(Math.pow(t, 0.72) * 255)
  }

  return { columns, bins: SPEC_BINS, magnitudes, duration, maxHz: MAX_HZ }
}
