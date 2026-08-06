import { MAX_OVERLAP_SECONDS, SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_SECONDS } from './constants'

export interface AudioWindow {
  /** Zero-based index in the analysis order. */
  index: number
  /** First sample of the window in the source signal. */
  offsetSamples: number
  /** Window start in seconds. */
  start: number
  /** Window end in seconds. Always `start + 3`, even when zero-padded. */
  end: number
  /** True when the source ran out and the tail was zero-padded. */
  padded: boolean
}

/**
 * Plan the 3-second windows for a signal, mirroring BirdNET-Analyzer.
 *
 * The hop is `144000 - overlap * 48000`; with the default overlap of 0 the
 * windows tile the file end to end. The last window is kept and zero-padded
 * rather than dropped, so a detection in the final seconds is not lost — this
 * matches upstream, which pads the tail the same way.
 *
 * @param totalSamples length of the 48 kHz mono signal
 * @param overlapSeconds overlap between consecutive windows, in [0, 2.9]
 */
export function planWindows(totalSamples: number, overlapSeconds = 0): AudioWindow[] {
  if (!Number.isFinite(overlapSeconds) || overlapSeconds < 0 || overlapSeconds > MAX_OVERLAP_SECONDS) {
    throw new RangeError(
      `overlap must be within [0, ${MAX_OVERLAP_SECONDS}] seconds, got ${overlapSeconds}`,
    )
  }
  if (totalSamples <= 0) return []

  const hop = WINDOW_SAMPLES - Math.round(overlapSeconds * SAMPLE_RATE)
  // The range check above keeps overlap below one full window, so hop >= 1.

  const windows: AudioWindow[] = []
  for (let offset = 0, index = 0; offset < totalSamples; offset += hop, index++) {
    windows.push({
      index,
      offsetSamples: offset,
      start: offset / SAMPLE_RATE,
      end: offset / SAMPLE_RATE + WINDOW_SECONDS,
      padded: offset + WINDOW_SAMPLES > totalSamples,
    })
  }
  return windows
}

/**
 * Copy one window out of the signal, zero-padding a short tail.
 *
 * Always returns a fresh `Float32Array` of exactly `WINDOW_SAMPLES`: the tensor
 * handed to ORT must own its buffer, and a subarray view of the source would
 * both alias and come up short at the end of the file.
 */
export function sliceWindow(samples: Float32Array, offsetSamples: number): Float32Array {
  const window = new Float32Array(WINDOW_SAMPLES)
  const available = Math.min(WINDOW_SAMPLES, samples.length - offsetSamples)
  if (available > 0) {
    window.set(samples.subarray(offsetSamples, offsetSamples + available))
  }
  return window
}
