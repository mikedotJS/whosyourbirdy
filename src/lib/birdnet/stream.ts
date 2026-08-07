import { SAMPLE_RATE, WINDOW_SAMPLES } from './constants'

/**
 * Sliding window over a live stream.
 *
 * Live listening cannot use `planWindows`: there is no total length to plan
 * against, samples arrive in small blocks, and a window has to be emitted the
 * moment it is complete rather than at the end of a file that never comes.
 *
 * What must NOT change is the window itself. The model's contract is 144000
 * samples of 48 kHz mono in [-1, 1], and the streaming path has to hand it
 * exactly the same array the file path would for the same offset — same samples,
 * same order, no normalisation, no fade, no overlap-add. Parity level F checks
 * that byte for byte and then requires the two paths' logits to be *exactly*
 * equal, because any difference here is a windowing bug rather than numerical
 * noise: it is the same model, on the same machine, in the same process.
 *
 * The hop is a second by default, so a bird singing now is scored within about
 * 3.1 s rather than up to 6 s. That costs three inferences per second of audio
 * instead of one — around 10% of a core at the measured ~100 ms per window.
 */

/** Default hop for live listening: 1 s. Detection latency stays ≤ ~3.1 s. */
export const LIVE_HOP_SAMPLES = SAMPLE_RATE

export class SlidingWindower {
  private readonly buffer: Float32Array
  /** Samples currently held, always < windowSamples + hopSamples. */
  private filled = 0
  /** Absolute index, in the stream, of `buffer[0]`. */
  private origin = 0

  constructor(
    private readonly windowSamples: number = WINDOW_SAMPLES,
    private readonly hopSamples: number = LIVE_HOP_SAMPLES,
  ) {
    if (!Number.isInteger(windowSamples) || windowSamples <= 0) {
      throw new RangeError(`windowSamples must be a positive integer, got ${windowSamples}`)
    }
    if (!Number.isInteger(hopSamples) || hopSamples <= 0 || hopSamples > windowSamples) {
      throw new RangeError(
        `hopSamples must be a positive integer no larger than the window, got ${hopSamples}`,
      )
    }
    // One window plus one hop is the most that can be held between emits; blocks
    // larger than a hop are split below so this bound always holds.
    this.buffer = new Float32Array(windowSamples + hopSamples)
  }

  /**
   * Feed a block and take whatever windows it completed.
   *
   * Returns a fresh array per window — the caller transfers them to a worker, so
   * they cannot be views onto this buffer.
   */
  push(block: Float32Array): { samples: Float32Array; offset: number }[] {
    const out: { samples: Float32Array; offset: number }[] = []
    let read = 0

    while (read < block.length) {
      const room = this.buffer.length - this.filled
      const take = Math.min(room, block.length - read)
      this.buffer.set(block.subarray(read, read + take), this.filled)
      this.filled += take
      read += take

      while (this.filled >= this.windowSamples) {
        const samples = new Float32Array(this.windowSamples)
        samples.set(this.buffer.subarray(0, this.windowSamples))
        out.push({ samples, offset: this.origin })

        // Slide by one hop. `copyWithin` runs once per hop — once a second at
        // the default settings — so the cost is irrelevant next to an inference.
        this.buffer.copyWithin(0, this.hopSamples, this.filled)
        this.filled -= this.hopSamples
        this.origin += this.hopSamples
      }
    }

    return out
  }

  /** Samples consumed so far, i.e. where the next window will start. */
  get streamOffset(): number {
    return this.origin
  }

  reset(): void {
    this.filled = 0
    this.origin = 0
  }
}
