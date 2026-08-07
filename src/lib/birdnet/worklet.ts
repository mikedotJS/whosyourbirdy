/**
 * The audio worklet, as source text.
 *
 * `AudioWorklet.addModule()` takes a URL and fetches it — it is not an import,
 * so the bundler has no say in resolving it. Rather than put a file in `public/`
 * and hope, the source lives here as a string and is handed over as a Blob URL.
 * After the `public/ort` episode, where Vite refused to serve a public file that
 * was reached by an import and broke the dev server while the production build
 * looked healthy, depending on no resolution at all is worth the awkwardness.
 *
 * The worklet does two things, both because the samples are already there:
 *
 *  - **Batches blocks.** `process()` is called with 128 frames, ~375 times a
 *    second. Posting each one is 375 messages/s for 512 bytes each; batching to
 *    100 ms makes it ten messages of 19 KB. The sliding window itself lives on
 *    the main thread in `stream.ts`, where parity level F can exercise it.
 *  - **Computes RMS.** It drives the listening animation, and computing it here
 *    costs a multiply per sample in a loop that already exists, instead of
 *    shipping PCM to the main thread to measure it.
 *
 * What it does NOT do is touch the samples. No gain, no filter, no
 * normalisation: the model normalises each window itself, and anything applied
 * here would be applied twice.
 */

/** Batch size posted to the main thread. 100 ms at 48 kHz. */
export const WORKLET_BATCH_SAMPLES = 4800

export const LIVE_WORKLET_NAME = 'birdnet-capture'

const SOURCE = `
const BATCH = ${WORKLET_BATCH_SAMPLES}

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.batch = new Float32Array(BATCH)
    this.filled = 0
    this.sumSquares = 0
    this.counted = 0
  }

  process(inputs) {
    const channels = inputs[0]
    // No input yet, or the track ended. Returning true keeps the node alive so
    // capture resumes if the device comes back.
    if (!channels || !channels.length || !channels[0]) return true

    const count = channels.length
    const first = channels[0]

    for (let i = 0; i < first.length; i++) {
      // Downmix by averaging, the same as the file path.
      //
      // \`channelCount: 1\` in the getUserMedia constraints is a *hint*; a source
      // can still arrive with two channels. Reading channels[0] alone would take
      // the left channel and silently drop anything panned right — the file path
      // averages, and the README says it averages, so this must too.
      let sample = first[i]
      if (count > 1) {
        for (let c = 1; c < count; c++) sample += channels[c][i]
        sample /= count
      }
      this.batch[this.filled++] = sample
      this.sumSquares += sample * sample
      this.counted++

      if (this.filled === BATCH) {
        // Transfer, not copy: this buffer is handed over and replaced.
        const out = this.batch
        this.batch = new Float32Array(BATCH)
        this.filled = 0
        const rms = Math.sqrt(this.sumSquares / this.counted)
        this.sumSquares = 0
        this.counted = 0
        this.port.postMessage({ samples: out, rms }, [out.buffer])
      }
    }
    return true
  }
}

registerProcessor(${JSON.stringify(LIVE_WORKLET_NAME)}, CaptureProcessor)
`

/**
 * A Blob URL for the worklet module. The caller revokes it once `addModule`
 * resolves — the module is compiled by then, and leaking one URL per session
 * would be sloppy in an app meant to run for a long listening session.
 */
export function createWorkletUrl(): string {
  return URL.createObjectURL(new Blob([SOURCE], { type: 'text/javascript' }))
}
