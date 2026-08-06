import { SAMPLE_RATE } from './constants'

/**
 * Decode an audio file to 48 kHz mono float PCM, the way BirdNET expects it.
 *
 * What this deliberately does NOT do
 * ----------------------------------
 * No peak normalisation, no RMS normalisation, no high-pass or low-pass filter,
 * no int16 round-trip. Two reasons:
 *
 *  - `decodeAudioData` already yields float samples in [-1, 1], which is exactly
 *    the model's input domain.
 *  - The model normalises each 3-second window itself: its mel layer starts with
 *    a per-window min-max rescale to [-1, 1] (see `scripts/birdnet_mel.py`).
 *    Normalising beforehand changes the window statistics the model was trained
 *    on and measurably degrades scores.
 *
 * Resampling is delegated to `OfflineAudioContext`, which is the browser's own
 * (high quality, implementation-defined) resampler. We do not hand-roll one:
 * matching BirdNET's `resampy` output sample-for-sample is not achievable in the
 * browser, and `scripts/parity.mjs` level C measures the residual difference
 * rather than papering over it.
 */

export interface DecodedAudio {
  /** Mono PCM at exactly `SAMPLE_RATE`. */
  samples: Float32Array
  /** Duration in seconds, derived from `samples.length`. */
  duration: number
  /**
   * The rate the browser handed back from `decodeAudioData`.
   *
   * This is usually already `SAMPLE_RATE`, because Chromium and Firefox resample
   * to the decoding context's rate — which is why we decode in a 48 kHz context
   * on purpose. It is NOT necessarily the file's own rate, so do not present it
   * to the user as "this file is 48 kHz".
   */
  decodedSampleRate: number
  /** True when `resampleTo48k` had to do work, i.e. the decoder did not resample. */
  resampled: boolean
  channels: number
}

function assertAudioSupport(): void {
  if (typeof globalThis.OfflineAudioContext !== 'function') {
    throw new Error('Web Audio API is unavailable; audio decoding requires a browser')
  }
}

/**
 * Decode compressed or uncompressed audio into raw PCM at its native rate.
 *
 * Format support is the browser's, not ours: wav and mp3 everywhere, flac and
 * m4a/aac in most engines. A failure here is a decode failure, not a pipeline
 * bug, so it is surfaced verbatim.
 */
async function decodeToBuffer(data: ArrayBuffer): Promise<AudioBuffer> {
  assertAudioSupport()
  // The context rate is deliberately 48 kHz, not incidental: Chromium and Firefox
  // resample during `decodeAudioData` to the decoding context's rate, so asking
  // for 48 kHz here gets the resampling done by the decoder in one pass. The
  // official BirdNET browser demo does the same thing, and for the same reason —
  // the default 44.1 kHz context would silently downsample every file.
  //
  // Safari has historically returned the file's own rate instead, which is why
  // `resampleTo48k` below is a real fallback and not dead code.
  const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE)
  try {
    return await ctx.decodeAudioData(data)
  } catch (cause) {
    throw new Error(
      'Could not decode this audio file. Supported formats depend on the browser ' +
        '(wav and mp3 always; flac, m4a and ogg usually).',
      { cause },
    )
  }
}

/** Average all channels into one. BirdNET analyses mono. */
function downmixToMono(buffer: AudioBuffer): Float32Array {
  const { numberOfChannels, length } = buffer
  if (numberOfChannels === 1) {
    // getChannelData returns a live view; copy so later resampling cannot alias.
    return new Float32Array(buffer.getChannelData(0))
  }

  const mono = new Float32Array(length)
  for (let c = 0; c < numberOfChannels; c++) {
    const channel = buffer.getChannelData(c)
    for (let i = 0; i < length; i++) mono[i] += channel[i]
  }
  const scale = 1 / numberOfChannels
  for (let i = 0; i < length; i++) mono[i] *= scale
  return mono
}

/**
 * Resample mono PCM to 48 kHz using the browser's resampler.
 *
 * Returns the input untouched when it is already at the target rate, which keeps
 * the common case (48 kHz field recordings) bit-exact against the reference.
 */
async function resampleTo48k(mono: Float32Array, sourceRate: number): Promise<Float32Array> {
  if (sourceRate === SAMPLE_RATE) return mono

  const targetLength = Math.max(1, Math.round((mono.length * SAMPLE_RATE) / sourceRate))
  const ctx = new OfflineAudioContext(1, targetLength, SAMPLE_RATE)

  // The source buffer must carry its *original* rate so the graph resamples it.
  const source = ctx.createBuffer(1, mono.length, sourceRate)
  // copyToChannel is typed against a non-shared ArrayBuffer; the copy in
  // downmixToMono already guarantees that, so restate it for the compiler.
  source.copyToChannel(mono as Float32Array<ArrayBuffer>, 0)

  const node = ctx.createBufferSource()
  node.buffer = source
  node.connect(ctx.destination)
  node.start()

  const rendered = await ctx.startRendering()
  return new Float32Array(rendered.getChannelData(0))
}

/** Decode → downmix → resample. The only audio entry point the pipeline uses. */
export async function decodeAudio(input: ArrayBuffer | Blob): Promise<DecodedAudio> {
  const data = input instanceof Blob ? await input.arrayBuffer() : input
  const buffer = await decodeToBuffer(data)

  const mono = downmixToMono(buffer)
  const samples = await resampleTo48k(mono, buffer.sampleRate)

  return {
    samples,
    duration: samples.length / SAMPLE_RATE,
    decodedSampleRate: buffer.sampleRate,
    resampled: buffer.sampleRate !== SAMPLE_RATE,
    channels: buffer.numberOfChannels,
  }
}
