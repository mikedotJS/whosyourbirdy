import { DEFAULT_SENSITIVITY, SIGMOID_CLIP } from './constants'

/**
 * BirdNET's "flat sigmoid", transcribed from the official Python package
 * (`birdnet/utils/helper.py`, `flat_sigmoid_logaddexp_fast`).
 *
 * The model emits logits; this turns them into the confidence scores BirdNET
 * reports. Three details matter for matching upstream exactly:
 *
 *  1. The logit is clipped to ±`clipVal` (15) *before* the activation. Without
 *     the clip, saturated logits diverge from the reference implementation.
 *  2. `sensitivity` enters negated, and `bias` is remapped as `(bias - 1) * 10`.
 *     At the defaults (1.0, 1.0) the whole thing collapses to the plain logistic
 *     function — which is why the simple formula is right in the common case and
 *     wrong as soon as anyone touches the sensitivity slider.
 *  3. The two branches avoid `exp()` overflow on large-magnitude inputs. Naive
 *     `1/(1+exp(-x))` returns Infinity/NaN territory well inside the clip range
 *     in float32.
 *
 * @param logits raw model output
 * @param sensitivity BirdNET's sigmoid sensitivity, valid range [0.5, 1.5]
 * @param bias remapped as `(bias - 1) * 10` and added to the logit
 * @param clipVal symmetric clamp applied before the activation
 * @returns a new array of scores in [0, 1]
 */
export function flatSigmoid(
  logits: Float32Array,
  sensitivity: number = DEFAULT_SENSITIVITY,
  bias = 1.0,
  clipVal: number = SIGMOID_CLIP,
): Float32Array {
  const out = new Float32Array(logits.length)
  const shift = (bias - 1.0) * 10.0

  for (let i = 0; i < logits.length; i++) {
    let v = logits[i] + shift
    if (v < -clipVal) v = -clipVal
    else if (v > clipVal) v = clipVal

    const y = -sensitivity * v
    // Branch on the sign so exp() only ever sees a non-positive argument.
    const expNegAbs = Math.exp(-Math.abs(y))
    out[i] = y >= 0 ? expNegAbs / (1.0 + expNegAbs) : 1.0 / (1.0 + expNegAbs)
  }
  return out
}

/** Single-value form, for thresholds and tests. */
export function flatSigmoidOne(
  logit: number,
  sensitivity: number = DEFAULT_SENSITIVITY,
  bias = 1.0,
  clipVal: number = SIGMOID_CLIP,
): number {
  const v = Math.min(clipVal, Math.max(-clipVal, logit + (bias - 1.0) * 10.0))
  const y = -sensitivity * v
  const expNegAbs = Math.exp(-Math.abs(y))
  return y >= 0 ? expNegAbs / (1.0 + expNegAbs) : 1.0 / (1.0 + expNegAbs)
}
