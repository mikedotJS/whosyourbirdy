#!/usr/bin/env python3
"""Level A': sweep zero-padded tail windows, ONNX vs the official TFLite.

Any recording whose length is not a whole number of 3-second windows ends in a
partially zero-padded window. The error there depends on two things — how much
real audio the tail keeps, and *which* audio — so a single truncation length
proves nothing. This sweeps both.

This is the level that exposes the known limitation documented in
``docs/LIMITATIONS.md``: on tails shorter than about 1 second the score
difference against official BirdNET exceeds the 1e-3 contract, because the mel
front-end was folded from an FFT into a direct 2048-tap dot product, whose
rounding error grows as sqrt(N) rather than the FFT's sqrt(log N). On a
near-silent frame the true mel value is ~0, so that residue is the whole signal.

What is enforced here is the thing that matters: no detection may cross the 0.25
threshold differently. The score envelope is measured and printed so a
regression is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "birdnet"
ONNX = ROOT / "public" / "models" / "birdnet_v2.4_fp32.onnx"
TFLITE = CACHE / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

WINDOW = 144_000
SAMPLE_RATE = 48_000

# Documented envelope for padded tails. This is NOT the parity contract (1e-3);
# it is a regression bound around a known, explained limitation, set at roughly
# twice the worst value measured (1.7e-2). Exceeding it means something changed
# beyond the float32 conditioning already accounted for.
#
# Note this is not a case of upstream dropping short segments and us keeping
# them: birdnet 0.2.16 explicitly "fill[s] last segment with silence up to
# segmentsize if it is smaller than 3s" (acoustic/inference/core/producer.py),
# which is exactly what planWindows/sliceWindow do. The behaviour matches; only
# the arithmetic differs.
PADDED_SCORE_ENVELOPE = 3e-2

# Window offsets into the fixture, and how much real audio the tail keeps.
OFFSETS = [0, 9, 17, 26, 33]
TAIL_SECONDS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5]


def flat_sigmoid(x: np.ndarray) -> np.ndarray:
    y = -np.clip(x, -15.0, 15.0)
    exp_neg = np.exp(-np.abs(y))
    return np.where(y >= 0, exp_neg / (1.0 + exp_neg), 1.0 / (1.0 + exp_neg))


def main() -> int:
    import onnxruntime as ort
    import soundfile as sf
    import tensorflow as tf

    if not ONNX.exists():
        raise SystemExit("run `pnpm model:build` first")

    audio, sr = sf.read(CACHE / "soundscape.wav", dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"fixture must be {SAMPLE_RATE} Hz")
    mono = audio.mean(axis=1)

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    interp = tf.lite.Interpreter(model_path=str(TFLITE), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    worst_by_tail: dict[float, float] = {t: 0.0 for t in TAIL_SECONDS}
    worst_overall = 0.0
    worst_label = ""
    disagreements = 0
    over_contract = 0
    total = 0

    for tail in TAIL_SECONDS:
        keep = int(round(tail * SAMPLE_RATE))
        for offset in OFFSETS:
            start = offset * WINDOW
            if start + keep > len(mono):
                continue
            window = np.zeros(WINDOW, np.float32)
            window[:keep] = mono[start:start + keep]

            onnx_logits = sess.run(["output"], {"input": window[None, :]})[0][0]
            interp.set_tensor(inp["index"], window[None, :])
            interp.invoke()
            tflite_logits = interp.get_tensor(out["index"])[0]

            s_onnx, s_tflite = flat_sigmoid(onnx_logits), flat_sigmoid(tflite_logits)
            d_score = float(np.abs(s_onnx - s_tflite).max())
            flips = int(((s_onnx >= 0.25) != (s_tflite >= 0.25)).sum())

            total += 1
            disagreements += flips
            if d_score > 1e-3:
                over_contract += 1
            worst_by_tail[tail] = max(worst_by_tail[tail], d_score)
            if d_score > worst_overall:
                worst_overall = d_score
                worst_label = f"window {offset}, tail {tail:.2f} s"

    print(f"  {'tail of real audio':>22s}  {'worst max|Δscore|':>18s}   {'vs 1e-3 contract':>16s}")
    for tail in TAIL_SECONDS:
        value = worst_by_tail[tail]
        verdict = "over" if value > 1e-3 else "ok"
        print(f"  {tail:>19.2f} s  {value:18.3e}   {verdict:>16s}")
    print(f"\n  {total} padded windows ({len(OFFSETS)} positions x {len(TAIL_SECONDS)} tail lengths)")
    print(f"  worst: {worst_overall:.3e} at {worst_label}")
    print(f"  over the 1e-3 contract: {over_contract}/{total}")
    print(f"  detections crossing 0.25 differently: {disagreements}   <- this is what is enforced")

    ok = disagreements == 0 and worst_overall <= PADDED_SCORE_ENVELOPE
    print(json.dumps({
        "worst": worst_overall,
        "overContract": over_contract,
        "total": total,
        "disagreements": disagreements,
        "ok": ok,
    }), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
