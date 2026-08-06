#!/usr/bin/env python3
"""Level D: compare ONNX against the official TFLite on degenerate inputs.

The soundscape fixture is well-behaved audio. Real files are not always: a
zero-padded tail is literally silence, a clipped recording saturates, and a
pure tone has almost no spectral content. Those are exactly the inputs where the
mel front-end is most fragile, because the model normalises each window by
``x / (max(x) + 1e-6)`` and then raises the result to a fractional power whose
derivative is unbounded at zero.

The all-zero case is not hypothetical. Any file whose length is not a whole
number of 3-second windows produces a partially zero window, and a file ending
in silence produces a fully zero one.

This runs the two implementations directly, with no audio chain in between, so
anything it finds is the model conversion and nothing else.
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
SCORE_TOL = 1e-3


def flat_sigmoid(x: np.ndarray) -> np.ndarray:
    y = -np.clip(x, -15.0, 15.0)
    exp_neg = np.exp(-np.abs(y))
    return np.where(y >= 0, exp_neg / (1.0 + exp_neg), 1.0 / (1.0 + exp_neg))


def cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW, dtype=np.float64) / SAMPLE_RATE
    real, _ = _load_real_window()
    return {
        # The zero-padded tail, in its purest form.
        "all zeros": np.zeros(WINDOW, np.float32),
        # Constant signal: min == max, so the normalisation divides by ~1e-6.
        "DC offset 0.5": np.full(WINDOW, 0.5, np.float32),
        "full-scale square 1 kHz": np.sign(np.sin(2 * np.pi * 1000 * t)).astype(np.float32),
        "pure 4 kHz tone": np.sin(2 * np.pi * 4000 * t).astype(np.float32),
        "white noise": (rng.standard_normal(WINDOW) * 0.1).astype(np.float32),
        "clipped to +-1": np.clip(rng.standard_normal(WINDOW) * 5, -1, 1).astype(np.float32),
        # Half real audio, half silence: the actual shape of a padded tail.
        "half audio / half silence": np.concatenate(
            [real[: WINDOW // 2], np.zeros(WINDOW // 2, np.float32)]
        ).astype(np.float32),
        "real audio (control)": real,
    }


def _load_real_window() -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(CACHE / "soundscape.wav", dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    # A window from the middle of the file, away from the ones the converter used.
    offset = 21 * WINDOW
    return mono[offset:offset + WINDOW].copy(), sr


def main() -> int:
    import onnxruntime as ort
    import tensorflow as tf

    if not ONNX.exists():
        raise SystemExit("run `pnpm model:build` first")

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    interp = tf.lite.Interpreter(model_path=str(TFLITE), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    rows = []
    worst_score = 0.0
    total_disagreements = 0

    for name, window in cases().items():
        onnx_logits = sess.run(["output"], {"input": window[None, :]})[0][0]
        interp.set_tensor(inp["index"], window[None, :])
        interp.invoke()
        tflite_logits = interp.get_tensor(out["index"])[0]

        d_logit = float(np.abs(onnx_logits - tflite_logits).max())
        s_onnx, s_tflite = flat_sigmoid(onnx_logits), flat_sigmoid(tflite_logits)
        d_score = float(np.abs(s_onnx - s_tflite).max())
        disagreements = int(((s_onnx >= 0.25) != (s_tflite >= 0.25)).sum())

        worst_score = max(worst_score, d_score)
        total_disagreements += disagreements
        rows.append((name, d_logit, d_score, disagreements))

    print(f"  {'input':28s} {'max|Δlogit|':>12s} {'max|Δscore|':>12s} {'disagree@0.25':>14s}")
    for name, d_logit, d_score, disagreements in rows:
        flag = "" if d_score <= SCORE_TOL and disagreements == 0 else "   <-- FAIL"
        print(f"  {name:28s} {d_logit:12.3e} {d_score:12.3e} {disagreements:14d}{flag}")

    ok = worst_score <= SCORE_TOL and total_disagreements == 0
    summary = {
        "maxScore": worst_score,
        "disagreements": total_disagreements,
        "ok": ok,
    }
    print(json.dumps(summary), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
