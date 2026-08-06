#!/usr/bin/env python3
"""Reference side of the parity harness: official BirdNET inference, in Python.

This is the ground truth the browser pipeline is measured against. It runs the
**official** ``BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`` through the TensorFlow
Lite interpreter and applies BirdNET's own ``flat_sigmoid`` -- i.e. exactly what
the ``birdnet`` PyPI package does, minus its model downloader (which fetches from
Zenodo and cannot be reached from every network).

Nothing about the model is reimplemented here. The audio chain mirrors
BirdNET-Analyzer: read with soundfile, average to mono, resample to 48 kHz with
librosa/resampy only when the file is not already at 48 kHz.

Outputs a .npz with the per-window logits so the comparison step can diff them
against the JavaScript runs without re-running anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "birdnet"
TFLITE = CACHE / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 144_000


def flat_sigmoid(x: np.ndarray, sensitivity: float = 1.0, bias: float = 1.0,
                 clip_val: float = 15.0) -> np.ndarray:
    """Transcribed from birdnet/utils/helper.py::flat_sigmoid_logaddexp_fast."""
    y = -sensitivity * np.clip(x + (bias - 1.0) * 10.0, -clip_val, clip_val)
    positive = y >= 0
    exp_neg = np.exp(-np.abs(y))
    return np.where(positive, exp_neg / (1.0 + exp_neg), 1.0 / (1.0 + exp_neg))


def load_audio(path: Path, *, force_resample: bool = False) -> tuple[np.ndarray, int]:
    """Read a file as 48 kHz mono float32. Returns (samples, original_rate)."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    if sr != SAMPLE_RATE or force_resample:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="kaiser_fast")
    return mono.astype(np.float32), sr


def plan_windows(total: int, overlap: float = 0.0) -> list[int]:
    # Mirrors lib/birdnet/windows.ts exactly. Note the rounding: Python's round()
    # is round-half-to-even while JavaScript's Math.round() is round-half-up, so
    # an overlap landing exactly on a half sample would put the two sides on
    # different hops. floor(x + 0.5) reproduces the JS rule.
    hop = WINDOW_SAMPLES - int(np.floor(overlap * SAMPLE_RATE + 0.5))
    return list(range(0, max(total, 1), hop))


def slice_window(samples: np.ndarray, offset: int) -> np.ndarray:
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    chunk = samples[offset:offset + WINDOW_SAMPLES]
    window[:len(chunk)] = chunk
    return window


def run(path: Path, overlap: float, force_resample: bool,
        truncate_seconds: float | None = None) -> dict:
    import tensorflow as tf

    samples, original_rate = load_audio(path, force_resample=force_resample)
    if truncate_seconds is not None:
        samples = samples[:int(round(truncate_seconds * SAMPLE_RATE))]
    offsets = plan_windows(len(samples), overlap)

    interp = tf.lite.Interpreter(model_path=str(TFLITE), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    logits = np.zeros((len(offsets), 6522), dtype=np.float32)
    for i, offset in enumerate(offsets):
        interp.set_tensor(inp["index"], slice_window(samples, offset)[None, :])
        interp.invoke()
        logits[i] = interp.get_tensor(out["index"])[0]

    return {
        "logits": logits,
        "offsets": np.array(offsets, dtype=np.int64),
        "samples": samples,
        "original_rate": original_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--force-resample", action="store_true",
                        help="resample even when the file is already at 48 kHz")
    parser.add_argument("--truncate-seconds", type=float, default=None,
                        help="cut the audio to this length; use a non-multiple of 3 s to "
                             "exercise the zero-padded final window")
    parser.add_argument("--dump-pcm", action="store_true",
                        help="also store the decoded PCM so the JS side can reuse it verbatim")
    args = parser.parse_args()

    if not TFLITE.exists():
        raise SystemExit(f"missing {TFLITE}; run scripts/fetch_artifacts.py first")

    result = run(args.audio, args.overlap, args.force_resample, args.truncate_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    payload = {"logits": result["logits"], "offsets": result["offsets"]}
    if args.dump_pcm:
        payload["samples"] = result["samples"]
    np.savez(args.out, **payload)

    # A sidecar JSON keeps the JS side from having to parse .npz.
    meta = {
        "audio": str(args.audio),
        "originalRate": int(result["original_rate"]),
        "windows": int(result["logits"].shape[0]),
        "classes": int(result["logits"].shape[1]),
        "overlap": args.overlap,
        "forceResample": args.force_resample,
        "truncateSeconds": args.truncate_seconds,
        "paddedTailSamples": int(
            len(result["offsets"]) * WINDOW_SAMPLES - len(result["samples"])
        ),
        "totalSamples": int(len(result["samples"])),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    if args.dump_pcm:
        # Raw little-endian float32, the exact bytes the JS side will feed to ORT.
        args.out.with_suffix(".pcm").write_bytes(result["samples"].astype("<f4").tobytes())

    scores = flat_sigmoid(result["logits"])
    print(f"{args.audio.name}: {meta['windows']} windows, "
          f"{int((scores >= 0.25).sum())} detections >= 0.25, "
          f"peak score {scores.max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
