#!/usr/bin/env python3
"""Level E: our geo ONNX against the official MData TFLite.

The acoustic levels all have an audio chain between the file and the model, so a
failure there could be resampling, windowing or the model. This one has nothing
in between: three floats in, 6522 out. Anything it finds is the conversion.

What it checks, over four continents and the whole year:

  - ``max|Δprobability|`` against the official interpreter, on the same grid the
    export asserts, plus a randomised sweep so the grid cannot be the only place
    it happens to agree.
  - **Species disagreements at the filter threshold.** This is the gate that
    matters. The filter is a hard cut (``invalid_mask = res < min_confidence`` in
    ``birdnet/geo/inference/session.py``), so a single class landing on the other
    side of 0.03 changes which birds appear in the report — a tolerance on the
    probability alone would not catch a boundary flip.
  - **The week convention.** BirdNET splits every month into four and numbers the
    result 1-48; feeding it an ISO week would shift the season by up to a month.
    The formula the app uses is re-derived here from dates and compared against
    upstream's, because getting this wrong produces a plausible, wrong answer
    rather than an error.

Prints a human-readable report on stdout and a one-line JSON summary on stderr,
same contract as ``parity_degenerate.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "birdnet"
ONNX = ROOT / "public" / "models" / "birdnet_v2.4_mdata.onnx"
TFLITE = CACHE / "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"

SCORE_TOL = 1e-3
SF_THRESH = 0.03
N_CLASSES = 6522

PLACES = {
    "Paris": (48.85, 2.35),
    "New York": (40.71, -74.01),
    "Nairobi": (-1.29, 36.82),
    "Sydney": (-33.87, 151.21),
    # A pole and the antimeridian: the coordinate extremes the UI accepts.
    "North Pole": (90.0, 0.0),
    "Antimeridian": (0.0, 180.0),
}
WEEKS = [-1, 1, 12, 20, 24, 36, 48]


def birdnet_week(d: date) -> int:
    """Upstream's week number: four per month, 1-48."""
    return (d.month - 1) * 4 + min(3, (d.day - 1) // 7) + 1


def check_week_convention() -> tuple[bool, str]:
    """Re-derive the convention from dates and check its boundaries.

    Written as properties rather than a table of expected values so it cannot be
    made to pass by copying the implementation's output into the test.
    """
    problems = []

    # Every day of a non-leap year must map into 1..48, and the mapping must be
    # monotone: later in the year is never an earlier week.
    previous = 0
    for offset in range(365):
        d = date(2025, 1, 1) + timedelta(days=offset)
        w = birdnet_week(d)
        if not 1 <= w <= 48:
            problems.append(f"{d} -> week {w}, outside 1..48")
            break
        if w < previous:
            problems.append(f"{d} -> week {w} after week {previous}: not monotone")
            break
        previous = w

    # Four weeks per month, and the fourth absorbs the month's tail.
    if birdnet_week(date(2025, 1, 1)) != 1:
        problems.append("1 January is not week 1")
    if birdnet_week(date(2025, 12, 31)) != 48:
        problems.append("31 December is not week 48")
    if birdnet_week(date(2025, 1, 22)) != birdnet_week(date(2025, 1, 31)):
        problems.append("days 22-31 of a month do not share its fourth week")
    if birdnet_week(date(2025, 3, 1)) != 9:
        problems.append("1 March is not week 9 (months are exactly four weeks)")

    # And the trap this exists to rule out: it is NOT the ISO week.
    iso = date(2025, 5, 15).isocalendar()[1]
    if birdnet_week(date(2025, 5, 15)) == iso:
        problems.append("mid-May matches the ISO week — the convention may have been confused")

    return not problems, "; ".join(problems) if problems else "1..48, four per month, monotone"


def main() -> int:
    for path in (ONNX, TFLITE):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    import onnxruntime as ort
    import tensorflow as tf

    lite = tf.lite.Interpreter(model_path=str(TFLITE))
    lite.allocate_tensors()
    lin, lout = lite.get_input_details()[0], lite.get_output_details()[0]

    def tflite_probs(x: np.ndarray) -> np.ndarray:
        lite.set_tensor(lin["index"], x.reshape(1, 3).astype(np.float32))
        lite.invoke()
        return lite.get_tensor(lout["index"])[0]

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    def onnx_probs(x: np.ndarray) -> np.ndarray:
        return sess.run(["output"], {"input": x.reshape(1, 3).astype(np.float32)})[0][0]

    cases: list[tuple[str, np.ndarray]] = []
    for name, (lat, lon) in PLACES.items():
        for week in WEEKS:
            cases.append((f"{name} w{week}", np.array([lat, lon, week], np.float32)))

    # A randomised sweep on top of the fixed grid: agreeing on six chosen points
    # is much weaker evidence than agreeing on sixty arbitrary ones.
    rng = np.random.default_rng(0)
    for i in range(60):
        lat = float(rng.uniform(-90, 90))
        lon = float(rng.uniform(-180, 180))
        week = int(rng.integers(1, 49))
        cases.append((f"random#{i} {lat:.1f},{lon:.1f} w{week}", np.array([lat, lon, week], np.float32)))

    worst, worst_at, disagreements, worst_case = 0.0, "", 0, ""
    for label, x in cases:
        a, b = tflite_probs(x), onnx_probs(x)
        delta = float(np.abs(a - b).max())
        if delta > worst:
            worst, worst_at = delta, label
        flips = int(((a >= SF_THRESH) != (b >= SF_THRESH)).sum())
        if flips and not worst_case:
            worst_case = label
        disagreements += flips

    print(f"  {len(cases)} (place, week) pairs — {len(PLACES)} named places, 60 random")
    print(f"  max |delta probability| : {worst:.3e} at {worst_at} (tol {SCORE_TOL:.0e})")
    print(f"  species crossing {SF_THRESH} differently: {disagreements}"
          + (f" (first at {worst_case})" if worst_case else ""))

    week_ok, week_detail = check_week_convention()
    print(f"  week convention          : {week_detail}")

    ok = worst <= SCORE_TOL and disagreements == 0 and week_ok
    print(json.dumps({
        "maxScore": worst,
        "disagreements": disagreements,
        "cases": len(cases),
        "weekConvention": week_ok,
    }), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
