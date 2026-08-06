#!/usr/bin/env python3
"""Build ``public/models/birdnet_v2.4_fp32.onnx`` from the official BirdNET artifacts.

Pipeline
--------
1. Rebuild the Keras topology from the official TFJS export's ``model_config``,
   substituting our ``MelSpecLayerSimple`` (see ``birdnet_mel.py``) for the layer
   whose code upstream does not serialise.
2. Load every weight by name from the TFJS weight shards.
3. Assert the folded Conv1D mel front-end matches the literal STFT one, in
   TensorFlow, on real audio.
4. Assert the whole Keras model matches the official FP32 TFLite interpreter.
5. Export to ONNX, normalise the tensor names to ``input`` / ``output``, and
   assert the graph contains no op the WASM execution provider cannot run.
6. Re-check the exported ONNX against TFLite through onnxruntime.

Every step that could silently produce a plausible-but-wrong model is an
assertion, not a log line: if this script exits 0, the ONNX behaves like BirdNET.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / ".cache" / "birdnet"
OUT = ROOT / "public" / "models"
WORK = ROOT / ".cache" / "onnx"

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 144_000
N_CLASSES = 6522
OPSET = 17

# ONNX ops that onnxruntime-web's WASM execution provider does not implement (or
# implements unreliably). Seeing one here means the export regressed to a graph
# that will fail in the browser -- fail loudly at build time instead.
FORBIDDEN_OPS = {"DFT", "STFT", "MelWeightMatrix", "HannWindow", "BlackmanWindow", "Rfft"}

# Tolerances. The Keras/TFLite comparison is the tight one because both run the
# same float32 arithmetic on the same host; ONNX adds another runtime's rounding.
TOL_FOLD = 2e-3          # folded conv vs literal STFT, on logits
TOL_KERAS_TFLITE = 5e-3  # rebuilt Keras vs official TFLite, on logits
TOL_ONNX_TFLITE = 5e-3   # exported ONNX vs official TFLite, on logits
TOL_SCORE = 1e-3         # anything, after the sigmoid -- the number that matters


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flat_sigmoid(x: np.ndarray, sensitivity: float = 1.0, bias: float = 1.0,
                 clip_val: float = 15.0) -> np.ndarray:
    """BirdNET's activation, transcribed from ``birdnet/utils/helper.py``.

    At the default ``sensitivity=1.0, bias=1.0`` this is the plain logistic
    function, clipped at +/-15. Kept in its general form so the parity harness can
    exercise the same knobs BirdNET-Analyzer exposes.
    """
    y = -sensitivity * np.clip(x + (bias - 1.0) * 10.0, -clip_val, clip_val)
    positive = y >= 0
    exp_neg = np.exp(-np.abs(y))
    return np.where(positive, exp_neg / (1.0 + exp_neg), 1.0 / (1.0 + exp_neg))


# --------------------------------------------------------------------------- #
# 1-2. rebuild the Keras model
# --------------------------------------------------------------------------- #

def load_tfjs_weights(spec: dict) -> dict[str, np.ndarray]:
    """Read the TFJS shards into ``{weight_name: array}``.

    The shards are one contiguous little-endian float32 buffer split on arbitrary
    4 MB boundaries, so they are concatenated before slicing.
    """
    group = spec["weightsManifest"][0]
    buf = b"".join((CACHE / "tfjs" / p).read_bytes() for p in group["paths"])
    weights: dict[str, np.ndarray] = {}
    offset = 0
    for entry in group["weights"]:
        if entry["dtype"] != "float32":
            raise ValueError(f"unexpected dtype {entry['dtype']} for {entry['name']}")
        count = int(np.prod(entry["shape"])) if entry["shape"] else 1
        nbytes = count * 4
        arr = np.frombuffer(buf, dtype="<f4", count=count, offset=offset)
        weights[entry["name"]] = arr.reshape(entry["shape"]).copy()
        offset += nbytes
    if offset != len(buf):
        raise ValueError(f"weight buffer has {len(buf) - offset} trailing bytes")
    return weights


def build_keras(mode: str):
    import tensorflow as tf
    from birdnet_mel import MelSpecLayerSimple

    MelSpecLayerSimple.export_mode = mode
    spec = json.loads((CACHE / "tfjs_model.json").read_text())
    model = tf.keras.models.model_from_json(
        json.dumps(spec["modelTopology"]["model_config"]),
        custom_objects={"MelSpecLayerSimple": MelSpecLayerSimple},
    )

    weights = load_tfjs_weights(spec)
    assigned = 0
    for layer in model.layers:
        if not layer.weights:
            continue
        values = []
        for w in layer.weights:
            # Keras weight paths look like "CONV_0/kernel:0"; TFJS drops the ":0".
            key = w.name.split(":")[0]
            if key not in weights:
                raise KeyError(f"no TFJS weight named {key!r} for layer {layer.name}")
            value = weights[key]
            if tuple(value.shape) != tuple(w.shape):
                raise ValueError(
                    f"{key}: TFJS shape {value.shape} != Keras shape {tuple(w.shape)}"
                )
            values.append(value)
            assigned += 1
        layer.set_weights(values)

    if assigned != len(weights):
        missing = set(weights) - {
            w.name.split(":")[0] for l in model.layers for w in l.weights
        }
        raise ValueError(f"{len(missing)} TFJS weights were never assigned: {sorted(missing)[:5]}")

    log(f"  keras[{mode}]: {len(model.layers)} layers, {model.count_params():,} params, "
        f"{assigned} weights restored")
    return model


def strip_final_sigmoid(model):
    """Cut the trailing ``CLASS_ACTIVATION`` layer so the model emits logits.

    The Keras/TFJS export ends with ``Dense(6522, linear) -> Activation(sigmoid)``
    and therefore returns probabilities, while the official TFLite graph stops at
    ``CLASS_DENSE_LAYER/BiasAdd`` and returns logits. We ship logits, for two
    reasons: it is the tensor the TFLite reference produces, so parity is checked
    against the same quantity; and BirdNET's sigmoid sensitivity is a *parameter*,
    which is only adjustable if the activation is applied outside the graph.
    """
    import tensorflow as tf

    last = model.layers[-1]
    activation = getattr(last, "activation", None)
    if last.name != "CLASS_ACTIVATION" or activation is not tf.keras.activations.sigmoid:
        raise ValueError(
            f"expected the model to end with a sigmoid CLASS_ACTIVATION, "
            f"found {last.name} ({last.__class__.__name__}, activation={activation})"
        )
    return tf.keras.Model(model.input, model.get_layer("CLASS_DENSE_LAYER").output,
                          name="birdnet_v2_4_logits")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def probe_windows(n: int = 3) -> np.ndarray:
    """Real audio, not noise: the mel front-end is scale- and structure-sensitive.

    Windows are taken from the official example soundscape, spread across the file
    so we exercise loud, quiet and mixed content.
    """
    import soundfile as sf

    audio, sr = sf.read(CACHE / "soundscape.wav", dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"fixture must be {SAMPLE_RATE} Hz, got {sr}")
    mono = audio.mean(axis=1)
    total = len(mono) // WINDOW_SAMPLES
    picks = np.linspace(0, total - 1, n, dtype=int)
    return np.stack([mono[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES] for i in picks])


def tflite_logits(batch: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(
        model_path=str(CACHE / "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"),
        num_threads=1,
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    results = []
    for window in batch:
        interp.set_tensor(inp["index"], window[None, :].astype(np.float32))
        interp.invoke()
        results.append(interp.get_tensor(out["index"])[0].copy())
    return np.stack(results)


def report(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """Compare logits and the scores derived from them; scores are what ship."""
    dlogit = float(np.abs(a - b).max())
    dscore = float(np.abs(flat_sigmoid(a) - flat_sigmoid(b)).max())
    ok = dlogit <= tol and dscore <= TOL_SCORE
    log(f"  {'PASS' if ok else 'FAIL'} {name}: "
        f"max|dlogit|={dlogit:.3e} (tol {tol:.0e})  max|dscore|={dscore:.3e} (tol {TOL_SCORE:.0e})")
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checks", action="store_true",
                        help="export without the equivalence assertions (debugging only)")
    args = parser.parse_args()

    import tensorflow as tf
    import onnx
    import tf2onnx

    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    log("[1/6] rebuilding the Keras model from the official topology + weights")
    model = strip_final_sigmoid(build_keras("conv"))

    windows = probe_windows()
    log(f"[2/6] probing with {len(windows)} real 3 s windows from soundscape.wav")

    ok = True
    if not args.skip_checks:
        log("[3/6] checking the folded mel front-end against the literal STFT")
        literal = strip_final_sigmoid(build_keras("stft"))
        ok &= report("folded conv vs tf.signal.stft",
                     model.predict(windows, verbose=0),
                     literal.predict(windows, verbose=0),
                     TOL_FOLD)
        del literal

        log("[4/6] checking the rebuilt model against the official FP32 TFLite")
        reference = tflite_logits(windows)
        ok &= report("keras vs tflite", model.predict(windows, verbose=0), reference, TOL_KERAS_TFLITE)
    else:
        reference = None
        log("[3/6] [4/6] skipped (--skip-checks)")

    log("[5/6] exporting to ONNX")
    signature = (tf.TensorSpec((1, WINDOW_SAMPLES), tf.float32, name="input"),)
    proto, _ = tf2onnx.convert.from_keras(model, input_signature=signature, opset=OPSET)

    # Normalise the IO names so the TypeScript side has a stable contract.
    # tf2onnx derives them from the Keras layer names, which we do not want to
    # leak into the app; renaming means rewriting every reference, not just the
    # graph's own input/output entries.
    if len(proto.graph.input) != 1 or len(proto.graph.output) != 1:
        raise SystemExit(
            f"expected exactly one input and one output, got "
            f"{len(proto.graph.input)}/{len(proto.graph.output)}"
        )
    rename = {proto.graph.input[0].name: "input", proto.graph.output[0].name: "output"}
    for node in proto.graph.node:
        node.input[:] = [rename.get(name, name) for name in node.input]
        node.output[:] = [rename.get(name, name) for name in node.output]
    for value in list(proto.graph.value_info) + list(proto.graph.initializer):
        value.name = rename.get(value.name, value.name)
    proto.graph.input[0].name = "input"
    proto.graph.output[0].name = "output"

    onnx.checker.check_model(proto)

    ops = {n.op_type for n in proto.graph.node}
    forbidden = ops & FORBIDDEN_OPS
    if forbidden:
        log(f"  FAIL exported graph contains ops the WASM backend cannot run: {sorted(forbidden)}")
        ok = False
    else:
        log(f"  PASS no unsupported ops; graph uses {len(ops)} distinct op types")

    gi, go = proto.graph.input[0], proto.graph.output[0]
    ishape = [d.dim_value for d in gi.type.tensor_type.shape.dim]
    oshape = [d.dim_value for d in go.type.tensor_type.shape.dim]
    if ishape != [1, WINDOW_SAMPLES] or oshape != [1, N_CLASSES]:
        log(f"  FAIL signature is {gi.name}{ishape} -> {go.name}{oshape}, "
            f"expected input[1, {WINDOW_SAMPLES}] -> output[1, {N_CLASSES}]")
        ok = False
    else:
        log(f"  PASS signature {gi.name}{ishape} -> {go.name}{oshape} float32 (logits)")

    model_path = OUT / "birdnet_v2.4_fp32.onnx"
    onnx.save(proto, str(model_path))
    log(f"  wrote {model_path.relative_to(ROOT)} ({model_path.stat().st_size / 1e6:.1f} MB)")

    if not args.skip_checks:
        log("[6/6] re-checking the exported ONNX through onnxruntime")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        got = np.concatenate([sess.run(["output"], {"input": w[None, :]})[0] for w in windows])
        ok &= report("onnx vs tflite", got, reference, TOL_ONNX_TFLITE)
    else:
        log("[6/6] skipped (--skip-checks)")

    # Labels and licence travel with the model.
    for src, dst in (("labels_fr.txt", "labels_fr.txt"), ("labels_en_us.txt", "labels_en.txt")):
        shutil.copyfile(CACHE / src, OUT / dst)
    lines = (OUT / "labels_en.txt").read_text().strip().split("\n")
    if len(lines) != N_CLASSES:
        log(f"  FAIL labels_en.txt has {len(lines)} entries, expected {N_CLASSES}")
        ok = False
    (OUT / "LICENSE").write_text(MODEL_LICENSE)

    manifest = {
        "model": "BirdNET-Analyzer V2.4",
        "file": model_path.name,
        "sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "sampleRate": SAMPLE_RATE,
        "windowSamples": WINDOW_SAMPLES,
        "classes": N_CLASSES,
        "output": "logits (apply the flat sigmoid)",
        "opset": OPSET,
        "provenance": {
            "repo": "https://github.com/birdnet-team/BirdNET-Analyzer",
            "tag": "v1.5.1",
            "topology": "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_TFJS/static/model/model.json",
            "weights": "the 13 TFJS weight shards alongside it",
            "verifiedAgainst": "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
        },
        "license": "CC BY-NC-SA 4.0",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"  wrote labels_fr.txt, labels_en.txt, LICENSE, manifest.json")

    log("\nOK" if ok else "\nFAILED: the exported model does not match BirdNET; do not ship it.")
    return 0 if ok else 1


MODEL_LICENSE = """\
BirdNET model weights -- Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)

The BirdNET v2.4 model in this directory is derived from BirdNET-Analyzer,
published by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell
Lab of Ornithology and by Chemnitz University of Technology.

  Source:  https://github.com/birdnet-team/BirdNET-Analyzer (tag v1.5.1)
  Licence: https://creativecommons.org/licenses/by-nc-sa/4.0/

You may share and adapt these weights, provided that you:

  - give appropriate credit (BY),
  - do not use them for commercial purposes (NC),
  - distribute any derivative under the same licence (SA).

This ONNX file is an adaptation: the model's own STFT + mel front-end has been
folded into an equivalent strided convolution so it can run under the ONNX Runtime
Web WASM backend. The weights, the mel filterbanks and the magnitude scaling are
unchanged, and the conversion is verified against the official FP32 TFLite model.

Please cite:

  Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning
  solution for avian diversity monitoring. Ecological Informatics, 61, 101236.
"""


if __name__ == "__main__":
    raise SystemExit(main())
