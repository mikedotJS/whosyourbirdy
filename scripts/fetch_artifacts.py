#!/usr/bin/env python3
"""Fetch the official BirdNET v2.4 artifacts from the BirdNET-Analyzer git history.

Why git and not Hugging Face / Zenodo
-------------------------------------
The upstream project stopped shipping weights inside the repository at v2.4.0 and
now downloads them from Zenodo at first run. Both Zenodo and Hugging Face are
unreachable from some networks (including CI sandboxes with an egress policy),
while ``github.com`` over plain git is not. Tag ``v1.5.1`` is the last tag that
still carries every artifact we need as a git blob, and those blobs are the same
V2.4 weights the current release downloads -- the model did not change, only its
distribution channel did.

A ``--filter=blob:none`` partial clone means we pay for the four blobs we ask for
(~68 MB) instead of the repository's full history.

Everything lands in ``.cache/birdnet/`` and is checksum-verified, so the download
happens once and any later step can trust the bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/birdnet-team/BirdNET-Analyzer.git"
TAG = "v1.5.1"

CKPT = "birdnet_analyzer/checkpoints/V2.4"
TFJS = f"{CKPT}/BirdNET_GLOBAL_6K_V2.4_Model_TFJS/static/model"

# path in the repository -> (local name, sha256)
# The digests are what this script observed at tag v1.5.1; a mismatch means the
# tag moved or the transfer is corrupt, and we refuse to continue either way.
ARTIFACTS: dict[str, tuple[str, str]] = {
    f"{CKPT}/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite": (
        "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
        "55f3e4055b1a13bfa9a2452731d0d34f6a02d6b775a334362665892794165e4c",
    ),
    f"{CKPT}/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite": (
        "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite",
        "1226f23fc20362617deb09178f366111b70cf085c2c82893f2814e5acedce6c2",
    ),
    f"{CKPT}/BirdNET_GLOBAL_6K_V2.4_Labels.txt": (
        "labels_en_us.txt",
        "b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16",
    ),
    "birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_fr.txt": (
        "labels_fr.txt",
        "e392281599d4a2b331e4471bce783313eba4c69150c7b4ff4f44df305880e6b3",
    ),
    "birdnet_analyzer/example/soundscape.wav": (
        "soundscape.wav",
        "df312b45bc82ce4c638c3e9e09d748702ea14a91ec29e4e8e0676d3e3e015fd7",
    ),
    # The TFJS export of the same V2.4 checkpoint. We take the Keras topology and
    # the named weights from here: the SavedModel in this tag ships without
    # `keras_metadata.pb`, so Keras cannot revive it as a functional model, and
    # its checkpoint restores by object-graph position, which a freshly built
    # model does not reproduce. The TFJS manifest names every weight
    # (`LAYER/kernel`, `LAYER/gamma`, ...), so the mapping is unambiguous.
    f"{TFJS}/model.json": (
        "tfjs_model.json",
        "cbc10d46bb3c5cac268e55ec3e1314cf520dfc0b15b626025cee941876d5e67a",
    ),
}

# The 13 weight shards referenced by tfjs_model.json's weightsManifest.
TFJS_SHARDS = [f"group1-shard{i}of13.bin" for i in range(1, 14)]

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "birdnet"
MIRROR = CACHE / "_repo"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_repo() -> None:
    """Create (or reuse) a blob-less partial clone pinned at TAG."""
    if not (MIRROR / ".git").is_dir():
        MIRROR.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {REPO} @ {TAG} (blobless partial clone)...", flush=True)
        run(
            [
                "git", "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                "--depth", "1",
                "--branch", TAG,
                REPO,
                str(MIRROR),
            ]
        )
    else:
        print(f"reusing partial clone at {MIRROR}", flush=True)


def extract(repo_path: str, dest: Path) -> None:
    """Materialise one blob from the pinned tag."""
    with dest.open("wb") as fh:
        proc = subprocess.run(
            ["git", "cat-file", "-p", f"{TAG}:{repo_path}"],
            cwd=MIRROR,
            stdout=fh,
            stderr=subprocess.PIPE,
            text=False,
        )
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"could not extract {repo_path}: {proc.stderr.decode(errors='replace').strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-digests",
        action="store_true",
        help="print the sha256 of every artifact and exit (used to fill in ARTIFACTS)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-extract even if the file is already cached"
    )
    args = parser.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "tfjs").mkdir(exist_ok=True)
    ensure_repo()

    digests: dict[str, str] = {}
    failures: list[str] = []

    targets = dict(ARTIFACTS)
    for shard in TFJS_SHARDS:
        targets[f"{TFJS}/{shard}"] = (f"tfjs/{shard}", "")

    for repo_path, (name, expected) in targets.items():
        dest = CACHE / name
        if args.force or not dest.exists():
            print(f"extracting {name}...", flush=True)
            extract(repo_path, dest)
        got = sha256(dest)
        digests[name] = got
        size_mb = dest.stat().st_size / 1e6
        if expected and got != expected:
            failures.append(f"{name}: expected {expected}, got {got}")
            print(f"  {name}  {size_mb:8.2f} MB  CHECKSUM MISMATCH")
        else:
            state = "verified" if expected else "unpinned"
            print(f"  {name}  {size_mb:8.2f} MB  {state}")

    if args.print_digests:
        print("\n# paste into ARTIFACTS:")
        for name, digest in digests.items():
            print(f'#   {name}: "{digest}"')

    if failures:
        print("\nchecksum verification failed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    # The TFJS shards are content-addressed by the manifest they belong to, and the
    # model.json we pin covers them; we do not maintain 13 extra digests by hand.
    unpinned = [n for _, (n, e) in ARTIFACTS.items() if not e]
    if unpinned and not args.print_digests:
        print(f"\nnote: no pinned digest for: {', '.join(unpinned)}")

    print(f"\nartifacts ready in {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
