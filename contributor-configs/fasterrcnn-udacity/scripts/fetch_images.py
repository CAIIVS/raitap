"""Materialise the 4 ``UdacitySelfDriving`` sample images at native resolution.

RAITAP's demo sample loader (``raitap.data.samples._load_sample``) hard-resizes
images to ``_DEMO_SIZE=224`` so heterogeneous samples can be stacked into one
tensor. Detection models need NATIVE resolution (1280x720) — labels are in that
coord space and the model expects realistic input dimensions for COCO-pretrained
weights to fire.

This script downloads (or hits the existing cache at
``~/.cache/raitap/UdacitySelfDriving/``) and copies the 4 images to
``contributor-configs/fasterrcnn-udacity/images/`` so the assessment config can
point ``data.source`` at a normal directory (no demo-loader resize).

Usage::

    uv run python contributor-configs/fasterrcnn-udacity/scripts/fetch_images.py

Idempotent — safe to run multiple times.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from raitap.data.samples import _resolve_sample

SAMPLE_NAME = "UdacitySelfDriving"
SAMPLE_FILES = [
    "straight_lines1.jpg",
    "straight_lines2.jpg",
    "test1.jpg",
    "test2.jpg",
]
TARGET = Path(__file__).resolve().parent.parent / "images"


def main() -> None:
    cache_dir = _resolve_sample(SAMPLE_NAME)
    if cache_dir is None:
        raise SystemExit(f"Sample {SAMPLE_NAME!r} unknown to raitap.data.samples.")
    TARGET.mkdir(parents=True, exist_ok=True)
    for filename in SAMPLE_FILES:
        src = cache_dir / filename
        dst = TARGET / filename
        if not src.exists():
            raise SystemExit(f"Source image missing after cache resolve: {src}")
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"= {filename} (already present)")
            continue
        shutil.copy2(src, dst)
        print(f"+ {filename}")
    print(f"\nReady: {TARGET}")


if __name__ == "__main__":
    main()
