"""
Demo sample datasets for quick-start usage.

These are curated small datasets downloaded on first use to ``~/.cache/raitap/``.
This file is intentionally separate from core data-loading logic — it exists only
to make the tool runnable out-of-the-box without any local data.

Preprocessing (resize to a fixed square) is applied here because demo images have
inconsistent source sizes. This does not affect consumer data, which is loaded raw.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Registry of named demo datasets
# ---------------------------------------------------------------------------

SAMPLE_SOURCES: dict[str, list[tuple[str, str]]] = {
    "imagenet_samples": [
        (
            "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02099601_golden_retriever.JPEG",
            "golden_retriever.jpg",
        ),
        (
            "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02123159_tiger_cat.JPEG",
            "tiger_cat.jpg",
        ),
        (
            "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01440764_tench.JPEG",
            "tench.jpg",
        ),
        (
            "https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n02086240_Shih-Tzu.JPEG",
            "shih_tzu.jpg",
        ),
    ],
}

_CACHE_DIR = Path.home() / ".cache" / "raitap"


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def resolve_sample(name: str) -> Path | None:
    """
    Return the local cache path for a named demo dataset, downloading files if needed.

    Args:
        name: A key from ``SAMPLE_SOURCES`` (e.g. ``"imagenet_samples"``).

    Returns:
        Local directory path, or ``None`` if *name* is not a known sample.
    """
    if name not in SAMPLE_SOURCES:
        return None

    cache_dir = _CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for url, filename in SAMPLE_SOURCES[name]:
        dest = cache_dir / filename
        if not dest.exists():
            print(f"  Downloading {filename}...")
            _download_file(url, dest)
    return cache_dir


# Default resize for demo images. Source images have inconsistent dimensions,
# so we normalise them here to allow stacking. Consumer data is never resized.
_DEMO_SIZE = 224


def load_sample(name: str, size: int = _DEMO_SIZE) -> torch.Tensor:
    """
    Load a named demo dataset as a resized tensor.

    Downloads files if needed, then resizes each image to ``(size, size)`` so
    they can be stacked into a batch. **Only used for demo samples** — consumer
    data is returned raw by :func:`~raitap.data.loader.load_data`.

    Args:
        name: A key from ``SAMPLE_SOURCES`` (e.g. ``"imagenet_samples"``).
        size: Edge length to resize images to (default 224).

    Returns:
        Float32 tensor of shape ``(N, 3, size, size)`` in ``[0, 1]``.
    """
    directory = resolve_sample(name)
    if directory is None:
        raise ValueError(f"{name!r} is not a known demo sample.")

    files = sorted(
        f
        for f in directory.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    tensors = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(img)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors)
