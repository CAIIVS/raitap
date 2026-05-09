"""
Demo sample datasets for quick-start usage.

These are curated small datasets downloaded on first use to ``~/.cache/raitap/``.
This file is intentionally separate from core data-loading logic — it exists only
to make the tool runnable out-of-the-box without any local data.

Preprocessing (resize to a fixed square) is applied here because demo images have
inconsistent source sizes. This does not affect consumer data, which is loaded raw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from raitap.data.utils import download_file

logger = logging.getLogger(__name__)

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
    # ISIC Archive — public CC-0 dermoscopic images (https://api.isic-archive.com)
    "isic2018": [
        (
            "https://isic-archive.s3.amazonaws.com/images/ISIC_0000000.jpg",
            "ISIC_0000000.jpg",
        ),
        (
            "https://isic-archive.s3.amazonaws.com/images/ISIC_0000001.jpg",
            "ISIC_0000001.jpg",
        ),
        (
            "https://isic-archive.s3.amazonaws.com/images/ISIC_0000002.jpg",
            "ISIC_0000002.jpg",
        ),
        (
            "https://isic-archive.s3.amazonaws.com/images/ISIC_0000003.jpg",
            "ISIC_0000003.jpg",
        ),
    ],
    # NIH malaria cell images — MIT-licensed sample sets (original data: US public domain)
    # Sources: HarshCasper/Malaria-Detection, prabhat-123/Malaria-Cell-Image-Classification
    "malaria": [
        (
            "https://raw.githubusercontent.com/HarshCasper/Malaria-Detection/master/samples/infected.png",
            "infected_1.png",
        ),
        (
            "https://raw.githubusercontent.com/HarshCasper/Malaria-Detection/master/samples/uninfected.png",
            "uninfected_1.png",
        ),
        (
            "https://raw.githubusercontent.com/prabhat-123/Malaria-Cell-Image-Classification/master/test_images/parasitized/0_sCZHuHECn0zkH3fR.png",
            "infected_2.png",
        ),
        (
            "https://raw.githubusercontent.com/prabhat-123/Malaria-Cell-Image-Classification/master/test_images/parasitized/1_Z6KEa0ZtwHUZjKOpRjlNLw.png",
            "infected_3.png",
        ),
    ],
    # ACAS Xu net 1-1 from VNN-COMP 2021 (Stanley Bak et al., BSD-3 license).
    # 5-input / 5-output MLP used as the canonical formal-verification fixture.
    # See https://github.com/stanleybak/vnncomp2021 for licence + provenance.
    "acas_xu_n1_1": [
        (
            "https://github.com/stanleybak/vnncomp2021/raw/main/benchmarks/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx",
            "ACASXU_run2a_1_1_batch_2000.onnx",
        ),
    ],
    # Udacity CarND dashcam road images — from udacity/CarND-Advanced-Lane-Lines (MIT)
    "UdacitySelfDriving": [
        (
            "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/test_images/straight_lines1.jpg",
            "straight_lines1.jpg",
        ),
        (
            "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/test_images/straight_lines2.jpg",
            "straight_lines2.jpg",
        ),
        (
            "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/test_images/test1.jpg",
            "test1.jpg",
        ),
        (
            "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/test_images/test2.jpg",
            "test2.jpg",
        ),
    ],
}

_CACHE_DIR = Path.home() / ".cache" / "raitap"

# Per-sample ground-truth labels keyed by image filename. Filled only for
# samples whose labels can be supplied honestly (e.g. ``imagenet_samples``
# matches a 1000-class ImageNet model). Other samples ship without labels —
# ``data.labels.source`` simply has nothing to resolve to.
SAMPLE_LABELS: dict[str, dict[str, int]] = {
    "imagenet_samples": {
        "tench.jpg": 0,
        "shih_tzu.jpg": 155,
        "golden_retriever.jpg": 207,
        "tiger_cat.jpg": 282,
    },
}

_LABELS_FILENAME = "labels.csv"


def _resolve_sample(name: str) -> Path | None:
    """
    Return the local cache path for a named demo dataset, downloading files if needed.

    Args:
        name: A key from ``SAMPLE_SOURCES`` (e.g. ``"imagenet_samples"``).

    Returns:
        Local directory path, or ``None`` if *name* is not a known sample.
    """
    if not isinstance(name, str) or name not in SAMPLE_SOURCES:
        return None

    cache_dir = _CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for url, filename in SAMPLE_SOURCES[name]:
        dest = cache_dir / filename
        if not dest.exists():
            logger.info("Downloading %s...", filename)
            download_file(url, dest)
    _materialise_sample_labels(name, cache_dir)
    return cache_dir


def _materialise_sample_labels(name: str, cache_dir: Path) -> None:
    """Write ``labels.csv`` into ``cache_dir`` for samples that ship labels.

    Rows are sorted by filename to match :func:`_load_sample`, which sorts
    image files alphabetically. This guarantees that even row-order label
    alignment (used when ``sample_ids`` is unavailable) produces the right
    label per image.
    """
    labels = SAMPLE_LABELS.get(name)
    if not labels:
        return
    dest = cache_dir / _LABELS_FILENAME
    if dest.exists():
        return
    rows = sorted(labels.items())
    lines = ["image,label", *(f"{filename},{idx}" for filename, idx in rows)]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_sample_labels_path(name: str) -> Path | None:
    """Return the labels CSV for a known sample, or ``None`` if it has none."""
    if name not in SAMPLE_LABELS:
        return None
    cache_dir = _resolve_sample(name)
    if cache_dir is None:
        return None
    return cache_dir / _LABELS_FILENAME


# Default resize for demo images. Source images have inconsistent dimensions,
# so we normalise them here to allow stacking. Consumer data is never resized.
_DEMO_SIZE = 224


def _load_sample(name: str, size: int = _DEMO_SIZE) -> tuple[torch.Tensor, list[str]]:
    """
    Load a named demo dataset as a resized tensor plus per-row sample IDs.

    Downloads files if needed, then resizes each image to ``(size, size)`` so
    they can be stacked into a batch. **Only used for demo samples** — consumer
    data is returned raw by :func:`~raitap.data.loader.load_data`.

    Args:
        name: A key from ``SAMPLE_SOURCES`` (e.g. ``"imagenet_samples"``).
        size: Edge length to resize images to (default 224).

    Returns:
        Tuple of ``(tensor, sample_ids)`` where ``tensor`` is float32 with
        shape ``(N, 3, size, size)`` in ``[0, 1]`` and ``sample_ids`` lists
        the source filenames in the same row order, so ``data.labels.source``
        can align labels by filename.
    """
    directory = _resolve_sample(name)
    if directory is None:
        raise ValueError(f"{name!r} is not a known demo sample.")

    files = sorted(
        f
        for f in directory.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    tensors = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(img)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors), [f.name for f in files]
