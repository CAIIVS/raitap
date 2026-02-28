"""Core data loading: resolve a source to a local path, then read files into raw tensors.

No preprocessing (normalization, cropping, resizing) is applied — that is the
responsibility of the consumer (model pipeline, user code, etc.).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .samples import SAMPLE_SOURCES, load_sample, resolve_sample

_CACHE_DIR = Path.home() / ".cache" / "raitap"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def resolve_data_source(source: str) -> Path:
    """
    Resolve *source* to a local path.

    Resolution order:

    1. Named demo sample (e.g. ``"imagenet_samples"``) → download to cache.
    2. URL (``http://`` / ``https://``) → download single file to cache.
    3. Existing local path → returned as-is.

    Args:
        source: Named sample, URL, or local path.

    Returns:
        Local :class:`~pathlib.Path` (file or directory).

    Raises:
        ValueError: If *source* cannot be resolved.
    """
    # 1. Named demo sample
    sample_path = resolve_sample(source)
    if sample_path is not None:
        return sample_path

    # 2. URL
    if source.startswith(("http://", "https://")):
        filename = source.rstrip("/").split("/")[-1] or "download"
        dest = _CACHE_DIR / "downloads" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"  Downloading {filename}...")
            _download_file(source, dest)
        return dest

    # 3. Local path
    path = Path(source)
    if path.exists():
        return path

    known = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
    raise ValueError(
        f"Data source {source!r} could not be resolved.\n"
        f"Expected a URL, a local path, or a named demo sample.\n"
        f"Known demo samples: {known}"
    )


def _load_images(path: Path) -> torch.Tensor:
    """Load image files from a directory (or a single file) as raw (C, H, W) tensors."""
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No image files found in {path}")
    else:
        files = [path]

    tensors = []
    for f in files:
        arr = np.array(Image.open(f).convert("RGB"))
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

    try:
        return torch.stack(tensors)
    except RuntimeError:
        sizes = {tuple(t.shape) for t in tensors}
        raise ValueError(
            f"Images have inconsistent shapes: {sizes}. "
            "Resize them to a common size before loading."
        )


def _load_tabular(path: Path) -> torch.Tensor:
    """Load a CSV / TSV / Parquet file as a float tensor of shape (N, F)."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    return torch.tensor(df.values, dtype=torch.float32)


def load_data(source: str) -> torch.Tensor:
    """
    Resolve *source* and load its contents as a raw tensor.

    No preprocessing is applied. Normalisation, cropping, and other
    transforms are the responsibility of the caller.

    Supported sources:

    - Named demo sample (e.g. ``"imagenet_samples"``)
    - URL to a single downloadable file
    - Local directory of images → ``(N, 3, H, W)`` float32 in ``[0, 1]``
    - Local image file → ``(1, 3, H, W)`` float32 in ``[0, 1]``
    - Local CSV / TSV / Parquet file → ``(N, F)`` float32

    Args:
        source: Named sample, URL, or local path.

    Returns:
        Raw data tensor.
    """
    # Demo samples are preprocessed (resized) in samples.py — see module docstring.
    if source in SAMPLE_SOURCES:
        return load_sample(source)

    path = resolve_data_source(source)

    if path.is_dir():
        return _load_images(path)

    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return _load_images(path)
    if suffix in _TABULAR_EXTENSIONS:
        return _load_tabular(path)

    raise ValueError(
        f"Cannot infer data type from extension {suffix!r}.\n"
        f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
        f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
    )
