from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig
    from raitap.tracking import BaseTracker

from .samples import SAMPLE_SOURCES, _load_sample

_CACHE_DIR = Path.home() / ".cache" / "raitap"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}


class Data:
    def __init__(self, cfg: AppConfig) -> None:
        self.source = cfg.data.source
        self.tensor = self._load_data(cfg)

    def _load_data(self, cfg: AppConfig) -> torch.Tensor:
        """
        Load data from a specified source into a raw tensor.

        Supported sources:

        - Named demo sample (e.g. ``"imagenet_samples"``)
        - URL to a single downloadable file
        - Local directory of images → ``(N, 3, H, W)`` float32 in ``[0, 1]``
        - Local directory of CSV / TSV / Parquet files → ``(N, F)`` float32 (rows concatenated)
        - Local image file → ``(1, 3, H, W)`` float32 in ``[0, 1]``
        - Local CSV / TSV / Parquet file → ``(N, F)`` float32

        Args:
            cfg: Application configuration.

        Returns:
            Raw data tensor.
        """
        source = cfg.data.source
        if not source:
            raise ValueError(
                "No data source specified. Set data.source in your config.\n"
                "Use a local path or a named sample set, e.g.: data=imagenet_samples"
            )

        if source in SAMPLE_SOURCES:
            return _load_sample(source)

        if source.startswith(("http://", "https://")):
            path = get_source_path(source)
        else:
            path = Path(source)
            if not path.exists():
                demo_samples = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
                raise ValueError(
                    f"Data source {source!r} does not exist.\n"
                    f"Expected a URL, an existing local path, or a named demo sample.\n"
                    f"Known demo samples: {demo_samples}"
                )

        if path.is_dir():
            all_files = list(path.iterdir())
            image_files = [f for f in all_files if f.suffix.lower() in _IMAGE_EXTENSIONS]
            tabular_files = [f for f in all_files if f.suffix.lower() in _TABULAR_EXTENSIONS]
            if image_files and tabular_files:
                raise ValueError(
                    f"Directory {path} contains both image and tabular files. "
                    "Separate them into different directories."
                )
            if image_files:
                return _load_images(path)
            if tabular_files:
                return _load_tabular_dir(path)
            raise FileNotFoundError(
                f"No supported files found in {path}.\n"
                f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
                f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
            )

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

    def describe(self) -> dict[str, Any]:
        """
        Build standard dataset metadata for tracking and reporting.

        Returns:
            Dictionary containing dataset metadata.
        """
        shape = [int(dim) for dim in self.tensor.shape]
        dataset_info: dict[str, Any] = {
            "name": getattr(self, "name", "dataset"),
            "source": self.source,
            "num_samples": shape[0],
            "shape": shape,
            "dtype": str(self.tensor.dtype),
        }
        if len(shape) > 1:
            dataset_info["sample_shape"] = shape[1:]
        return dataset_info

    def log(self, tracker: BaseTracker) -> None:
        """Log dataset metadata to the tracker."""
        tracker.log_dataset(self.describe())


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())


def get_source_path(source: str) -> Path:
    """
    Obtain the local path to the specified source.
    There are two possible cases:
    1. URL (``http://`` / ``https://``) → download to a cache, of which the path is returned.
    2. Existing local path → returned as-is.


    Args:
        source: URL or local path.

    Returns:
        Local :class:`~pathlib.Path` (file or directory).

    Raises:
        ValueError: If *source* cannot be resolved.
    """
    if source.startswith(("http://", "https://")):
        filename = source.rstrip("/").split("/")[-1] or "download"
        dest = _CACHE_DIR / "downloads" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"  Downloading {filename}...")
            _download_file(source, dest)
        return dest

    path = Path(source)
    if path.exists():
        return path

    demo_samples = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
    raise ValueError(
        f"Data source {source!r} could not be resolved.\n"
        f"Expected a URL or a local path.\n"
        f"For named demo samples use load_data directly. Known demo samples: {demo_samples}"
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
        ) from None


def _load_tabular(path: Path) -> torch.Tensor:
    """Load a CSV / TSV / Parquet file as a float tensor of shape (N, F)."""
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


def _load_tabular_dir(path: Path) -> torch.Tensor:
    """Load all tabular files from a directory as a single float tensor, concatenating rows."""
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in _TABULAR_EXTENSIONS)
    tensors = [_load_tabular(f) for f in files]
    try:
        return torch.cat(tensors)
    except RuntimeError:
        shapes = {tuple(t.shape) for t in tensors}
        raise ValueError(
            f"Tabular files have inconsistent column counts: {shapes}. "
            "All files must have the same number of columns."
        ) from None
