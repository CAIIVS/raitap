from __future__ import annotations

import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from raitap.data.utils import download_file
from raitap.tracking.base_tracker import BaseTracker, Trackable

from .samples import SAMPLE_SOURCES, _load_sample

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from raitap.configs.schema import AppConfig


_CACHE_DIR = Path.home() / ".cache" / "raitap"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}


class Data(Trackable):
    def __init__(self, cfg: AppConfig) -> None:
        self.name = cfg.data.name
        self.source = cfg.data.source
        self.tensor, self.sample_ids = self._load_data(cfg)
        self.labels = self._load_labels(cfg)

    def _load_data(self, cfg: AppConfig) -> tuple[torch.Tensor, list[str] | None]:
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
        if not source or not source.strip():
            raise ValueError(
                "No data source specified. Set data.source in your config.\n"
                "Use a local path or a named sample set, e.g.: data=imagenet_samples"
            )

        if source in SAMPLE_SOURCES:
            return _load_sample(source), None

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
            image_files = _list_images_recursive(path)
            tabular_files = _list_tabular_recursive(path)
            if image_files and tabular_files:
                raise ValueError(
                    f"Directory {path} contains both image and tabular files. "
                    "Separate them into different directories."
                )
            if image_files:
                tensor = torch.from_numpy(_stack_images_numpy(image_files))
                return tensor, _resolve_sample_ids(image_files, root=path)
            if tabular_files:
                return torch.from_numpy(_concat_tabular_numpy(tabular_files)), None
            raise FileNotFoundError(
                f"No supported files found in {path}.\n"
                f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
                f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
            )

        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            return _load_images(path), _resolve_sample_ids([path], root=path.parent)
        if suffix in _TABULAR_EXTENSIONS:
            return _load_tabular(path), None

        raise ValueError(
            f"Cannot infer data type from extension {suffix!r}.\n"
            f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
            f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
        )

    def _load_labels(self, cfg: AppConfig) -> torch.Tensor | None:
        labels_cfg = _get_optional_config_value(cfg.data, "labels")
        labels_source = _get_optional_config_value(labels_cfg, "source")
        if not labels_source:
            return None

        labels_path = get_source_path(labels_source)
        labels_df = _load_tabular_frame(labels_path)
        if labels_df.empty:
            warnings.warn(
                "Labels file is empty; falling back to predictions as targets.", stacklevel=2
            )
            return None

        labels_id_column = _get_optional_config_value(labels_cfg, "id_column")
        id_column = _resolve_labels_id_column(labels_df, labels_id_column)
        labels_column = _get_optional_config_value(labels_cfg, "column")
        labels_encoding = _get_optional_config_value(labels_cfg, "encoding")
        labels_id_strategy = _get_optional_config_value(labels_cfg, "id_strategy") or "auto"
        encoded_labels = _extract_class_labels(
            labels_df,
            labels_column=labels_column,
            id_column=id_column,
            labels_encoding=labels_encoding,
        )

        expected = int(self.tensor.shape[0])
        if self.sample_ids and id_column:
            id_series = _column_as_series(labels_df, id_column)
            strategy = _resolve_id_strategy(labels_id_strategy, id_series)
            try:
                aligned_labels = _align_labels_to_samples(
                    sample_ids=self.sample_ids,
                    raw_label_ids=id_series,
                    encoded_labels=encoded_labels,
                    strategy=strategy,
                )
            except ValueError as error:
                warnings.warn(
                    f"{error} Falling back to predictions as metric targets.",
                    stacklevel=2,
                )
                return None
            return torch.tensor(aligned_labels, dtype=torch.long)

        if self.sample_ids and not id_column:
            warnings.warn(
                "Could not find a labels id column for filename alignment; using row-order labels.",
                stacklevel=2,
            )

        if len(encoded_labels) != expected:
            warnings.warn(
                f"Label count ({len(encoded_labels)}) does not match sample count ({expected}); "
                "falling back to predictions as targets.",
                stacklevel=2,
            )
            return None

        return torch.tensor(encoded_labels, dtype=torch.long)

    def describe(self) -> dict[str, Any]:
        """
        Build standard dataset metadata for tracking and reporting.

        Returns:
            Dictionary containing dataset metadata.
        """
        shape = [int(dim) for dim in self.tensor.shape]
        dataset_info: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "num_samples": shape[0],
            "shape": shape,
            "dtype": str(self.tensor.dtype),
            "has_labels": self.labels is not None,
        }
        if len(shape) > 1:
            dataset_info["sample_shape"] = shape[1:]
        return dataset_info

    def log(self, tracker: BaseTracker, **kwargs: Any) -> None:
        """Log dataset metadata to the tracker."""
        tracker.log_dataset(self.describe())


def load_tensor_from_source(source: str, n_samples: int | None = None) -> torch.Tensor:
    """
    Load a raw tensor from a named demo sample, URL, or local path.

    This is the same loading logic used by :class:`Data`, but without label handling.
    Useful for loading background data for SHAP explainers.

    Args:
        source: Named demo sample (e.g. ``"imagenet_samples"``), URL, or local path.
        n_samples: If set, randomly subsample *n_samples* rows from the loaded tensor.
            Useful for keeping background datasets small (e.g. for KernelExplainer).

    Returns:
        Raw tensor of shape ``(N, ...)`` where *N* is the number of samples.

    Raises:
        ValueError: If *source* cannot be resolved, does not exist,
            or the file type is not supported.
    """
    if source in SAMPLE_SOURCES:
        tensor = _load_sample(source)
    elif source.startswith(("http://", "https://")):
        path = get_source_path(source)
        tensor = _load_tensor_from_path(path)
    else:
        path = Path(source)
        if not path.exists():
            demo_samples = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
            raise ValueError(
                f"Data source {source!r} does not exist.\n"
                f"Expected a URL, an existing local path, or a named demo sample.\n"
                f"Known demo samples: {demo_samples}"
            )
        tensor = _load_tensor_from_path(path)

    if n_samples is not None and tensor.shape[0] > n_samples:
        indices = torch.randperm(tensor.shape[0])[:n_samples]
        tensor = tensor[indices]

    return tensor


def load_numpy_from_source(source: str, n_samples: int | None = None) -> np.ndarray[Any, Any]:
    """
    Load data as a NumPy array using the same resolution rules as :func:`load_tensor_from_source`.

    For file-based sources (local paths and URLs), no intermediate torch tensor is allocated.
    Demo sample sources (``SAMPLE_SOURCES``) use ``raitap.data.samples._load_sample`` (torch-based);
    all other paths are torch-free.
    """
    if source in SAMPLE_SOURCES:
        arr: np.ndarray[Any, Any] = _load_sample(source).numpy()
    elif source.startswith(("http://", "https://")):
        path = get_source_path(source)
        arr = _load_numpy_from_path(path)
    else:
        path = Path(source)
        if not path.exists():
            demo_samples = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
            raise ValueError(
                f"Data source {source!r} does not exist.\n"
                f"Expected a URL, an existing local path, or a named demo sample.\n"
                f"Known demo samples: {demo_samples}"
            )
        arr = _load_numpy_from_path(path)

    if n_samples is not None and arr.shape[0] > n_samples:
        rng = np.random.default_rng()
        indices = rng.choice(arr.shape[0], size=n_samples, replace=False)
        arr = arr[indices]

    return arr


def _load_numpy_from_path(path: Path) -> np.ndarray[Any, Any]:
    """Load a numpy array from a single file or directory (no sample IDs returned)."""
    if path.is_dir():
        image_files = _list_images_recursive(path)
        tabular_files = _list_tabular_recursive(path)
        if image_files and tabular_files:
            raise ValueError(
                f"Directory {path} contains both image and tabular files. "
                "Separate them into different directories."
            )
        if image_files:
            return _stack_images_numpy(image_files)
        if tabular_files:
            return _concat_tabular_numpy(tabular_files)
        raise FileNotFoundError(
            f"No supported files found in {path}.\n"
            f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
            f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
        )

    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return _load_images_numpy(path)
    if suffix in _TABULAR_EXTENSIONS:
        return _load_tabular_numpy(path)
    raise ValueError(
        f"Cannot infer data type from extension {suffix!r}.\n"
        f"Supported image formats: {_IMAGE_EXTENSIONS}\n"
        f"Supported tabular formats: {_TABULAR_EXTENSIONS}"
    )


def _load_tensor_from_path(path: Path) -> torch.Tensor:
    """Load a tensor from a single file or a directory (no sample IDs returned)."""
    return torch.from_numpy(_load_numpy_from_path(path))


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
            logger.info("Downloading %s...", filename)
            download_file(source, dest)
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


def _list_images_recursive(root: Path) -> list[Path]:
    """Discover image files under ``root`` recursively, sorted by relative posix path."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _list_tabular_recursive(root: Path) -> list[Path]:
    """Discover tabular files under ``root`` recursively, sorted by relative posix path."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TABULAR_EXTENSIONS]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _stack_images_numpy(files: list[Path]) -> np.ndarray[Any, Any]:
    """Stack pre-discovered image files into an NCHW float32 array in [0, 1]."""
    arrays = []
    for f in files:
        arr = np.array(Image.open(f).convert("RGB"))  # HWC uint8
        arrays.append(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)  # CHW float32

    try:
        return np.stack(arrays)  # NCHW float32
    except ValueError:
        sizes = {arr.shape for arr in arrays}
        raise ValueError(
            f"Images have inconsistent shapes: {sizes}. "
            "Resize them to a common size before loading."
        ) from None


def _load_images_numpy(path: Path) -> np.ndarray[Any, Any]:
    """Load image files from a directory (or a single file) as NCHW float32 arrays in [0, 1]."""
    if path.is_dir():
        files = _list_images_recursive(path)
        if not files:
            raise FileNotFoundError(f"No image files found in {path}")
    else:
        files = [path]
    return _stack_images_numpy(files)


def _load_images(path: Path) -> torch.Tensor:
    """Load image files from a directory (or a single file) as raw (C, H, W) tensors."""
    return torch.from_numpy(_load_images_numpy(path))


def _load_tabular_numpy(path: Path) -> np.ndarray[Any, Any]:
    """Load a CSV / TSV / Parquet file as a float32 numpy array of shape (N, F)."""
    df = _load_tabular_frame(path)
    return np.array(df.values, dtype=np.float32)


def _load_tabular(path: Path) -> torch.Tensor:
    """Load a CSV / TSV / Parquet file as a float tensor of shape (N, F)."""
    return torch.from_numpy(_load_tabular_numpy(path))


def _load_tabular_frame(path: Path) -> pd.DataFrame:
    """Load a CSV / TSV / Parquet file as a pandas DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")
    return df


def _get_optional_config_value(config: Any, key: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _resolve_labels_id_column(df: pd.DataFrame, configured_column: str | None) -> str | None:
    if configured_column:
        if configured_column not in df.columns:
            warnings.warn(
                f"Configured data.labels.id_column {configured_column!r} not found in labels file.",
                stacklevel=2,
            )
            return None
        return configured_column

    for candidate in ("image", "filename", "file", "id", "name"):
        if candidate in df.columns:
            return candidate
    return None


def _extract_class_labels(
    df: pd.DataFrame,
    labels_column: str | None,
    id_column: str | None,
    labels_encoding: str | None,
) -> list[int]:
    encoding = (labels_encoding or "").strip().lower()
    if encoding and encoding not in {"index", "one_hot", "argmax"}:
        raise ValueError(
            "Unsupported data.labels.encoding "
            f"{labels_encoding!r}. Use 'index', 'one_hot', or 'argmax'."
        )

    if labels_column:
        if labels_column not in df.columns:
            raise ValueError(f"data.labels.column {labels_column!r} not found in labels file")
        label_series = _column_as_series(df, labels_column)
        numeric_values = pd.to_numeric(label_series, errors="raise")
        if not isinstance(numeric_values, pd.Series):
            raise ValueError("Expected a pandas Series for label conversion.")
        if encoding == "one_hot":
            raise ValueError(
                "data.labels.column cannot be combined with data.labels.encoding='one_hot'"
            )
        return [int(value) for value in numeric_values.to_list()]

    excluded = {id_column} if id_column else set()
    candidate_columns = [
        col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not candidate_columns:
        raise ValueError(
            "Could not infer label columns. "
            "Set data.labels.column or provide numeric one-hot columns."
        )

    matrix = df[candidate_columns].to_numpy()
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Labels file does not contain valid label columns")

    if encoding == "index" and matrix.shape[1] != 1:
        raise ValueError(
            "data.labels.encoding='index' requires exactly one numeric label column "
            "(or set data.labels.column)."
        )

    if matrix.shape[1] == 1:
        if encoding == "one_hot":
            raise ValueError(
                "data.labels.encoding='one_hot' requires multiple numeric label columns "
                "to represent one-hot targets."
            )
        label_series = _column_as_series(df, candidate_columns[0])
        numeric_values = pd.to_numeric(label_series, errors="raise")
        if not isinstance(numeric_values, pd.Series):
            raise ValueError("Expected a pandas Series for label conversion.")
        return [int(value) for value in numeric_values.to_list()]

    return matrix.argmax(axis=1).astype(int).tolist()


def _resolve_sample_ids(files: list[Path], root: Path) -> list[str]:
    """Sample ids are posix-style paths relative to ``root`` (extension included)."""
    return sorted(p.relative_to(root).as_posix() for p in files)


def _normalise_sample_id(value: object, strategy: str = "stem") -> str:
    """Normalise a label-file id or discovered sample id into a comparable key.

    - ``strategy="stem"``: legacy behaviour. Strip the directory and the
      extension; e.g. ``"NORMAL/IM-0001.jpeg"`` → ``"IM-0001"``.
    - ``strategy="relative_path"``: keep the directory; strip only the
      extension. e.g. ``"NORMAL\\IM-0001.jpeg"`` → ``"NORMAL/IM-0001"``.
    """
    text = str(value).strip().replace("\\", "/")
    p = Path(text)
    if strategy == "relative_path":
        return p.with_suffix("").as_posix()
    return p.stem


def _resolve_id_strategy(strategy: str, raw_label_ids: pd.Series) -> str:
    if strategy == "auto":
        for raw in raw_label_ids.tolist():
            text = str(raw)
            if "/" in text or "\\" in text:
                return "relative_path"
        return "stem"
    if strategy in {"relative_path", "stem"}:
        return strategy
    raise ValueError(
        f"Unsupported data.labels.id_strategy {strategy!r}. Use 'auto', 'relative_path', or 'stem'."
    )


def _column_as_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    column_data = df[column_name]
    if isinstance(column_data, pd.DataFrame):
        raise ValueError(
            f"Column name {column_name!r} is duplicated in labels file; names must be unique."
        )
    return column_data


def _align_labels_to_samples(
    sample_ids: list[str],
    raw_label_ids: pd.Series,
    encoded_labels: list[int],
    strategy: str = "stem",
) -> list[int]:
    normalised_sample_ids = [_normalise_sample_id(sid, strategy) for sid in sample_ids]
    normalised_label_ids = [
        _normalise_sample_id(raw_id, strategy) for raw_id in raw_label_ids.tolist()
    ]
    duplicates = sorted(
        [row_id for row_id, count in Counter(normalised_label_ids).items() if count > 1]
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Duplicate label IDs detected ({preview}{'...' if len(duplicates) > 5 else ''})."
        )

    label_by_id = {
        row_id: int(label)
        for row_id, label in zip(normalised_label_ids, encoded_labels, strict=False)
    }
    missing_ids = [sid for sid in normalised_sample_ids if sid not in label_by_id]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            "Missing labels for some sample IDs "
            f"({preview}{'...' if len(missing_ids) > 5 else ''})."
        )
    return [label_by_id[sid] for sid in normalised_sample_ids]


def _concat_tabular_numpy(files: list[Path]) -> np.ndarray[Any, Any]:
    """Concatenate pre-discovered tabular files row-wise into a float32 array."""
    arrays = [_load_tabular_numpy(f) for f in files]
    try:
        return np.concatenate(arrays)
    except ValueError:
        shapes = {arr.shape for arr in arrays}
        raise ValueError(
            f"Tabular files have inconsistent column counts: {shapes}. "
            "All files must have the same number of columns."
        ) from None


def _load_tabular_dir_numpy(path: Path) -> np.ndarray[Any, Any]:
    """Load all tabular files from a directory as a single float32 array, concatenating rows."""
    return _concat_tabular_numpy(_list_tabular_recursive(path))


def _load_tabular_dir(path: Path) -> torch.Tensor:
    """Load all tabular files from a directory as a single float tensor, concatenating rows."""
    return torch.from_numpy(_load_tabular_dir_numpy(path))
