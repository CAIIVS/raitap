from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from PIL import Image

from raitap import raitap_log
from raitap.data.preprocessing import module_as_per_image_callable, resolve_preprocessing
from raitap.data.types import (
    MODALITY_EXTENSIONS,
    IdStrategy,
    InputModality,
    LabelEncoding,
)
from raitap.data.utils import download_file
from raitap.tracking.base_tracker import BaseTracker, Trackable
from raitap.types import DetectionInputs, TaskKind
from raitap.utils.lazy import lazy_import

from .samples import SAMPLE_SOURCES, _load_sample, resolve_sample_labels_path

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from raitap.configs.schema import AppConfig
    from raitap.data.preprocessing import ResolvedPreprocessing
else:
    torch = lazy_import("torch")


_CACHE_DIR = Path.home() / ".cache" / "raitap"
_IMAGE_EXTENSIONS = MODALITY_EXTENSIONS[InputModality.image]
_TABULAR_EXTENSIONS = MODALITY_EXTENSIONS[InputModality.tabular]
# Pre-formatted for error messages: deterministic order, no ``frozenset(...)`` repr.
_IMAGE_FORMATS = ", ".join(sorted(_IMAGE_EXTENSIONS))
_TABULAR_FORMATS = ", ".join(sorted(_TABULAR_EXTENSIONS))


class SourceKind(StrEnum):
    """What :func:`get_source_path` should resolve for a demo sample or parallel config field."""

    DATA = "data"
    LABELS = "labels"


class Data(Trackable):
    def __init__(
        self,
        cfg: AppConfig,
        *,
        resolved_preprocessing: ResolvedPreprocessing | None = None,
        task_kind: TaskKind = TaskKind.classification,
    ) -> None:
        from raitap.task_families import resolve_task_family

        self.name = cfg.data.name
        self.source = cfg.data.source
        self.task_kind = task_kind
        self.tensor: torch.Tensor | DetectionInputs
        family = resolve_task_family(task_kind)
        self.input_modality: InputModality
        raw_tensor, self.sample_ids, self.input_modality = self._load_data(
            cfg,
            resolved_preprocessing=resolved_preprocessing,
        )
        self.tensor = family.adapt_loaded_inputs(raw_tensor)
        family.validate_inputs(self.tensor)
        self.labels: torch.Tensor | list[dict[str, torch.Tensor]] | None
        self.labels = _resolve_and_parse_labels(
            cfg, task_kind=self.task_kind, tensor=self.tensor, sample_ids=self.sample_ids
        )
        family.validate_labels(self.labels)

    def _load_data(
        self,
        cfg: AppConfig,
        *,
        resolved_preprocessing: ResolvedPreprocessing | None = None,
    ) -> tuple[torch.Tensor | DetectionInputs, list[str] | None, InputModality]:
        """
        Load data from a specified source into a raw tensor.

        Supported sources:

        - Named demo sample (e.g. ``"imagenet_samples"``)
        - URL to a single downloadable file
        - Local directory of images → ``(N, 3, H, W)`` float32 in ``[0, 1]``
          (or ``list[Tensor]`` of per-image ``(C, H, W)`` for detection)
        - Local directory of CSV / TSV / Parquet files → ``(N, F)`` float32 (rows concatenated)
        - Local image file → ``(1, 3, H, W)`` float32 in ``[0, 1]``
          (or ``list[Tensor]`` of length 1 for detection)
        - Local CSV / TSV / Parquet file → ``(N, F)`` float32

        Args:
            cfg: Application configuration.

        Returns:
            ``(tensor, sample_ids, input_modality)`` where ``tensor`` is the raw
            data tensor (or a ragged ``list[Tensor]`` of per-image tensors for
            detection), ``sample_ids`` are posix-relative file ids (``None`` for
            tabular), and ``input_modality`` is the recorded
            :class:`~raitap.data.types.InputModality`.
        """
        source = cfg.data.source
        if not source or not source.strip():
            raise ValueError(
                "No data source specified. Set data.source in your config.\n"
                "Use a local path or a named sample set, e.g.: data=imagenet_samples"
            )

        # Use the run-level resolution when the orchestrator supplies it. Direct
        # ``Data(config)`` callers keep the legacy fallback path.
        if resolved_preprocessing is not None:
            data_module = resolved_preprocessing.data_module
        else:
            model_cfg = getattr(cfg, "model", None)
            if model_cfg is None:
                data_module = None
            else:
                resolved = resolve_preprocessing(model_cfg, cfg.data)
                data_module = resolved.data_module
        per_image_transform = module_as_per_image_callable(data_module)

        is_detection = self.task_kind is TaskKind.detection

        # Demo samples need their own loader: source images have inconsistent
        # dimensions and ``_load_sample`` resizes them to a common shape so
        # they can be stacked. The generic dir-loader expects pre-aligned
        # shapes and would fail on raw demo images.
        if source in SAMPLE_SOURCES:
            # Forward the per-image transform so ``_load_sample`` loads images
            # at their native resolution instead of pre-squashing them to
            # ``_DEMO_SIZE`` before the bundled Resize/CenterCrop sees them.
            tensor, sample_ids = _load_sample(source, per_image_transform=per_image_transform)
            return tensor, sample_ids, InputModality.image

        path = get_source_path(source, kind=SourceKind.DATA)

        if path.is_dir():
            image_files = _list_images_recursive(path)
            tabular_files = _list_tabular_recursive(path)
            if image_files and tabular_files:
                raise ValueError(
                    f"Directory {path} contains both image and tabular files. "
                    "Separate them into different directories."
                )
            if image_files:
                if is_detection:
                    ragged = _load_images_ragged(
                        image_files, per_image_transform=per_image_transform
                    )
                    return (
                        ragged,
                        _resolve_sample_ids(image_files, root=path),
                        InputModality.image,
                    )
                tensor = torch.from_numpy(
                    _stack_images_numpy(image_files, per_image_transform=per_image_transform)
                )
                return tensor, _resolve_sample_ids(image_files, root=path), InputModality.image
            if tabular_files:
                tensor = torch.from_numpy(_concat_tabular_numpy(tabular_files))
                return (
                    _apply_data_module_to_batch(data_module, tensor),
                    None,
                    InputModality.tabular,
                )
            raise FileNotFoundError(
                f"No supported files found in {path}.\n"
                f"Supported image formats: {_IMAGE_FORMATS}\n"
                f"Supported tabular formats: {_TABULAR_FORMATS}"
            )

        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            if is_detection:
                ragged = _load_images_ragged([path], per_image_transform=per_image_transform)
                return ragged, _resolve_sample_ids([path], root=path.parent), InputModality.image
            return (
                _load_images(path, per_image_transform=per_image_transform),
                _resolve_sample_ids([path], root=path.parent),
                InputModality.image,
            )
        if suffix in _TABULAR_EXTENSIONS:
            return (
                _apply_data_module_to_batch(data_module, _load_tabular(path)),
                None,
                InputModality.tabular,
            )

        raise ValueError(
            f"Cannot infer data type from extension {suffix!r}.\n"
            f"Supported image formats: {_IMAGE_FORMATS}\n"
            f"Supported tabular formats: {_TABULAR_FORMATS}"
        )

    def describe(self) -> dict[str, Any]:
        """
        Build standard dataset metadata for tracking and reporting.

        Returns:
            Dictionary containing dataset metadata.
        """
        if isinstance(self.tensor, list):
            # Ragged detection batch: report count and dtype only; no fixed shape.
            dtype_str = str(self.tensor[0].dtype) if self.tensor else "unknown"
            return {
                "name": self.name,
                "source": self.source,
                "num_samples": len(self.tensor),
                "shape": "ragged",
                "dtype": dtype_str,
                "has_labels": self.labels is not None,
            }
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


def _resolve_and_parse_labels(
    cfg: Any,
    *,
    task_kind: TaskKind,
    tensor: Any,
    sample_ids: list[str] | None,
) -> Any:
    """Resolve cfg.data.labels to a parser, gate supported_tasks, call parse.

    Returns None when cfg.data.labels is not set.
    """
    from raitap.data.label_parsers.base import LabelParser
    from raitap.data.label_parsers.factory import create_label_parser

    labels_cfg = _get_optional_config_value(cfg.data, "labels")
    if labels_cfg is None:
        return None

    parser = create_label_parser(labels_cfg)
    if not isinstance(parser, LabelParser):
        raise ValueError(
            f"data.labels._target_ resolved to {type(parser).__name__}, which does not "
            "implement the LabelParser protocol (needs 'supported_tasks' and 'parse')."
        )

    if task_kind not in parser.supported_tasks:
        supported = ", ".join(sorted(str(t) for t in parser.supported_tasks))
        raise ValueError(
            f"{type(parser).__name__} does not support task_kind={task_kind!r}. "
            f"Supported tasks: {supported}."
        )

    data_source = _get_optional_config_value(cfg.data, "source")
    model = getattr(cfg, "model", None)
    class_names = _get_optional_config_value(model, "class_names")

    return parser.parse(
        task_kind=task_kind,
        tensor=tensor,
        sample_ids=sample_ids,
        data_source=data_source,
        class_names=class_names,
    )


def load_tensor_from_source(
    source: str,
    n_samples: int | None = None,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Load a raw tensor from a named demo sample, URL, or local path.

    This is the same loading logic used by :class:`Data`, but without label handling.
    Useful for loading background data for SHAP explainers and other call-kwarg
    data-source references resolved by ``resolve_call_data_sources``.

    Args:
        source: Named demo sample (e.g. ``"imagenet_samples"``), URL, or local path.
        n_samples: If set, randomly subsample *n_samples* rows from the loaded tensor.
            Useful for keeping background datasets small (e.g. for KernelExplainer).
        per_image_transform: Optional shape-normalising transform (Resize +
            CenterCrop) applied per-image so mixed-size image directories can
            be stacked. Mirrors the data-side preprocessing applied to the
            primary ``Data.tensor`` so explainer/assessor call-data tensors
            match the assessed input shape.

    Returns:
        Raw tensor of shape ``(N, ...)`` where *N* is the number of samples.

    Raises:
        ValueError: If *source* cannot be resolved, does not exist,
            or the file type is not supported.
    """
    if source in SAMPLE_SOURCES:
        tensor, _ = _load_sample(source, per_image_transform=per_image_transform)
    else:
        tensor = _load_tensor_from_path(
            get_source_path(source, kind=SourceKind.DATA),
            per_image_transform=per_image_transform,
        )

    if n_samples is not None and tensor.shape[0] > n_samples:
        indices = torch.randperm(tensor.shape[0])[:n_samples]
        tensor = tensor[indices]

    return tensor


def load_numpy_from_source(
    source: str,
    n_samples: int | None = None,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray[Any, Any]:
    """
    Load data as a NumPy array using the same resolution rules as :func:`load_tensor_from_source`.

    For file-based sources (local paths and URLs), no intermediate torch tensor is allocated.
    Demo sample sources (``SAMPLE_SOURCES``) use ``raitap.data.samples._load_sample`` (torch-based);
    all other paths are torch-free.
    """
    if source in SAMPLE_SOURCES:
        sample_tensor, _ = _load_sample(source, per_image_transform=per_image_transform)
        arr: np.ndarray[Any, Any] = sample_tensor.numpy()
    else:
        arr = _load_numpy_from_path(
            get_source_path(source, kind=SourceKind.DATA),
            per_image_transform=per_image_transform,
        )

    if n_samples is not None and arr.shape[0] > n_samples:
        rng = np.random.default_rng()
        indices = rng.choice(arr.shape[0], size=n_samples, replace=False)
        arr = arr[indices]

    return arr


def _load_numpy_from_path(
    path: Path,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray[Any, Any]:
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
            return _stack_images_numpy(image_files, per_image_transform=per_image_transform)
        if tabular_files:
            return _concat_tabular_numpy(tabular_files)
        raise FileNotFoundError(
            f"No supported files found in {path}.\n"
            f"Supported image formats: {_IMAGE_FORMATS}\n"
            f"Supported tabular formats: {_TABULAR_FORMATS}"
        )

    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return _load_images_numpy(path, per_image_transform=per_image_transform)
    if suffix in _TABULAR_EXTENSIONS:
        return _load_tabular_numpy(path)
    raise ValueError(
        f"Cannot infer data type from extension {suffix!r}.\n"
        f"Supported image formats: {_IMAGE_FORMATS}\n"
        f"Supported tabular formats: {_TABULAR_FORMATS}"
    )


def _load_tensor_from_path(
    path: Path,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Load a tensor from a single file or a directory (no sample IDs returned)."""
    return torch.from_numpy(_load_numpy_from_path(path, per_image_transform=per_image_transform))


def get_source_path(source: str, *, kind: SourceKind = SourceKind.DATA) -> Path:
    """
    Obtain the local path to the specified source.

    Resolution order:

    1. URL (``http://`` / ``https://``) → download to a cache, return file path.
    2. Named demo sample (key in ``SAMPLE_SOURCES``) → download bundle, return
       cache directory for :attr:`SourceKind.DATA` or the bundled labels CSV for
       :attr:`SourceKind.LABELS`. Sample names take precedence over local paths so the
       resolver matches :meth:`Data._load_data`; use ``./<name>`` or an absolute
       path to force the local-path branch when a directory shadows a sample
       key.
    3. Existing local path → returned as-is.

    Args:
        source: URL, sample name, or local path.
        kind: :attr:`SourceKind.DATA` (default) returns sample image directories;
            :attr:`SourceKind.LABELS` returns the sample-bundled ``labels.csv``.

    Returns:
        Local :class:`~pathlib.Path` (file or directory).

    Raises:
        ValueError: If *source* cannot be resolved or *kind* is not a
            :class:`SourceKind` member.
    """
    if not isinstance(kind, SourceKind):
        allowed = ", ".join(repr(m.value) for m in SourceKind)
        raise ValueError(f"Invalid kind {kind!r}; expected one of: {allowed}.")
    if source.startswith(("http://", "https://")):
        filename = source.rstrip("/").split("/")[-1] or "download"
        dest = _CACHE_DIR / "downloads" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            raitap_log.info("Downloading %s...", filename)
            download_file(source, dest)
        return dest

    if source in SAMPLE_SOURCES:
        if kind is SourceKind.LABELS:
            labels_path = resolve_sample_labels_path(source)
            if labels_path is None:
                raise ValueError(
                    f"Sample {source!r} does not ship ground-truth labels.\n"
                    "Set ``data.labels.source`` to a labels file (CSV/TSV/Parquet) instead."
                )
            return labels_path
        # ``SourceKind.DATA`` — materialise the image bundle and return its directory.
        from .samples import _resolve_sample

        cache_dir = _resolve_sample(source)
        assert cache_dir is not None  # guaranteed by SAMPLE_SOURCES membership
        return cache_dir

    path = Path(source)
    if path.exists():
        return path

    demo_samples = ", ".join(f'"{s}"' for s in SAMPLE_SOURCES)
    raise ValueError(
        f"{kind.capitalize()} source {source!r} could not be resolved.\n"
        f"Expected a URL, local path, or sample name. Known samples: {demo_samples}"
    )


def _list_images_recursive(root: Path) -> list[Path]:
    """Discover image files under ``root`` recursively, sorted by relative posix path."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _list_tabular_recursive(root: Path) -> list[Path]:
    """Discover tabular files under ``root`` recursively, sorted by relative posix path."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _TABULAR_EXTENSIONS]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _stack_images_numpy(
    files: list[Path],
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray[Any, Any]:
    """Stack pre-discovered image files into an NCHW float32 array in [0, 1].

    When ``per_image_transform`` is supplied, it runs on each per-image
    ``(C, H, W)`` tensor before stacking — this is how the resolver's shape
    half (Resize + CenterCrop) normalises mixed-size directories so they can
    be batched.
    """
    arrays = []
    for f in files:
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"))  # HWC uint8
        chw = arr.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW float32
        if per_image_transform is not None:
            chw = per_image_transform(torch.from_numpy(chw)).numpy()
        arrays.append(chw)

    try:
        return np.stack(arrays)  # NCHW float32
    except ValueError:
        sizes = {arr.shape for arr in arrays}
        hint = (
            "Set `data.preprocessing: model-bundled` so the model's bundled "
            "Resize/CenterCrop runs per-image before stacking, or pre-resize "
            "your images externally."
        )
        raise ValueError(
            f"Images have inconsistent shapes after preprocessing: {sizes}. {hint}"
        ) from None


def _load_images_numpy(
    path: Path,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray[Any, Any]:
    """Load image files from a directory (or a single file) as NCHW float32 arrays in [0, 1]."""
    if path.is_dir():
        files = _list_images_recursive(path)
        if not files:
            raise FileNotFoundError(f"No image files found in {path}")
    else:
        files = [path]
    return _stack_images_numpy(files, per_image_transform=per_image_transform)


def _load_images(
    path: Path,
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Load image files from a directory (or a single file) as raw (C, H, W) tensors."""
    return torch.from_numpy(_load_images_numpy(path, per_image_transform=per_image_transform))


def _load_images_ragged(
    files: list[Path],
    *,
    per_image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Load pre-discovered image files as a ragged list of per-image ``(C, H, W)`` float32 tensors.

    Unlike :func:`_stack_images_numpy`, this function does NOT stack the images,
    so files with different spatial dimensions are handled without error. Each
    returned tensor is a ``(C, H, W)`` float32 in ``[0, 1]``.  When
    ``per_image_transform`` is supplied it is applied per image (as in
    :func:`_stack_images_numpy`) before the tensor is appended, but no resize
    is forced — the transform is passed through transparently.

    Used for detection task inputs where batches are ragged by design.
    """
    result: list[torch.Tensor] = []
    for f in files:
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"))  # HWC uint8
        chw = arr.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW float32
        t: torch.Tensor = torch.from_numpy(chw)
        if per_image_transform is not None:
            t = per_image_transform(t)
        result.append(t)
    return result


def _apply_data_module_to_batch(
    module: Any | None,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply ``data_module`` to an already-stacked tabular tensor.

    Images go through ``per_image_transform`` while loading because mixed
    sizes can only be stacked after a Resize step. Tabular rows are always
    uniform, so the module is applied once on the whole ``(N, F)`` batch.
    """
    if module is None:
        return tensor
    module.eval()
    with torch.no_grad():
        return module(tensor)


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
            raitap_log.warn(
                f"Configured data.labels.id_column {configured_column!r} not found in labels file.",
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
    if encoding:
        try:
            LabelEncoding(encoding)
        except ValueError as exc:
            allowed = ", ".join(repr(member.value) for member in LabelEncoding)
            raise ValueError(
                f"Unsupported data.labels.encoding {labels_encoding!r}. Use {allowed}."
            ) from exc

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

    - ``strategy="stem"``: flat-dir / basename matching. Strip the directory
      and the extension; e.g. ``"NORMAL/IM-0001.jpeg"`` → ``"IM-0001"``.
    - ``strategy="relative_path"``: keep the directory; strip only the
      extension. e.g. ``"NORMAL\\IM-0001.jpeg"`` → ``"NORMAL/IM-0001"``.
    """
    text = str(value).strip().replace("\\", "/")
    p = Path(text)
    if strategy == "relative_path":
        return p.with_suffix("").as_posix()
    return p.stem


def _resolve_id_strategy(strategy: str, raw_label_ids: pd.Series) -> str:
    try:
        resolved = IdStrategy(strategy)
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in IdStrategy)
        raise ValueError(
            f"Unsupported data.labels.id_strategy {strategy!r}. Use {allowed}."
        ) from exc
    if resolved is IdStrategy.auto:
        for raw in raw_label_ids.tolist():
            text = str(raw)
            if "/" in text or "\\" in text:
                return IdStrategy.relative_path.value
        return IdStrategy.stem.value
    return resolved.value


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
    raw_label_ids_list = raw_label_ids.tolist()
    normalised_sample_ids = [_normalise_sample_id(sid, strategy) for sid in sample_ids]
    normalised_label_ids = [_normalise_sample_id(raw_id, strategy) for raw_id in raw_label_ids_list]
    duplicates = sorted(
        [row_id for row_id, count in Counter(normalised_label_ids).items() if count > 1]
    )
    if duplicates:
        preview = ", ".join(duplicates[:5])
        more = "..." if len(duplicates) > 5 else ""
        hint = ""
        if strategy == "stem":
            hint = (
                " Hint: stem-only matching collapses ids that share a filename "
                "across subdirs (e.g. NORMAL/IM-0001.jpeg vs PNEUMONIA/IM-0001.jpeg). "
                "Set data.labels.id_strategy=relative_path (or =auto, the default) "
                "and use posix-style relative paths in the id column."
            )
        raise ValueError(
            f"Duplicate label IDs detected ({preview}{more}) under id_strategy={strategy!r}.{hint}"
        )

    label_by_id = {
        row_id: int(label)
        for row_id, label in zip(normalised_label_ids, encoded_labels, strict=False)
    }
    missing_ids = [sid for sid in normalised_sample_ids if sid not in label_by_id]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        more = "..." if len(missing_ids) > 5 else ""
        # Strategy hints only fire under ``relative_path`` — stem mode strips
        # directory components from both sides symmetrically, so a missing
        # match means basenames don't line up (genuine data/label gap), not
        # a strategy mismatch. Inspect the *raw* (pre-normalisation) inputs
        # since normalised ids no longer carry separators.
        samples_have_separators = any("/" in s or "\\" in s for s in sample_ids)
        labels_have_separators = any("/" in str(r) or "\\" in str(r) for r in raw_label_ids_list)
        hint = ""
        if strategy == "relative_path" and samples_have_separators and not labels_have_separators:
            hint = (
                " Hint: data.source has a nested layout (sample ids contain "
                "path separators) but label ids don't — under "
                "id_strategy='relative_path' both sides must use the same "
                "relative-path form (e.g. 'NORMAL/IM-0001.jpeg'). Either add "
                "the directory prefix to your label ids, or use "
                "id_strategy=stem to match by basename only."
            )
        elif strategy == "relative_path" and labels_have_separators and not samples_have_separators:
            hint = (
                " Hint: label ids contain path separators but data.source is "
                "flat (sample ids don't). Drop the directory prefix from the "
                "label ids, or use id_strategy=stem to match by basename only."
            )
        raise ValueError(
            f"Missing labels for some sample IDs ({preview}{more}) under "
            f"id_strategy={strategy!r}.{hint}"
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
