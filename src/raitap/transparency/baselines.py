"""Capture the reference input (baseline) an attribution method used.

Single helper invoked once at the explain chokepoint
(:meth:`AttributionOnlyExplainer.explain`). Resolves the baseline ``mode``,
hashes the exact tensor used as baseline, and — for image modality — renders a
first-sample preview. See issue #210 and the design doc
``docs/superpowers/specs/2026-05-26-baseline-documentation-design.md``.

The render helpers are inlined (not imported from ``input_thumbnail``) to avoid
cross-module coupling on another module's private functions; ``matplotlib`` /
``numpy`` are imported lazily inside the render function to keep importing
``build_baseline_record`` light.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raitap.utils.lazy import lazy_import

from .contracts import BaselineRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
else:
    torch = lazy_import("torch")

_BASELINE_IMAGE_NAME = "baseline.png"
# A multi-image baseline (e.g. a SHAP background set) is rendered as a grid
# preview: up to ``_BASELINE_GRID_CAP`` tiles, ``_BASELINE_GRID_COLS`` per row.
# The full count is always recorded in the descriptor's ``n_samples``/``shape``.
_BASELINE_GRID_CAP = 25
_BASELINE_GRID_COLS = 5


def build_baseline_record(
    *,
    explainer: object,
    inputs: torch.Tensor,
    call_kwargs: Mapping[str, Any],
    call_provenance: Mapping[str, Mapping[str, Any]] | None,
    input_spec: object,
    run_dir: Path,
) -> BaselineRecord | None:
    """Return a :class:`BaselineRecord` for *explainer*, or ``None``.

    ``None`` when the explainer family takes no baseline, or the kwarg is absent
    and the algorithm has no meaningful implicit default (e.g. Saliency,
    TreeExplainer-without-background).
    """
    kwarg_name = getattr(explainer, "baseline_kwarg", None)
    if kwarg_name is None:
        return None

    algorithm = str(getattr(explainer, "algorithm", ""))
    value = call_kwargs.get(kwarg_name)

    source: str | None = None
    n_samples: int | None = None

    if isinstance(value, torch.Tensor):
        provenance = (call_provenance or {}).get(kwarg_name)
        if provenance is not None:
            mode = "configured"
            raw_source = provenance.get("source")
            source = None if raw_source is None else str(raw_source)
            raw_n = provenance.get("n_samples")
            n_samples = None if raw_n is None else int(raw_n)
        else:
            mode = "user_tensor"
        baseline_tensor = value
    else:
        defaults = getattr(explainer, "baseline_defaults", {})
        mode = defaults.get(algorithm)
        if mode is None:
            return None
        if mode == "zero":
            baseline_tensor = torch.zeros_like(inputs[:1])
        elif mode == "input_batch":
            baseline_tensor = inputs
        else:  # defensive: unknown declared default
            return None

    sha256 = _hash_tensor(baseline_tensor)
    image_path: Path | None = None
    if _is_image_modality(input_spec):
        image_path = _render_baseline_image(baseline_tensor, run_dir)

    return BaselineRecord(
        kwarg_name=str(kwarg_name),
        mode=str(mode),
        source=source,
        n_samples=n_samples,
        shape=tuple(int(dim) for dim in baseline_tensor.shape),
        dtype=str(baseline_tensor.dtype),
        sha256=sha256,
        image_path=image_path,
    )


def _hash_tensor(tensor: torch.Tensor) -> str:
    # Cast to float32 first: NumPy has no bfloat16, so a bf16 baseline would
    # otherwise raise in ``.numpy()``. The cast is a no-op for float32 inputs and
    # keeps the hash deterministic across dtypes.
    data = tensor.detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _is_image_modality(input_spec: object) -> bool:
    kind = getattr(input_spec, "kind", None)
    if kind is not None:
        return str(kind).lower() == "image"
    layout = getattr(input_spec, "layout", None)
    return str(layout or "").upper().replace(" ", "") == "NCHW"


def _montage_caption(shown: int, total: int) -> str | None:
    """Label for a capped baseline montage, or ``None`` when all tiles are shown."""
    return f"Showing {shown} of {total}" if shown < total else None


def _render_baseline_image(baseline: torch.Tensor, run_dir: Path) -> Path:
    """Render a preview of *baseline* to ``run_dir/baseline.png``.

    A single-image baseline renders one tile; a multi-image baseline (e.g. a
    SHAP background set) renders a capped grid of its images (see
    ``_BASELINE_GRID_CAP``). Each tile mirrors ``InputThumbnailVisualiser``'s
    normalisation/layout (min-max normalise, NCHW->HWC, grayscale handling)
    without importing its private helpers. Returns the path relative to
    ``run_dir`` for storage in the record.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    total = int(baseline.shape[0]) if baseline.ndim >= 4 else 1
    if baseline.ndim >= 4:
        tiles = [baseline[index] for index in range(min(total, _BASELINE_GRID_CAP))]
    else:
        tiles = [baseline]

    count = len(tiles)
    cols = min(_BASELINE_GRID_COLS, count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), squeeze=False)
    try:
        flat_axes = list(axes.flat)
        for ax, tile in zip(flat_axes, tiles, strict=False):
            # float32 cast: NumPy has no bfloat16, so a bf16 tile would raise here.
            image = np.asarray(tile.detach().cpu().to(torch.float32).numpy())
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            lo, hi = float(image.min()), float(image.max())
            if hi > lo:
                image = (image - lo) / (hi - lo)
            if image.ndim == 3 and image.shape[-1] == 1:
                ax.imshow(image[..., 0], cmap="gray")
            else:
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.axis("off")
        for ax in flat_axes[count:]:
            ax.axis("off")
        caption = _montage_caption(count, total)
        if caption is not None:
            fig.suptitle(caption, fontsize=10)
        fig.tight_layout()
        run_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / _BASELINE_IMAGE_NAME, bbox_inches="tight", dpi=150)
    finally:
        plt.close(fig)
    return Path(_BASELINE_IMAGE_NAME)
