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
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _is_image_modality(input_spec: object) -> bool:
    kind = getattr(input_spec, "kind", None)
    if kind is not None:
        return str(kind).lower() == "image"
    layout = getattr(input_spec, "layout", None)
    return str(layout or "").upper().replace(" ", "") == "NCHW"


def _render_baseline_image(baseline: torch.Tensor, run_dir: Path) -> Path:
    """Render the first sample of *baseline* to ``run_dir/baseline.png``.

    Mirrors the normalisation/layout of ``InputThumbnailVisualiser`` (min-max
    normalise, NCHW->HWC, grayscale handling) without importing its private
    helpers. Returns the path relative to ``run_dir`` for storage in the record.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sample = baseline[0] if baseline.ndim >= 4 else baseline
    image = np.asarray(sample.detach().cpu().numpy())
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    lo, hi = float(image.min()), float(image.max())
    if hi > lo:
        image = (image - lo) / (hi - lo)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    try:
        if image.ndim == 3 and image.shape[-1] == 1:
            ax.imshow(image[..., 0], cmap="gray")
        else:
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.axis("off")
        fig.tight_layout()
        run_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(run_dir / _BASELINE_IMAGE_NAME, bbox_inches="tight", dpi=150)
    finally:
        plt.close(fig)
    return Path(_BASELINE_IMAGE_NAME)
