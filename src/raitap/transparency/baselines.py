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
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raitap.utils.lazy import lazy_import

from .contracts import BaselineCardinality, BaselineMode, BaselineRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
else:
    torch = lazy_import("torch")

_BASELINE_CONFIG_KEY = "baseline"
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
    render_cache: dict[str, Path] | None = None,
) -> BaselineRecord | None:
    """Return a :class:`BaselineRecord` for *explainer*, or ``None``.

    ``None`` when the explainer family takes no baseline, or the kwarg is absent
    and the algorithm has no meaningful implicit default (e.g. Saliency,
    TreeExplainer-without-background).

    ``render_cache`` (keyed by the baseline ``sha256``) lets callers that invoke
    this helper repeatedly for the *same* baseline — e.g. the detection K-loop,
    one ``explain`` per box — render the preview image once and copy it for the
    rest, instead of re-running matplotlib per box.
    """
    kwarg_name = getattr(explainer, "baseline_kwarg_name", None)
    if kwarg_name is None:
        return None

    algorithm = str(getattr(explainer, "algorithm", ""))
    value = call_kwargs.get(kwarg_name)

    source: str | None = None
    n_samples: int | None = None

    if isinstance(value, torch.Tensor):
        provenance = (call_provenance or {}).get(kwarg_name)
        if provenance is not None:
            mode: BaselineMode | None = BaselineMode.CONFIGURED
            raw_source = provenance.get("source")
            source = None if raw_source is None else str(raw_source)
            raw_n = provenance.get("n_samples")
            n_samples = None if raw_n is None else int(raw_n)
        else:
            mode = BaselineMode.USER_TENSOR
        baseline_tensor = value
    else:
        # The implicit default mode lives on the algorithm's
        # ``ExplainerSemanticsHints.baseline_default`` in ``algorithm_registry``;
        # read defensively so stubs without a registry simply yield no baseline.
        # ``baseline_default`` is a ``BaselineMode`` member; comparison below also
        # accepts the bare-string equivalent (StrEnum) for forward compat.
        registry = getattr(explainer, "algorithm_registry", None) or {}
        hints = registry.get(algorithm)
        mode = getattr(hints, "baseline_default", None)
        if mode is None:
            return None
        if mode == BaselineMode.ZERO:
            baseline_tensor = torch.zeros_like(inputs[:1])
        elif mode == BaselineMode.INPUT_BATCH:
            baseline_tensor = inputs
        else:  # defensive: unknown declared default
            return None

    sha256 = _hash_tensor(baseline_tensor)
    image_path: Path | None = None
    if _is_image_modality(input_spec):
        image_path = _resolve_baseline_image(baseline_tensor, run_dir, sha256, render_cache)

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


def apply_config_baseline(
    *,
    explainer: object,
    call_kwargs: dict[str, Any],
    raitap_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Route the library-agnostic ``raitap.baseline`` field to the adapter's call kwarg.

    Users set the baseline by one stable name — ``raitap.baseline`` — regardless of
    the underlying library's own kwarg (Captum ``baselines``, SHAP ``background_data``);
    the adapter's ``baseline_kwarg_name`` ClassVar names the destination. The value is the
    same ``{source, n_samples}`` data-source descriptor (or a tensor via the Python API)
    accepted under ``call:``, so the existing :func:`resolve_call_data_sources` /
    :func:`build_baseline_record` path resolves and documents it unchanged.

    Call this on the *final* call dict (config ``call:`` merged with any runtime
    Python-API kwargs) so the precedence rule is uniform regardless of source.
    Mutates and returns ``call_kwargs`` with the baseline injected; pops ``baseline``
    from ``raitap_kwargs`` so it is not mistaken for a runtime option. Precedence: when
    the baseline is *also* set via the adapter's own kwarg (in ``call:`` or passed at
    runtime), ``raitap.baseline`` wins and a warning fires — it is never overridden
    silently. Raises :class:`RaitapError` when ``raitap.baseline`` is set on an
    explainer whose family takes no baseline.
    """
    baseline = raitap_kwargs.pop(_BASELINE_CONFIG_KEY, None)
    if baseline is None:
        return call_kwargs

    kwarg_name = getattr(explainer, "baseline_kwarg_name", None)
    if kwarg_name is None:
        from raitap.utils.errors import RaitapError

        algorithm = str(getattr(explainer, "algorithm", "")) or "<unknown>"
        raise RaitapError(
            f"raitap.baseline was set, but explainer {type(explainer).__name__} "
            f"(algorithm {algorithm!r}) takes no baseline. Remove raitap.baseline."
        )

    if kwarg_name in call_kwargs:
        from raitap import raitap_log

        raitap_log.warn(
            "Baseline set both via raitap.baseline and the %r kwarg (call: block or "
            "runtime); using raitap.baseline and ignoring the %r kwarg.",
            kwarg_name,
            kwarg_name,
        )

    _warn_on_baseline_cardinality_mismatch(explainer=explainer, baseline=baseline)

    call_kwargs[kwarg_name] = baseline
    return call_kwargs


def _warn_on_baseline_cardinality_mismatch(*, explainer: object, baseline: Any) -> None:
    """Flag (never reshape) a baseline whose sample count contradicts the method.

    A ``SINGLE``-reference method (e.g. Integrated Gradients) given a multi-sample
    data-source baseline — typically a SHAP background set reused by mistake — will
    fail unless it happens to match the input batch. We can't know the batch here,
    so we warn rather than block the rare valid per-sample case. ``SET`` methods and
    direct tensors (advanced Python-API use) are left alone.
    """
    if not isinstance(baseline, dict):
        return
    n_samples = baseline.get("n_samples")
    if not isinstance(n_samples, int) or n_samples <= 1:
        return

    algorithm = str(getattr(explainer, "algorithm", ""))
    registry = getattr(explainer, "algorithm_registry", None) or {}
    hints = registry.get(algorithm)
    if getattr(hints, "baseline_cardinality", None) is not BaselineCardinality.SINGLE:
        return

    from raitap import raitap_log

    raitap_log.warn(
        "%s takes a single baseline reference, but raitap.baseline sets "
        "n_samples=%d (a sample-set baseline). It will fail unless it matches the "
        "input batch; use n_samples=1 for a broadcast baseline.",
        algorithm or type(explainer).__name__,
        n_samples,
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


def _resolve_baseline_image(
    baseline: torch.Tensor,
    run_dir: Path,
    sha256: str,
    render_cache: dict[str, Path] | None,
) -> Path:
    """Render the baseline preview, or copy a cached render of the same content.

    The expensive matplotlib render runs once per distinct ``sha256``; repeat
    callers (detection's per-box K-loop) get a cheap file copy into their own
    ``run_dir`` so every artefact stays self-contained.
    """
    if render_cache is not None:
        cached = render_cache.get(sha256)
        if cached is not None and cached.exists():
            return _copy_baseline_image(cached, run_dir)

    image_path = _render_baseline_image(baseline, run_dir)
    if render_cache is not None:
        render_cache[sha256] = run_dir / image_path
    return image_path


def _copy_baseline_image(source: Path, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, run_dir / _BASELINE_IMAGE_NAME)
    return Path(_BASELINE_IMAGE_NAME)


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
