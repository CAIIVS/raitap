"""torch<->numpy + explain_func bridge — the ONLY coupling to explainer internals (#341)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from raitap.transparency.contracts import ExplanationOutputSpace, TensorLayout

if TYPE_CHECKING:
    from collections.abc import Callable

    from raitap.transparency.explainers.base_explainer import AttributionOnlyExplainer
    from raitap.transparency.results import ExplanationResult


@dataclass(frozen=True)
class QuantusArrays:
    x_batch: np.ndarray
    y_batch: np.ndarray
    a_batch: np.ndarray


def derive_channel_first(result: ExplanationResult) -> bool:
    """Whether ``result``'s attributions are laid out channel-first (NCHW).

    True for CAM-style spatial maps (``IMAGE_SPATIAL_MAP``) as well as any
    other output space whose layout is ``NCHW`` — e.g. plain gradient
    attributions (Saliency, IntegratedGradients) on image inputs, which keep
    the input's channel-first layout (``INPUT_FEATURES`` space) rather than
    collapsing to a spatial map.
    """
    output_space = result.semantics.output_space
    if output_space.space is ExplanationOutputSpace.IMAGE_SPATIAL_MAP:
        return True
    return output_space.layout is TensorLayout.BATCH_CHANNEL_HEIGHT_WIDTH


def resolve_target(result: ExplanationResult) -> int | list[int] | None:
    """Resolve the classification target from ``call_kwargs``, falling back to semantics."""
    tgt: Any = result.call_kwargs.get("target")
    if tgt is None and result.semantics.target is not None:
        tgt = result.semantics.target.target
    if isinstance(tgt, torch.Tensor):
        return tgt.detach().cpu().tolist()
    if tgt is None or isinstance(tgt, int):
        return tgt
    if isinstance(tgt, str):
        raise TypeError(f"str target {tgt!r} is not supported by the Quantus bridge")
    return list(tgt)


def _y_batch(target: int | list[int] | None, batch: int) -> np.ndarray:
    if target is None:
        return np.zeros(batch, dtype=np.int64)
    if isinstance(target, int):
        return np.full(batch, target, dtype=np.int64)
    return np.asarray(list(target), dtype=np.int64)


def to_quantus_arrays(
    result: ExplanationResult, *, target: int | list[int] | None
) -> QuantusArrays:
    """Convert an ``ExplanationResult`` to the ``(x_batch, y_batch, a_batch)`` numpy triple
    Quantus metrics expect. ``result.inputs``/``.attributions`` are already CPU-detached
    (``ExplanationResult.__post_init__``); this only casts dtype and moves to numpy.
    """
    x = result.inputs.float().numpy()
    a = result.attributions.float().numpy()
    return QuantusArrays(x_batch=x, y_batch=_y_batch(target, x.shape[0]), a_batch=a)


def explainer_to_explain_func(
    explainer: AttributionOnlyExplainer, device: torch.device
) -> Callable[..., np.ndarray]:
    """Wrap an attribution-only explainer as a Quantus ``explain_func``.

    Uses ``compute_attributions`` (raw tensor, no artifact write) rather than
    ``explain()`` (which writes artifacts and is not meant to run once per
    perturbation). Quantus calls the returned callable with ``model`` (the
    possibly-perturbed ``nn.Module``), ``inputs`` and ``targets`` as numpy
    arrays; this tensorises them, re-explains, and returns numpy.
    """

    def explain_func(model: Any, inputs: Any, targets: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        t = torch.as_tensor(np.asarray(inputs), device=device, dtype=torch.float32)
        tgt = torch.as_tensor(np.asarray(targets), device=device)
        with torch.enable_grad():
            attr = explainer.compute_attributions(model, t, target=tgt)
        return attr.detach().cpu().float().numpy()

    return explain_func
