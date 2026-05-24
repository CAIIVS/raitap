"""Per-sample signed-perturbation heatmap for empirical attacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap.robustness.visualisers.registration import robustness_visualiser
from raitap.utils.lazy import lazy_import

if TYPE_CHECKING:
    import torch
else:
    torch = lazy_import("torch")

from ...contracts import AssessmentKind
from ..base_visualiser import BaseRobustnessVisualiser
from .image_pair_visualiser import _require_image_modality, _signed_perturbation_heatmap

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


_SUPPORTED_MODES = frozenset({"signed_dominant", "mean_abs", "mean", "max_abs"})


@robustness_visualiser(
    registry_name="perturbation_heatmap",
    supported_assessment_kinds=frozenset({AssessmentKind.EMPIRICAL_ATTACK}),
    embeds_perturbation_map=True,
)
class PerturbationHeatmapVisualiser(BaseRobustnessVisualiser):
    """Render the perturbation tensor as a heatmap.

    Default ``aggregate_channels="signed_dominant"`` keeps the signed value of
    the channel with the largest absolute deviation per pixel — preserves sign
    *and* avoids the cancellation that happens when ``mean`` averages opposing
    signs across channels (e.g. ``+eps`` on R and ``-eps`` on G displaying as
    ~0). Other modes (``mean``, ``mean_abs``, ``max_abs``) are kept as opt-ins.
    """

    def __init__(
        self,
        *,
        max_samples: int = 4,
        cmap: str = "seismic",
        aggregate_channels: str = "signed_dominant",
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        self.cmap = cmap
        if aggregate_channels not in _SUPPORTED_MODES:
            raise ValueError(
                f"aggregate_channels must be one of {sorted(_SUPPORTED_MODES)}, "
                f"got {aggregate_channels!r}."
            )
        self.aggregate_channels = aggregate_channels

    def visualise(
        self,
        result: RobustnessResult,
        *,
        context: RobustnessVisualisationContext,
        **kwargs: Any,
    ) -> Figure:
        kwargs.pop("include_clean_input", None)
        include_perturbation_map = bool(kwargs.pop("include_perturbation_map", True))
        del kwargs
        if not include_perturbation_map:
            raise ValueError(
                "PerturbationHeatmapVisualiser requires include_perturbation_map=True."
            )
        if result.perturbed_inputs is None:
            raise ValueError(
                "PerturbationHeatmapVisualiser requires perturbed_inputs on the result."
            )
        _require_image_modality(result, type(self).__name__)

        clean = _to_image_batch(result.clean_inputs)
        perturbed = _to_image_batch(result.perturbed_inputs)
        delta = (perturbed - clean).detach().cpu().to(torch.float32)
        n = min(int(delta.shape[0]), self.max_samples)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)

        sample_names = context.sample_names or []
        global_extreme = float(delta[:n].abs().max().item())
        global_extreme = max(global_extreme, 1e-6)

        for col in range(n):
            heatmap = _aggregate(delta[col], self.aggregate_channels)
            axes[0][col].imshow(heatmap, cmap=self.cmap, vmin=-global_extreme, vmax=global_extreme)
            axes[0][col].set_axis_off()
            sample_title = (
                sample_names[col] if context.show_sample_names and col < len(sample_names) else ""
            )
            axes[0][col].set_title(sample_title or f"sample {col}")
        fig.suptitle(f"{context.algorithm} — perturbation heatmap", fontsize=12)
        fig.tight_layout()
        return fig


def _to_image_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _aggregate(delta: torch.Tensor, mode: str) -> np.ndarray:
    if mode == "signed_dominant":
        return _signed_perturbation_heatmap(delta)
    array = delta.numpy()
    if array.ndim == 2:
        return array
    # (C, H, W)
    if mode == "mean_abs":
        return np.abs(array).mean(axis=0)
    if mode == "mean":
        return array.mean(axis=0)
    return np.abs(array).max(axis=0)
