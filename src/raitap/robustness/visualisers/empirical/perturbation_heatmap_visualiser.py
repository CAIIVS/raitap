"""Per-sample signed-perturbation heatmap for empirical attacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch

from ...contracts import MethodKind
from ..base_visualiser import BaseRobustnessVisualiser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...contracts import RobustnessVisualisationContext
    from ...results import RobustnessResult


class PerturbationHeatmapVisualiser(BaseRobustnessVisualiser):
    """Render the signed perturbation tensor as a diverging heatmap."""

    supported_method_kinds: ClassVar[frozenset[MethodKind]] = frozenset(
        {MethodKind.EMPIRICAL_ATTACK}
    )

    def __init__(
        self,
        *,
        max_samples: int = 4,
        cmap: str = "seismic",
        aggregate_channels: str = "mean_abs",
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        self.cmap = cmap
        if aggregate_channels not in {"mean_abs", "mean", "max_abs"}:
            raise ValueError(
                "aggregate_channels must be one of {'mean_abs', 'mean', 'max_abs'}, "
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
        del kwargs
        if result.perturbed_inputs is None:
            raise ValueError(
                "PerturbationHeatmapVisualiser requires perturbed_inputs on the result."
            )

        clean = _to_image_batch(result.clean_inputs)
        perturbed = _to_image_batch(result.perturbed_inputs)
        delta = (perturbed - clean).detach().cpu().to(torch.float32)
        n = min(int(delta.shape[0]), self.max_samples)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)

        sample_names = context.sample_names or []
        global_extreme = float(delta[:n].abs().max().item())
        global_extreme = max(global_extreme, 1e-6)

        for col in range(n):
            heatmap = _aggregate(delta[col].numpy(), self.aggregate_channels)
            axes[0][col].imshow(
                heatmap, cmap=self.cmap, vmin=-global_extreme, vmax=global_extreme
            )
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


def _aggregate(array: np.ndarray, mode: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    # array: (C, H, W)
    if mode == "mean_abs":
        return np.abs(array).mean(axis=0)
    if mode == "mean":
        return array.mean(axis=0)
    return np.abs(array).max(axis=0)
