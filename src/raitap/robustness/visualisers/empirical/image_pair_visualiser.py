"""Side-by-side clean / perturbed / diff renderer for empirical attacks."""

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


class ImagePairVisualiser(BaseRobustnessVisualiser):
    """Render N rows by 3 columns: clean, perturbed, signed perturbation."""

    supported_method_kinds: ClassVar[frozenset[MethodKind]] = frozenset(
        {MethodKind.EMPIRICAL_ATTACK}
    )

    def __init__(
        self,
        *,
        max_samples: int = 4,
        cmap: str = "RdBu_r",
        diff_scale: float | None = None,
    ) -> None:
        self.max_samples = max(int(max_samples), 1)
        self.cmap = cmap
        self.diff_scale = diff_scale

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
                "ImagePairVisualiser requires perturbed_inputs to be present on the result."
            )
        clean = _to_image_batch(result.clean_inputs)
        perturbed = _to_image_batch(result.perturbed_inputs)
        n = min(int(clean.shape[0]), self.max_samples)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

        scale = self.diff_scale
        diff_extreme = float(
            (perturbed[:n] - clean[:n]).abs().max().item() if scale is None else scale
        )
        diff_extreme = max(diff_extreme, 1e-6)

        sample_names = context.sample_names or []

        for row in range(n):
            clean_image = _as_displayable(clean[row])
            perturbed_image = _as_displayable(perturbed[row])
            diff = perturbed_image - clean_image

            axes[row][0].imshow(clean_image, cmap="gray" if clean_image.ndim == 2 else None)
            axes[row][0].set_axis_off()
            axes[row][1].imshow(perturbed_image, cmap="gray" if perturbed_image.ndim == 2 else None)
            axes[row][1].set_axis_off()
            axes[row][2].imshow(diff, cmap=self.cmap, vmin=-diff_extreme, vmax=diff_extreme)
            axes[row][2].set_axis_off()

            target_label = int(result.targets[row].item())
            clean_pred = int(result.clean_predictions[row].item())
            adv_pred = (
                int(result.perturbed_predictions[row].item())
                if result.perturbed_predictions is not None
                else -1
            )
            sample_title = (
                sample_names[row] if context.show_sample_names and row < len(sample_names) else ""
            )

            axes[row][0].set_title(_format_title("clean", clean_pred, target_label, sample_title))
            axes[row][1].set_title(_format_title("perturbed", adv_pred, target_label, sample_title))
            axes[row][2].set_title("perturbation")

        fig.suptitle(f"{context.algorithm} — clean vs perturbed", fontsize=12)
        fig.tight_layout()
        return fig


def _to_image_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _as_displayable(sample: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) or (H, W) tensor to a HWC / HW float32 array in [0, 1]."""
    array = sample.detach().cpu().to(torch.float32).numpy()
    if array.ndim == 3:
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] == 1:
            array = array[..., 0]
    array = np.clip(array, 0.0, 1.0) if array.max() <= 1.0 + 1e-3 else _normalise01(array)
    return array


def _normalise01(array: np.ndarray) -> np.ndarray:
    lo, hi = float(array.min()), float(array.max())
    if hi - lo < 1e-6:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)


def _format_title(label: str, predicted: int, target: int, sample_id: str) -> str:
    parts = [label, f"pred={predicted}", f"target={target}"]
    if sample_id:
        parts.insert(0, sample_id)
    return " | ".join(parts)
