"""Layer-space activation attribution visualiser (#267)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import (
    ExplanationOutputSpace,
    ExplanationScope,
)
from raitap.transparency.visualisers.registration import transparency_visualiser

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


@transparency_visualiser(
    registry_name="layer_activation",
    supported_scopes={ExplanationScope.LOCAL},
    supported_output_spaces={ExplanationOutputSpace.LAYER_ACTIVATION},
)
class LayerActivationVisualiser(BaseVisualiser):
    """Render layer-space attributions (captum ``Layer*`` methods).

    Layer attributions are aligned to a hidden layer, not the input grid, so this
    renders a per-channel / per-feature magnitude summary rather than an
    input-space heatmap. One row per sample, capped at ``max_samples``.

    Per-sample tensor rank drives the rendering:

    * ``>= 3``: conv ``(C, H, W, ...)``: per-channel mean ``|attribution|`` bars.
    * ``2``: sequence ``(tokens, hidden)`` (e.g. a ViT block): a magnitude
      heatmap, since neither axis is a single "unit".
    * ``1``: linear ``(F,)``: per-feature magnitude bars.
    """

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        title: str | None = None,
        max_samples: int = 8,
        **kwargs: Any,
    ) -> Figure:
        del inputs, context, kwargs

        attrs = _to_numpy(attributions)
        if attrs.ndim < 2:
            attrs = attrs.reshape(1, -1)
        batch = min(attrs.shape[0], max_samples)

        fig, axes = plt.subplots(batch, 1, figsize=(10, 2.4 * batch), squeeze=False)
        for row in range(batch):
            ax = axes[row][0]
            sample = np.abs(attrs[row])
            if sample.ndim >= 3:
                # Conv layer (C, H, W, ...): per-channel mean magnitude.
                magnitude = sample.reshape(sample.shape[0], -1).mean(axis=1)
                ax.bar(np.arange(len(magnitude)), magnitude)
                ax.set_xlabel("Channel")
                ax.set_ylabel("Mean |attribution|")
                ax.grid(axis="y", alpha=0.3)
            elif sample.ndim == 2:
                # Sequence layer (tokens, hidden): 2-D magnitude heatmap.
                image = ax.imshow(sample, aspect="auto", cmap="viridis")
                ax.set_xlabel("Hidden unit")
                ax.set_ylabel("Token")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Linear layer (F,): per-feature magnitude.
                ax.bar(np.arange(len(sample)), sample)
                ax.set_xlabel("Feature")
                ax.set_ylabel("|attribution|")
                ax.grid(axis="y", alpha=0.3)
            ax.set_title(f"Sample {row} - layer-space (not input-aligned)")

        fig.suptitle(title or "Layer activation attribution")
        fig.tight_layout()
        return fig


def _to_numpy(attributions: object) -> np.ndarray:
    if hasattr(attributions, "detach"):
        return attributions.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(attributions)
