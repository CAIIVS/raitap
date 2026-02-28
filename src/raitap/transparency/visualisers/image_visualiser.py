"""Image modality visualization (heatmaps, overlays)"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import BaseVisualiser


class ImageHeatmapvisualiser(BaseVisualiser):
    """
    Visualize attributions for image inputs as heatmaps.

    Works with any attribution method (Captum, SHAP, etc.)
    """

    def visualise(
        self, attributions, inputs=None, cmap="jet", alpha=0.4, max_samples=8, **kwargs
    ) -> Figure:
        """
        Create heatmap visualization.

        Args:
            attributions: (B, C, H, W) or (B, H, W) tensor/array
            inputs: Original images (B, C, H, W) for overlay
            cmap: Matplotlib colormap (default: "jet" for better contrast)
            alpha: Transparency of heatmap overlay (0-1, default: 0.4)
            max_samples: Maximum samples to display (default: 8)

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if hasattr(attributions, "detach"):
            attributions = attributions.detach().cpu().numpy()
        elif hasattr(attributions, "numpy"):  # torch.Tensor without grad
            attributions = attributions.cpu().numpy()

        # Aggregate across channels if needed
        if attributions.ndim == 4:  # (B, C, H, W)
            # Take absolute value and then mean for better visualization
            attributions = np.mean(np.abs(attributions), axis=1)  # (B, H, W)

        # Normalize each sample independently for better contrast
        normalized_attrs = []
        for attr in attributions:
            # Normalize to [0, 1] range
            attr_min, attr_max = attr.min(), attr.max()
            attr_norm = (attr - attr_min) / (attr_max - attr_min) if attr_max > attr_min else attr
            normalized_attrs.append(attr_norm)
        attributions = np.array(normalized_attrs)

        # Limit display to max_samples
        n_display = min(attributions.shape[0], max_samples)
        attributions = attributions[:n_display]

        if inputs is not None:
            if hasattr(inputs, "detach"):
                inputs = inputs.detach().cpu().numpy()
            elif hasattr(inputs, "numpy"):
                inputs = inputs.cpu().numpy()
            inputs = inputs[:n_display]

        # Create subplots
        fig, axes_result = plt.subplots(1, n_display, figsize=(4 * n_display, 4))
        # Ensure axes is always iterable for consistent iteration
        axes_list = [axes_result] if n_display == 1 else axes_result.tolist()  # type: ignore[union-attr]

        for idx, (ax, attr) in enumerate(zip(axes_list, attributions, strict=False)):
            # Show original image first (fully opaque background)
            if inputs is not None:
                img = inputs[idx]
                if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    img = img.transpose(1, 2, 0)
                # Normalize to [0, 1]
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                ax.imshow(img)

                # Get image dimensions for extent
                h, w = img.shape[:2]
            else:
                # If no input image, use attribution dimensions
                h, w = attr.shape

            # Overlay heatmap with extent to match image size
            # extent=(left, right, bottom, top) in data coordinates
            im = ax.imshow(
                attr,
                cmap=cmap,
                alpha=alpha,
                interpolation="bilinear",
                extent=(0, w, h, 0),  # Stretch heatmap to match image dimensions
            )
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.tight_layout()
        return fig
