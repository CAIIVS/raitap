"""Framework-agnostic input preview visualisers for reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from raitap.transparency.contracts import ExplanationScope

from .base_visualiser import BaseVisualiser

if TYPE_CHECKING:
    import torch
    from matplotlib.figure import Figure

    from raitap.transparency.contracts import VisualisationContext


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    val: Any = x
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val)


def _normalise_image(image: np.ndarray) -> np.ndarray:
    lo, hi = image.min(), image.max()
    if hi > lo:
        return (image - lo) / (hi - lo)
    return image


def _input_kind(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "kind", "") or "").lower()


def _input_layout(explanation: object) -> str:
    semantics = getattr(explanation, "semantics", None)
    input_spec = getattr(semantics, "input_spec", None)
    return str(getattr(input_spec, "layout", "") or "").upper().replace(" ", "")


def _has_image_input_metadata(explanation: object) -> bool:
    kind = _input_kind(explanation)
    if kind == "image":
        return True
    if kind:
        return False
    return _input_layout(explanation) == "NCHW"


def _display_image(ax: Any, image: np.ndarray) -> None:
    if image.ndim == 3 and image.shape[-1] == 1:
        ax.imshow(image[..., 0], cmap="gray")
        return
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)


class InputThumbnailVisualiser(BaseVisualiser):
    """Render a compact preview of the original input for report sample headers."""

    supported_scopes: ClassVar[frozenset[ExplanationScope]] = frozenset({ExplanationScope.LOCAL})
    embeds_original_input: ClassVar[bool] = False

    def __init__(self, title: str = "Input"):
        self.title = title

    def validate_explanation(
        self,
        explanation: object,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None,
    ) -> None:
        super().validate_explanation(explanation, attributions, inputs)
        if not _has_image_input_metadata(explanation):
            self._raise_incompatibility("input metadata", _input_kind(explanation), "image")
        if inputs is None:
            self._raise_incompatibility("inputs", "missing", "image input tensor")

    def visualise(
        self,
        attributions: torch.Tensor,
        inputs: torch.Tensor | None = None,
        *,
        context: VisualisationContext | None = None,
        max_samples: int = 8,
        **kwargs: Any,
    ) -> Figure:
        del attributions, kwargs
        if inputs is None:
            raise ValueError("InputThumbnailVisualiser requires `inputs` to render a thumbnail.")

        images = _to_numpy(inputs)
        if images.ndim == 3:
            images = images[np.newaxis]
        n = min(images.shape[0], max_samples)
        images = images[:n]

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        sample_names = context.sample_names if context is not None else None
        show_sample_names = context.show_sample_names if context is not None else False
        names = [] if sample_names is None else [str(name) for name in sample_names[:n]]

        for index, ax in enumerate(axes[0]):
            image = images[index]
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            image = _normalise_image(image)
            _display_image(ax, image)
            title = self.title
            if show_sample_names and index < len(names):
                title = f"{title}: {names[index]}"
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        return fig
